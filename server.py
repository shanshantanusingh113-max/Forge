#!/usr/bin/env python3
"""Forge — your own AI website builder.

Run:  python server.py        then open http://localhost:8765

No packages to install: this uses only the Python standard library.
It talks to a model through Ollama (free, runs on your computer) or any
OpenAI-compatible API, writes the files the model produces, serves a live
preview, and checks the result for problems so they can be fixed automatically.
"""
import hashlib
import hmac
import io
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from forge_prompt import system_message

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "projects")
STATIC = os.path.join(ROOT, "static")
SETTINGS_FILE = os.path.join(ROOT, "settings.json")
HOST = os.environ.get("FORGE_HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT") or os.environ.get("FORGE_PORT") or "8765")  # PORT is set by cloud hosts
PASSWORD = os.environ.get("FORGE_PASSWORD", "")

DEFAULT_SETTINGS = {
    "provider": os.environ.get("FORGE_PROVIDER", "ollama"),   # ollama | openai | demo
    "base_url": os.environ.get("FORGE_BASE_URL", "http://localhost:11434"),
    "model": os.environ.get("FORGE_MODEL", "qwen2.5-coder:7b"),
    "api_key": "",
    "temperature": 0.3,
    "num_ctx": 16384,                # context window requested from Ollama
    "auto_fix": True,
    "max_fix_attempts": 3,
}
ENV_API_KEY = os.environ.get("FORGE_API_KEY", "")  # never written to settings.json

HISTORY_MESSAGES = 12        # how many past chat messages are sent to the model
FILE_CONTEXT_CHARS = 40000   # how much of the current project is shown to the model

_lock = threading.Lock()
os.makedirs(DATA, exist_ok=True)


# ----------------------------------------------------------------- storage --

def load_settings():
    s = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            s.update(json.load(f))
    except (OSError, ValueError):
        pass
    if not s.get("api_key") and ENV_API_KEY:
        s["api_key"] = ENV_API_KEY
    return s


def save_settings(new):
    s = load_settings()
    for k in DEFAULT_SETTINGS:
        if k in new:
            s[k] = new[k]
    stored = dict(s)
    if stored.get("api_key") == ENV_API_KEY:
        stored["api_key"] = ""
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(stored, f, indent=2)
    return s


def public_settings(s):
    out = {k: v for k, v in s.items() if k != "api_key"}
    out["has_api_key"] = bool(s.get("api_key"))
    return out


def valid_pid(pid):
    return bool(re.fullmatch(r"[a-f0-9]{12}", pid or ""))


def pdir(pid):
    return os.path.join(DATA, pid)


def site_dir(pid):
    return os.path.join(DATA, pid, "site")


def read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def create_project(title="New project"):
    pid = uuid.uuid4().hex[:12]
    os.makedirs(site_dir(pid), exist_ok=True)
    now = time.time()
    meta = {"id": pid, "title": title, "created": now, "updated": now}
    write_json(os.path.join(pdir(pid), "meta.json"), meta)
    write_json(os.path.join(pdir(pid), "chat.json"), [])
    return meta


def list_projects():
    out = []
    for pid in os.listdir(DATA):
        if valid_pid(pid):
            meta = read_json(os.path.join(pdir(pid), "meta.json"), None)
            if meta:
                out.append(meta)
    return sorted(out, key=lambda m: m.get("updated", 0), reverse=True)


TEXT_EXT = {".html", ".htm", ".css", ".js", ".mjs", ".json", ".svg", ".txt", ".md", ".xml", ".webmanifest"}


def read_site(pid):
    files = {}
    base = site_dir(pid)
    for dirpath, _, names in os.walk(base):
        for n in sorted(names):
            full = os.path.join(dirpath, n)
            rel = os.path.relpath(full, base).replace(os.sep, "/")
            if os.path.splitext(n)[1].lower() in TEXT_EXT:
                try:
                    with open(full, encoding="utf-8") as f:
                        files[rel] = f.read()
                except (OSError, UnicodeDecodeError):
                    files[rel] = None
            else:
                files[rel] = None
    return dict(sorted(files.items(), key=lambda kv: (kv[0] != "index.html", kv[0])))


def safe_rel_path(p):
    p = (p or "").strip().replace("\\", "/").lstrip("./")
    while p.startswith("/"):
        p = p[1:]
    if not p or ".." in p.split("/") or not re.fullmatch(r"[A-Za-z0-9_\-./ ]+", p):
        return None
    return p


def write_site_file(pid, rel, content):
    full = os.path.join(site_dir(pid), *rel.split("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)


# ---------------------------------------------------------- output parsing --

FILE_RE = re.compile(r"<file\s+path\s*=\s*[\"']([^\"']+)[\"']\s*>[ \t]*\n?(.*?)</file>", re.S)
FENCE_RE = re.compile(r"```(?:html)?\s*\n(.*?)```", re.S | re.I)


def strip_fence(body):
    s = body.strip("\n")
    m = re.fullmatch(r"\s*```[\w+-]*\s*\n(.*?)\n?```\s*", s, re.S)
    return (m.group(1) if m else s).rstrip() + "\n"


def extract_files(text):
    files = []
    for m in FILE_RE.finditer(text):
        rel = safe_rel_path(m.group(1))
        if rel:
            files.append((rel, strip_fence(m.group(2))))
    if not files:  # fallback: a model that ignored the format but wrote a full page
        for m in FENCE_RE.finditer(text):
            if re.search(r"<!doctype html|<html", m.group(1), re.I):
                files.append(("index.html", m.group(1).rstrip() + "\n"))
                break
    return files


def summarize_for_history(text):
    """Replace file bodies in old assistant messages; the current files are sent separately."""
    def repl(m):
        return f'[wrote {m.group(1)}]'
    return FILE_RE.sub(repl, text)


# ------------------------------------------------------------------ checks --

REF_RE = re.compile(r"""(?:src|href)\s*=\s*["']([^"'#?]+)""", re.I)


def check_site(pid, reply_text=""):
    """Static checks run after every generation. Runtime errors are caught in the preview."""
    issues = []
    files = read_site(pid)
    opened = len(re.findall(r"<file\s+path", reply_text))
    closed = len(FILE_RE.findall(reply_text))
    if opened > closed:
        issues.append("A file was cut off before </file> (the model hit its output limit). "
                      "Rewrite that file completely, keeping it shorter if needed.")
    if files and "index.html" not in files:
        issues.append("There is no index.html entry page.")
    for path, content in files.items():
        if content is None:
            continue
        if path.endswith((".html", ".htm")):
            here = os.path.dirname(path)
            for ref in set(REF_RE.findall(content)):
                r = ref.strip()
                if re.match(r"^(https?:|//|data:|mailto:|tel:|javascript:|\{|\$)", r, re.I) or not r:
                    continue
                target = os.path.normpath(os.path.join(here, r)).replace(os.sep, "/").lstrip("/")
                if r.startswith("/"):
                    target = r.lstrip("/")
                if target.endswith("/"):
                    target += "index.html"
                if target not in files and target + "/index.html" not in files:
                    issues.append(f"{path} links to '{r}', but that file does not exist.")
    node = shutil.which("node")
    if node:
        for path, content in files.items():
            if path.endswith(".js") and content is not None:
                full = os.path.join(site_dir(pid), *path.split("/"))
                try:
                    r = subprocess.run([node, "--check", full], capture_output=True, text=True, timeout=15)
                except (OSError, subprocess.TimeoutExpired):
                    continue
                if r.returncode != 0 and "import statement" not in r.stderr and "export" not in r.stderr:
                    lines = r.stderr.strip().splitlines()
                    msg = next((l.strip() for l in lines if "Error" in l and not l.strip().startswith("at ")), "")
                    where = re.search(r":(\d+)\s*$", lines[0]) if lines else None
                    code = lines[1].strip() if len(lines) > 1 else ""
                    loc = f" line {where.group(1)}" if where else ""
                    issues.append(f"{msg or 'Syntax error'} in {path}{loc}" + (f": `{code[:120]}`" if code else ""))
    return issues


# ------------------------------------------------------------------- model --

class ModelError(Exception):
    pass


def build_messages(pid, history):
    msgs = [{"role": "system", "content": system_message(read_site(pid), FILE_CONTEXT_CHARS)}]
    recent = history[-HISTORY_MESSAGES:]
    while recent and recent[0]["role"] != "user":   # chat templates expect a user turn first
        recent = recent[1:]
    for m in recent:
        content = m["content"]
        if m["role"] == "assistant":
            content = summarize_for_history(content) or "(no reply)"
        msgs.append({"role": m["role"], "content": content})
    return msgs


def _open(req, timeout=900):
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        if e.code == 404 and "not found" in detail.lower():
            raise ModelError(f"The model was not found. If you use Ollama, run:  ollama pull <model name>\n\nDetails: {detail}")
        if e.code in (401, 403):
            raise ModelError("The API rejected the key. Check the API key in Settings.")
        raise ModelError(f"The model server returned an error ({e.code}): {detail}")
    except urllib.error.URLError as e:
        raise ModelError(f"Could not reach the model server at {req.full_url.rsplit('/', 2)[0]}. "
                         f"Is it running? (For Ollama, open the Ollama app or run:  ollama serve)\n\nDetails: {e.reason}")


def stream_model(settings, messages):
    prov = settings.get("provider", "ollama")
    base = (settings.get("base_url") or "").rstrip("/")
    if prov == "demo":
        yield from demo_stream(messages)
        return
    if prov == "ollama":
        body = {
            "model": settings["model"], "messages": messages, "stream": True,
            "options": {"temperature": float(settings.get("temperature", 0.3)),
                        "num_ctx": int(settings.get("num_ctx", 16384))},
        }
        req = urllib.request.Request(base + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with _open(req) as r:
            for raw in r:
                line = raw.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("error"):
                    raise ModelError(d["error"])
                c = (d.get("message") or {}).get("content")
                if c:
                    yield c
                if d.get("done"):
                    break
        return
    # OpenAI-compatible: OpenAI, OpenRouter, Groq, Together, LM Studio, vLLM, llama.cpp server ...
    body = {"model": settings["model"], "messages": messages, "stream": True,
            "temperature": float(settings.get("temperature", 0.3))}
    headers = {"Content-Type": "application/json"}
    if settings.get("api_key"):
        headers["Authorization"] = "Bearer " + settings["api_key"]
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers=headers)
    with _open(req) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                d = json.loads(data)
            except ValueError:
                continue
            if d.get("error"):
                raise ModelError(str(d["error"]))
            choices = d.get("choices") or []
            if choices:
                c = (choices[0].get("delta") or {}).get("content")
                if c:
                    yield c


def list_models(settings):
    prov = settings.get("provider")
    base = (settings.get("base_url") or "").rstrip("/")
    if prov == "demo":
        return {"ok": True, "models": ["demo"]}
    try:
        if prov == "ollama":
            with _open(urllib.request.Request(base + "/api/tags"), timeout=5) as r:
                d = json.load(r)
            return {"ok": True, "models": [m["name"] for m in d.get("models", [])]}
        headers = {"Authorization": "Bearer " + settings["api_key"]} if settings.get("api_key") else {}
        with _open(urllib.request.Request(base + "/models", headers=headers), timeout=8) as r:
            d = json.load(r)
        return {"ok": True, "models": sorted(m.get("id", "") for m in d.get("data", []))[:300]}
    except (ModelError, ValueError, OSError) as e:
        return {"ok": False, "error": str(e), "models": []}


# ------------------------------------------------------------- demo model --
# Lets you try the whole app with no model installed. It builds one starter
# site, deliberately ships a small bug, and then fixes it when the preview
# reports the error — so you can watch the auto-fix loop work.

DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <header class="nav">
    <a class="brand" href="#">{title}</a>
    <button class="menu-btn" aria-label="Open menu" aria-expanded="false">Menu</button>
    <nav class="links">
      <a href="#features">Features</a>
      <a href="#pricing">Pricing</a>
      <a href="#contact" class="btn btn-small">Get started</a>
    </nav>
  </header>
  <main>
    <section class="hero">
      <p class="eyebrow">Built with Forge</p>
      <h1>{headline}</h1>
      <p class="lead">A starter site generated in demo mode. Connect a real model in Settings and describe anything you like — Forge will build it, check it, and fix its own mistakes.</p>
      <div class="actions">
        <a href="#pricing" class="btn">See pricing</a>
        <a href="#features" class="btn btn-ghost">Learn more</a>
      </div>
    </section>
    <section id="features" class="grid">
      <article class="card"><h3>Fast</h3><p>Loads in under a second on any phone, with no build step.</p></article>
      <article class="card"><h3>Responsive</h3><p>Looks right from a 360px phone to a wide desktop monitor.</p></article>
      <article class="card"><h3>Accessible</h3><p>Semantic markup, real labels and visible focus states.</p></article>
    </section>
    <section id="pricing" class="pricing">
      <h2>Simple pricing</h2>
      <div class="toggle"><button class="active" data-period="month">Monthly</button><button data-period="year">Yearly</button></div>
      <div class="price"><span class="amount" data-month="₹499" data-year="₹4,990">₹499</span><span class="per">/ month</span></div>
    </section>
    <section id="contact" class="contact">
      <h2>Stay in touch</h2>
      <form class="signup">
        <label for="email">Email</label>
        <div class="row"><input id="email" type="email" placeholder="you@example.com" required><button class="btn" type="submit">Join</button></div>
        <p class="note" role="status"></p>
      </form>
    </section>
  </main>
  <footer>© 2026 {title}</footer>
  <script src="script.js"></script>
</body>
</html>
"""

DEMO_CSS = """:root {
  --bg: #faf8f5; --surface: #ffffff; --text: #1d1b19; --muted: #6b655e;
  --accent: #2f6f5e; --accent-ink: #ffffff; --line: #e8e3dc;
  --radius: 14px; --shadow: 0 1px 2px rgba(0,0,0,.04), 0 8px 24px rgba(0,0,0,.06);
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #141312; --surface: #1d1c1a; --text: #f1ede7; --muted: #a39d95; --accent: #6cc3a8; --accent-ink: #0d1f1a; --line: #2c2a27; }
}
* { box-sizing: border-box; margin: 0; }
body { font-family: Inter, system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }
h1, h2 { font-family: Fraunces, Georgia, serif; line-height: 1.15; }
.nav { display: flex; align-items: center; justify-content: space-between; padding: 16px 24px; max-width: 1100px; margin: 0 auto; }
.brand { font-weight: 600; color: var(--text); text-decoration: none; }
.links { display: flex; gap: 24px; align-items: center; }
.links a { color: var(--muted); text-decoration: none; }
.links a:hover { color: var(--text); }
.menu-btn { display: none; background: none; border: 1px solid var(--line); border-radius: 8px; padding: 6px 12px; color: var(--text); }
.btn { display: inline-block; background: var(--accent); color: var(--accent-ink) !important; padding: 12px 20px; border-radius: 10px; text-decoration: none; border: 0; font: inherit; font-weight: 600; cursor: pointer; transition: transform .15s, opacity .15s; }
.btn:hover { transform: translateY(-1px); opacity: .92; }
.btn:focus-visible, input:focus-visible, button:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
.btn-small { padding: 8px 14px; }
.btn-ghost { background: transparent; color: var(--text) !important; border: 1px solid var(--line); }
main { max-width: 1100px; margin: 0 auto; padding: 0 24px; }
.hero { padding: 72px 0 56px; max-width: 720px; }
.eyebrow { color: var(--accent); font-weight: 600; font-size: 14px; letter-spacing: .04em; text-transform: uppercase; }
.hero h1 { font-size: clamp(36px, 6vw, 60px); margin: 12px 0 16px; }
.lead { color: var(--muted); font-size: 18px; }
.actions { display: flex; gap: 12px; margin-top: 28px; flex-wrap: wrap; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; padding: 24px 0 64px; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 24px; box-shadow: var(--shadow); }
.card h3 { margin-bottom: 6px; }
.card p { color: var(--muted); }
.pricing, .contact { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 40px 24px; text-align: center; margin-bottom: 24px; }
.toggle { display: inline-flex; border: 1px solid var(--line); border-radius: 999px; padding: 4px; margin: 20px 0; }
.toggle button { border: 0; background: none; padding: 8px 16px; border-radius: 999px; font: inherit; color: var(--muted); cursor: pointer; }
.toggle button.active { background: var(--accent); color: var(--accent-ink); }
.amount { font-family: Fraunces, serif; font-size: 48px; }
.per { color: var(--muted); }
.signup { max-width: 420px; margin: 20px auto 0; text-align: left; }
.signup label { font-weight: 500; font-size: 14px; }
.row { display: flex; gap: 8px; margin-top: 6px; }
.row input { flex: 1; padding: 12px; border: 1px solid var(--line); border-radius: 10px; font: inherit; background: var(--bg); color: var(--text); }
.note { margin-top: 8px; color: var(--accent); min-height: 1.6em; }
footer { text-align: center; color: var(--muted); padding: 32px; font-size: 14px; }
@media (max-width: 640px) {
  .menu-btn { display: block; }
  .links { display: none; position: absolute; top: 60px; right: 16px; flex-direction: column; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 16px; box-shadow: var(--shadow); }
  .links.open { display: flex; }
  .row { flex-direction: column; }
}
"""

DEMO_JS_BUGGY = """document.addEventListener('DOMContentLoaded', () => {
  initMenu();
  initPricingToggle();
  initSignup();
});

function initMenu() {
  const btn = document.querySelector('.menu-btn');
  const links = document.querySelector('.links');
  if (!btn || !links) return;
  btn.addEventListener('click', () => {
    const open = links.classList.toggle('open');
    btn.setAttribute('aria-expanded', String(open));
  });
}

function initSignup() {
  const form = document.querySelector('.signup');
  if (!form) return;
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    form.querySelector('.note').textContent = 'Thanks! You are on the list.';
    form.reset();
  });
}
"""

DEMO_JS_FIXED = DEMO_JS_BUGGY.replace("function initSignup() {", """function initPricingToggle() {
  const buttons = document.querySelectorAll('.toggle button');
  const amount = document.querySelector('.amount');
  const per = document.querySelector('.per');
  if (!buttons.length || !amount || !per) return;
  buttons.forEach((b) => b.addEventListener('click', () => {
    buttons.forEach((x) => x.classList.toggle('active', x === b));
    const period = b.dataset.period;
    amount.textContent = amount.dataset[period];
    per.textContent = period === 'year' ? '/ year' : '/ month';
  }));
}

function initSignup() {""")


def _chunks(text, n=24):
    for i in range(0, len(text), n):
        yield text[i:i + n]
        time.sleep(0.004)


def demo_stream(messages):
    last = messages[-1]["content"] if messages else ""
    has_files = "The project is empty" not in messages[0]["content"]
    if last.startswith("[auto-fix]") and "initPricingToggle" in last:
        text = ("The page calls `initPricingToggle()` on load, but that function was never defined, "
                "so the script stopped there. I've added it.\n\n"
                f'<file path="script.js">\n{DEMO_JS_FIXED}</file>\n\n'
                "- Added the missing `initPricingToggle()` that switches between monthly and yearly prices\n"
                "- The menu and sign-up form now run too, since the script no longer crashes first\n\n"
                "Next idea: add a testimonials section.")
    elif last.startswith("[auto-fix]"):
        text = "I checked the reported problems against the current files; there's nothing left to change."
    elif not has_files:
        stop = {"a", "an", "the", "for", "of", "with", "and", "to", "in", "on", "my", "me", "page", "landing",
                "site", "website", "web", "make", "build", "create", "personal", "app", "that", "simple"}
        words = [w for w in re.findall(r"[A-Za-z]+", last) if w.lower() not in stop]
        title = " ".join(w.capitalize() for w in words[:3]) if words else "My Site"
        headline = (last.strip().rstrip(".!?")[:60] or "Build something people love").capitalize()
        html = DEMO_HTML.replace("{title}", title).replace("{headline}", headline)
        text = (f"I'll build a responsive starter site for **{title}** with a hero, features, "
                "a pricing toggle and a sign-up form.\n\n"
                f'<file path="index.html">\n{html}</file>\n\n'
                f'<file path="style.css">\n{DEMO_CSS}</file>\n\n'
                f'<file path="script.js">\n{DEMO_JS_BUGGY}</file>\n\n'
                "- Mobile-first layout with a collapsible menu\n"
                "- Light and dark themes\n"
                "- Pricing toggle and a working sign-up form\n\n"
                "Next idea: add a testimonials section.")
    else:
        text = ("You're in **demo mode**, so I can only build the starter site. "
                "Open **Settings** and connect a real model (free with Ollama) to make any change you want.")
    yield from _chunks(text)


# ------------------------------------------------------------------- server --

PREVIEW_PROBE = """<script>(function(){
function send(k,m){try{parent.postMessage({forge:1,kind:k,msg:String(m).slice(0,600)},'*')}catch(e){}}
window.addEventListener('error',function(e){var t=e.target;
 if(t&&t!==window&&(t.src||t.href)){var u=t.src||t.href;
  if(u.indexOf(location.origin)!==0&&t.tagName!=='SCRIPT')return;
  send('resource','Failed to load '+u);return;}
 var where=e.filename?' ('+e.filename.split('/').pop()+(e.lineno?' line '+e.lineno:'')+')':'';
 send('runtime',(e.message||'Script error')+where);},true);
window.addEventListener('unhandledrejection',function(e){var r=e.reason;send('runtime','Unhandled promise rejection: '+(r&&r.message||r));});
var ce=console.error;console.error=function(){try{send('console',[].map.call(arguments,function(a){return a&&a.message||String(a)}).join(' '))}catch(_){}return ce.apply(console,arguments);};
window.addEventListener('load',function(){send('loaded','')});
})();</script>"""


def inject_probe(html):
    m = re.search(r"<head[^>]*>", html, re.I)
    if m:
        return html[:m.end()] + PREVIEW_PROBE + html[m.end():]
    return PREVIEW_PROBE + html


SESSION_TOKEN = hmac.new(PASSWORD.encode(), b"forge-session-v1", hashlib.sha256).hexdigest() if PASSWORD else ""

LOGIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Sign in · Forge</title>
<style>
:root{--bg:#f6f4f0;--card:#fff;--text:#1c1a17;--muted:#77716a;--line:#d3cec4;--accent:#e0912b;--danger:#c2412d}
@media (prefers-color-scheme:dark){:root{--bg:#171614;--card:#1d1c1a;--text:#ece8e2;--muted:#9c968e;--line:#3a3733;--accent:#f0a640;--danger:#ef7a62}}
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;padding:16px}
form{width:100%;max-width:340px;background:var(--card);border:1px solid var(--line);border-radius:16px;padding:28px}
.logo{width:40px;height:40px;border-radius:11px;background:var(--accent);display:grid;place-items:center;margin-bottom:14px}
h1{font:500 24px Georgia,serif;margin:0 0 4px}p{color:var(--muted);margin:0 0 20px}
input{width:100%;padding:11px 12px;border:1px solid var(--line);border-radius:10px;background:var(--bg);color:var(--text);font:inherit}
input:focus{outline:2px solid var(--accent);outline-offset:1px}
button{width:100%;margin-top:12px;padding:11px;border:0;border-radius:10px;background:var(--accent);color:#1a1816;font-family:inherit;font-size:15px;font-weight:600;cursor:pointer}
.err{color:var(--danger);font-size:13px;min-height:20px;margin-top:8px}
</style></head><body>
<form id="f"><div class="logo"><svg width="22" height="22" viewBox="0 0 24 24"><path d="M13.6 2.5 5.2 13.4h5.4L9.4 21.5l8.4-10.9h-5.4z" fill="#1a1816"/></svg></div>
<h1>Forge</h1><p>Enter the password to continue.</p>
<input id="pw" type="password" autocomplete="current-password" aria-label="Password" placeholder="Password" autofocus required>
<button type="submit">Sign in</button><div class="err" id="err" role="alert"></div></form>
<script>
document.getElementById('f').onsubmit=async function(e){e.preventDefault();
var b=this.querySelector('button');b.disabled=true;document.getElementById('err').textContent='';
var r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:document.getElementById('pw').value})});
if(r.ok){location.reload();}else{document.getElementById('err').textContent='Wrong password.';b.disabled=false;}};
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "Forge/1.0"

    def log_message(self, fmt, *args):
        if os.environ.get("FORGE_DEBUG"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # helpers
    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, data, ctype, code=200, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def not_found(self):
        self.send_bytes(b"Not found", "text/plain", 404)

    def body_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n))
        except ValueError:
            return {}

    # password protection (only when FORGE_PASSWORD is set)
    def authed(self):
        if not PASSWORD:
            return True
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == "forge_session" and hmac.compare_digest(v, SESSION_TOKEN):
                return True
        return False

    def deny(self, path):
        if path.startswith("/api/"):
            return self.send_json({"error": "Not signed in"}, 401)
        return self.send_bytes(LOGIN_PAGE.encode(), "text/html; charset=utf-8", 401)

    def login(self):
        pw = str(self.body_json().get("password") or "")
        if not PASSWORD or not hmac.compare_digest(pw.encode(), PASSWORD.encode()):
            time.sleep(1.0)  # slow down password guessing
            return self.send_json({"ok": False}, 401)
        secure = "; Secure" if self.headers.get("X-Forwarded-Proto") == "https" else ""
        cookie = f"forge_session={SESSION_TOKEN}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000{secure}"
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(body)

    # routing
    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        parts = [p for p in path.split("/") if p]
        if path == "/healthz":  # for the host's uptime check
            return self.send_bytes(b"ok", "text/plain")
        if not self.authed():
            return self.deny(path)
        if path in ("/", "/index.html"):
            return self.serve_file(os.path.join(STATIC, "index.html"))
        if parts[:1] == ["static"] and len(parts) > 1:
            rel = safe_rel_path("/".join(parts[1:]))
            return self.serve_file(os.path.join(STATIC, rel)) if rel else self.not_found()
        if parts[:1] == ["preview"] and len(parts) >= 2 and valid_pid(parts[1]):
            return self.serve_preview(parts[1], "/".join(parts[2:]), path.endswith("/"))
        if parts[:2] == ["api", "settings"]:
            return self.send_json(public_settings(load_settings()))
        if parts[:2] == ["api", "health"]:
            return self.send_json(list_models(load_settings()))
        if parts[:2] == ["api", "projects"]:
            if len(parts) == 2:
                return self.send_json(list_projects())
            pid = parts[2]
            if not valid_pid(pid) or not os.path.isdir(pdir(pid)):
                return self.not_found()
            if len(parts) == 3:
                return self.send_json({
                    "meta": read_json(os.path.join(pdir(pid), "meta.json"), {}),
                    "messages": read_json(os.path.join(pdir(pid), "chat.json"), []),
                    "files": read_site(pid),
                })
            if parts[3] == "zip":
                return self.send_zip(pid)
        return self.not_found()

    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        parts = [p for p in path.split("/") if p]
        if parts == ["api", "login"]:
            return self.login()
        if not self.authed():
            return self.deny(path)
        if parts[:2] == ["api", "settings"]:
            return self.send_json(public_settings(save_settings(self.body_json())))
        if parts[:2] == ["api", "health"]:
            s = load_settings()
            s.update({k: v for k, v in self.body_json().items() if k in DEFAULT_SETTINGS and (k != "api_key" or v)})
            return self.send_json(list_models(s))
        if parts == ["api", "projects"]:
            return self.send_json(create_project(self.body_json().get("title") or "New project"))
        if len(parts) == 4 and parts[:2] == ["api", "projects"] and valid_pid(parts[2]):
            pid = parts[2]
            if not os.path.isdir(pdir(pid)):
                return self.not_found()
            if parts[3] == "chat":
                return self.chat(pid, self.body_json())
            if parts[3] == "rename":
                meta = read_json(os.path.join(pdir(pid), "meta.json"), {})
                meta["title"] = (self.body_json().get("title") or meta.get("title", ""))[:80]
                write_json(os.path.join(pdir(pid), "meta.json"), meta)
                return self.send_json(meta)
        return self.not_found()

    def do_DELETE(self):
        if not self.authed():
            return self.deny(urlparse(self.path).path)
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        if len(parts) == 3 and parts[:2] == ["api", "projects"] and valid_pid(parts[2]):
            shutil.rmtree(pdir(parts[2]), ignore_errors=True)
            return self.send_json({"ok": True})
        return self.not_found()

    # handlers
    def serve_file(self, full):
        if not full or not os.path.isfile(full):
            return self.not_found()
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        with open(full, "rb") as f:
            self.send_bytes(f.read(), ctype)

    def serve_preview(self, pid, rel, trailing):
        if rel == "" or trailing:
            rel = (rel + "/index.html").lstrip("/")
        rel = safe_rel_path(rel)
        if not rel:
            return self.not_found()
        full = os.path.join(site_dir(pid), *rel.split("/"))
        if os.path.isdir(full):
            full = os.path.join(full, "index.html")
        if not os.path.isfile(full):
            if rel == "index.html":
                page = ("<!doctype html><meta name=viewport content='width=device-width'>"
                        "<body style='font:15px system-ui;color:#8a847c;display:grid;place-items:center;height:100vh;margin:0'>"
                        "Your site will appear here.</body>")
                return self.send_bytes(page.encode(), "text/html; charset=utf-8")
            return self.not_found()
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        with open(full, "rb") as f:
            data = f.read()
        if ctype == "text/html":
            data = inject_probe(data.decode("utf-8", "replace")).encode()
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json", "image/svg+xml"):
            ctype += "; charset=utf-8"
        self.send_bytes(data, ctype)

    def send_zip(self, pid):
        meta = read_json(os.path.join(pdir(pid), "meta.json"), {})
        buf = io.BytesIO()
        base = site_dir(pid)
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for dirpath, _, names in os.walk(base):
                for n in names:
                    full = os.path.join(dirpath, n)
                    z.write(full, os.path.relpath(full, base))
        name = re.sub(r"[^A-Za-z0-9_-]+", "-", meta.get("title", "site")).strip("-").lower() or "site"
        self.send_bytes(buf.getvalue(), "application/zip",
                        extra={"Content-Disposition": f'attachment; filename="{name}.zip"'})

    def chat(self, pid, body):
        text_in = (body.get("message") or "").strip()
        if not text_in:
            return self.send_json({"error": "empty message"}, 400)
        auto = body.get("kind") == "fix"
        chat_path = os.path.join(pdir(pid), "chat.json")
        meta_path = os.path.join(pdir(pid), "meta.json")
        with _lock:
            history = read_json(chat_path, [])
            history.append({"role": "user", "content": text_in, "auto": auto, "ts": time.time()})
            write_json(chat_path, history)
            meta = read_json(meta_path, {})
            if meta.get("title") in (None, "", "New project") and not auto:
                meta["title"] = re.sub(r"\s+", " ", text_in)[:48].strip().rstrip(".")
            meta["updated"] = time.time()
            write_json(meta_path, meta)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def emit(obj):
            self.wfile.write((json.dumps(obj) + "\n").encode())
            self.wfile.flush()

        settings = load_settings()
        reply, error, stopped = [], None, False
        try:
            emit({"t": "start", "title": meta.get("title")})
            for chunk in stream_model(settings, build_messages(pid, history)):
                reply.append(chunk)
                emit({"t": "delta", "v": chunk})
        except (BrokenPipeError, ConnectionResetError):
            stopped = True
        except ModelError as e:
            error = str(e)
        except Exception as e:  # keep the app alive whatever the model server does
            error = f"{type(e).__name__}: {e}"

        full = "".join(reply)
        written = []
        for rel, content in extract_files(full):
            write_site_file(pid, rel, content)
            written.append(rel)
        issues = check_site(pid, full) if (written or full) and not error else []

        with _lock:
            history = read_json(chat_path, [])
            msg = {"role": "assistant", "content": full, "files": written, "issues": issues, "ts": time.time()}
            if error:
                msg["error"] = error
            if stopped:
                msg["stopped"] = True
            history.append(msg)
            write_json(chat_path, history)
            meta = read_json(meta_path, {})
            meta["updated"] = time.time()
            write_json(meta_path, meta)
        if stopped:
            return
        try:
            if error:
                emit({"t": "error", "v": error})
            emit({"t": "done", "files": written, "issues": issues})
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("application/javascript", ".mjs")
    mimetypes.add_type("text/css", ".css")
    mimetypes.add_type("image/svg+xml", ".svg")
    public = HOST not in ("127.0.0.1", "localhost", "::1")
    if public and not PASSWORD and os.environ.get("FORGE_ALLOW_OPEN") != "1":
        sys.exit("\n  Refusing to start: Forge is reachable from other machines (FORGE_HOST="
                 f"{HOST}) but has no password.\n  Set FORGE_PASSWORD, or set FORGE_ALLOW_OPEN=1 "
                 "if you really want it open.\n")
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://localhost:{PORT}"
    s = load_settings()
    print(f"\n  Forge is running at {url}" + ("   (password protected)" if PASSWORD else ""))
    print(f"  Model: {s['provider']} / {s['model']}   (change it in Settings)")
    print("  Press Ctrl+C to stop.\n")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")


if __name__ == "__main__":
    main()
