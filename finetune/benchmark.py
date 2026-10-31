#!/usr/bin/env python3
"""Compare models on the same website prompts, so you know if fine-tuning helped.

    python benchmark.py                                  # model from Forge settings
    python benchmark.py --model forge-coder              # your fine-tuned Ollama model
    python benchmark.py --model qwen2.5-coder:7b --keep  # keep the sites to open in Forge

For each prompt it measures speed, whether usable files came out, and how many
problems Forge's checks found (missing files, JavaScript syntax errors, cut-off
output). Lower problems and a higher pass rate = better model.
Runtime errors (found in the live preview) are not measured here.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import server  # noqa: E402

PROMPTS = [
    "A landing page for a specialty coffee roaster with a menu section and an order form",
    "A personal portfolio for a UI designer with a filterable project grid",
    "A habit tracker that saves to localStorage, with streaks and a weekly view",
    "A pricing page for a SaaS product with a monthly/yearly toggle and an FAQ accordion",
    "A restaurant website with a hero, menu tabs (starters, mains, desserts) and opening hours",
    "A weather dashboard UI with a search box and a 5-day forecast using sample data",
    "A multi-step signup form with validation and a progress indicator",
    "A blog homepage with article cards, categories, and a dark mode toggle",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider")
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--prompts", help="text file with one prompt per line")
    ap.add_argument("--keep", action="store_true", help="save the generated sites as Forge projects")
    args = ap.parse_args()

    settings = server.load_settings()
    for k, v in (("provider", args.provider), ("base_url", args.base_url), ("model", args.model)):
        if v:
            settings[k] = v
    prompts = PROMPTS
    if args.prompts:
        with open(args.prompts, encoding="utf-8") as f:
            prompts = [l.strip() for l in f if l.strip()]

    if not args.keep:
        server.DATA = tempfile.mkdtemp(prefix="forge-bench-")
    print(f"Benchmarking {settings['provider']} / {settings['model']} on {len(prompts)} prompts\n")
    rows = []
    for i, prompt in enumerate(prompts, 1):
        meta = server.create_project(f"[bench] {prompt[:40]}")
        pid = meta["id"]
        history = [{"role": "user", "content": prompt}]
        t0, text, err = time.time(), "", None
        try:
            text = "".join(server.stream_model(settings, server.build_messages(pid, history)))
        except Exception as e:
            err = str(e).splitlines()[0]
        secs = time.time() - t0
        files = server.extract_files(text)
        for rel, content in files:
            server.write_site_file(pid, rel, content)
        issues = server.check_site(pid, text) if files else []
        passed = bool(files) and not issues and not err and any(p == "index.html" for p, _ in files)
        if args.keep:
            server.write_json(os.path.join(server.pdir(pid), "chat.json"), history + [
                {"role": "assistant", "content": text, "files": [p for p, _ in files], "issues": issues}])
        rows.append({"prompt": prompt, "seconds": round(secs, 1), "files": len(files),
                     "chars": len(text), "issues": issues, "error": err, "passed": passed})
        status = "PASS" if passed else "FAIL"
        print(f"{i:>2}. {status}  {secs:6.1f}s  {len(files)} files  {len(issues)} problems  {prompt[:60]}")
        for p in issues[:3]:
            print(f"        - {p[:110]}")
        if err:
            print(f"        ! {err[:110]}")

    n = len(rows)
    passed = sum(r["passed"] for r in rows)
    print(f"\nPass rate: {passed}/{n} ({100 * passed // max(n, 1)}%)   "
          f"avg time: {sum(r['seconds'] for r in rows) / max(n, 1):.1f}s   "
          f"total problems: {sum(len(r['issues']) for r in rows)}")
    out = os.path.join(HERE, f"benchmark-{settings['model'].replace('/', '_').replace(':', '_')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"settings": {k: settings[k] for k in ("provider", "model")}, "results": rows}, f, indent=2)
    print(f"Saved details to {out}")
    if not args.keep:
        shutil.rmtree(server.DATA, ignore_errors=True)
    else:
        print("The sites are in Forge's project list (titles start with [bench]).")


if __name__ == "__main__":
    main()
