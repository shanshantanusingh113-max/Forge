#!/usr/bin/env python3
"""Build a training set (train.jsonl) for fine-tuning your Forge model.

Two sources, combined:

1. Your own Forge history (../projects). Every reply that wrote files AND passed
   the preview checks (no auto-fix was needed afterwards) becomes an example.
   Successful auto-fixes are included too, which teaches the model to repair code.
   The project's files are replayed turn by turn, so each example shows the model
   exactly what the project looked like at that moment.

2. Hand-made examples (./examples/<name>/): a prompt.txt, a site/ folder with the
   ideal files, and optionally reply_intro.txt / reply_summary.txt.

Usage:
    python build_dataset.py                      # writes data/train.jsonl and data/eval.jsonl
    python build_dataset.py --no-history         # only hand-made examples
    python build_dataset.py --stronger-model-only gpt-4o   (see README: distillation)
"""
import argparse
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from forge_prompt import system_message  # noqa: E402

FILE_RE = re.compile(r"<file\s+path\s*=\s*[\"']([^\"']+)[\"']\s*>[ \t]*\n?(.*?)</file>", re.S)
TEXT_EXT = {".html", ".htm", ".css", ".js", ".mjs", ".json", ".svg", ".txt", ".md", ".xml"}


def file_blocks(files):
    return "\n\n".join(f'<file path="{p}">\n{c.rstrip()}\n</file>' for p, c in files.items())


def from_history(projects_dir, max_context):
    samples, skipped = [], 0
    if not os.path.isdir(projects_dir):
        return samples, skipped
    for pid in sorted(os.listdir(projects_dir)):
        chat_path = os.path.join(projects_dir, pid, "chat.json")
        if not os.path.isfile(chat_path):
            continue
        with open(chat_path, encoding="utf-8") as f:
            chat = json.load(f)
        state = {}  # files as they were before each turn
        for i, m in enumerate(chat):
            if m.get("role") != "assistant":
                continue
            prev = chat[i - 1] if i > 0 else None
            nxt = chat[i + 1] if i + 1 < len(chat) else None
            written = {p: c for p, c in FILE_RE.findall(m.get("content", ""))}
            ok = (
                prev is not None and prev.get("role") == "user"
                and written and not m.get("error") and not m.get("stopped")
                and not m.get("issues")
                and not (nxt and nxt.get("role") == "user" and nxt.get("auto"))
            )
            if ok:
                samples.append({"messages": [
                    {"role": "system", "content": system_message(dict(state), max_context)},
                    {"role": "user", "content": prev["content"]},
                    {"role": "assistant", "content": m["content"].strip()},
                ], "source": f"history:{pid}"})
            elif written:
                skipped += 1
            for p, c in written.items():  # replay this turn's files
                state[p.strip().lstrip("./")] = c.strip("\n") + "\n"
    return samples, skipped


def from_examples(examples_dir):
    samples = []
    if not os.path.isdir(examples_dir):
        return samples
    for name in sorted(os.listdir(examples_dir)):
        ex = os.path.join(examples_dir, name)
        prompt_file, site = os.path.join(ex, "prompt.txt"), os.path.join(ex, "site")
        if not (os.path.isfile(prompt_file) and os.path.isdir(site)):
            continue
        files = {}
        for dirpath, _, names in os.walk(site):
            for n in sorted(names):
                if os.path.splitext(n)[1].lower() in TEXT_EXT:
                    full = os.path.join(dirpath, n)
                    rel = os.path.relpath(full, site).replace(os.sep, "/")
                    with open(full, encoding="utf-8") as f:
                        files[rel] = f.read()
        files = dict(sorted(files.items(), key=lambda kv: (kv[0] != "index.html", kv[0])))

        def read(fn, default):
            p = os.path.join(ex, fn)
            return open(p, encoding="utf-8").read().strip() if os.path.isfile(p) else default

        intro = read("reply_intro.txt", "Here's the complete site.")
        summary = read("reply_summary.txt", "- Built the complete site as requested")
        with open(prompt_file, encoding="utf-8") as f:
            prompt = f.read().strip()
        samples.append({"messages": [
            {"role": "system", "content": system_message({})},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"{intro}\n\n{file_blocks(files)}\n\n{summary}"},
        ], "source": f"example:{name}"})
    return samples


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects", default=os.path.join(HERE, "..", "projects"))
    ap.add_argument("--examples", default=os.path.join(HERE, "examples"))
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--no-history", action="store_true")
    ap.add_argument("--eval-fraction", type=float, default=0.05)
    ap.add_argument("--max-context", type=int, default=24000, help="chars of project files shown per example")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    hist, skipped = ([], 0) if args.no_history else from_history(args.projects, args.max_context)
    exs = from_examples(args.examples)
    data = hist + exs
    if not data:
        sys.exit("No examples found. Use Forge to build some sites, or add folders to finetune/examples/.")

    random.Random(args.seed).shuffle(data)
    n_eval = int(len(data) * args.eval_fraction) if len(data) >= 20 else 0
    os.makedirs(args.out, exist_ok=True)
    for name, rows in (("train.jsonl", data[n_eval:]), ("eval.jsonl", data[:n_eval])):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    chars = sum(len(m["content"]) for r in data for m in r["messages"])
    print(f"From your Forge history: {len(hist)} good examples ({skipped} skipped: needed fixes or had errors)")
    print(f"From examples/:          {len(exs)}")
    print(f"Wrote {len(data) - n_eval} train + {n_eval} eval examples to {args.out}  (~{chars // 4:,} tokens)")
    if len(data) < 200:
        print("\nTip: aim for 200–1,000 high-quality examples before expecting a big improvement.")


if __name__ == "__main__":
    main()
