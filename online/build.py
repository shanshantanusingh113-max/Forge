#!/usr/bin/env python3
"""Build Forge Online (a single page that runs on Claude) from the main app's parts.

    python online/build.py   ->  online/forge-online.html

It reuses the main app's stylesheet and icons and the exact system prompt from
forge_prompt.py, so the two versions stay in step.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from forge_prompt import SYSTEM_PROMPT  # noqa: E402

app = open(os.path.join(ROOT, "static", "index.html"), encoding="utf-8").read()
css = re.search(r"<style>.*?</style>", app, re.S).group(0)
icons = re.search(r"/\* =+ icons =+ \*/.*?document\.querySelectorAll\('\[data-logo\]'\)[^\n]*\n", app, re.S).group(0)

page = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
page = (page.replace("/*BASE_CSS*/", css)
            .replace("/*SHARED_ICONS*/", icons)
            .replace("/*SYSTEM_PROMPT*/", json.dumps(SYSTEM_PROMPT).replace("</", "<\\/")))
out = os.path.join(HERE, "forge-online.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(page)
print(f"Wrote {out} ({len(page):,} bytes)")
