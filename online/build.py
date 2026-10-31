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

# A full HTML document for normal web hosting (GitHub Pages, Netlify, any static host).
favicon = re.search(r'<link rel="icon"[^>]*>', app).group(0)
site = ("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"UTF-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0, viewport-fit=cover\">\n"
        f"<meta name=\"description\" content=\"Forge: describe a website and AI builds it, previews it, and fixes its own errors.\">\n{favicon}\n"
        "<style>html,body{height:100%;margin:0}</style>\n</head>\n<body>\n" + page + "\n</body>\n</html>\n")
site_dir = os.path.join(HERE, "site")
os.makedirs(site_dir, exist_ok=True)
with open(os.path.join(site_dir, "index.html"), "w", encoding="utf-8") as f:
    f.write(site)
print(f"Wrote {os.path.join(site_dir, 'index.html')} (for static hosting)")
