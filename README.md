# Forge — your own AI website builder

Describe a website. Forge writes the code, shows it live, checks it for errors, and fixes its own mistakes — all running on your own computer with a free open-source model.

- Chat interface with project history, light/dark themes, works on phones too
- Live preview with desktop / tablet / phone widths, and a code viewer
- **Auto-fix loop:** after every reply Forge checks for missing files, JavaScript syntax errors, cut-off output, and runtime errors in the preview, then sends them back to the model to fix (up to 3 tries)
- Download any site as a .zip, ready to upload to Netlify, Vercel, GitHub Pages or any host
- Works with Ollama (free, local), any OpenAI-compatible API, or a built-in demo mode
- Fine-tuning kit in `finetune/` to train the model on your own best results

## Quick start

**1. Install Python 3.9+** from [python.org](https://www.python.org/downloads/) (on Windows, tick "Add Python to PATH").

**2. Try it right away (no model needed):**

- Windows: double-click `start.bat`
- Mac / Linux: run `./start.sh` in a terminal

Your browser opens at http://localhost:8765. If no model is found, click **Use demo mode** and pick a suggestion. The demo deliberately ships a small bug so you can watch Forge catch and fix it.

**3. Add a real model (free):**

1. Install [Ollama](https://ollama.com/download) and open it.
2. In a terminal, download a coding model sized for your computer:

| Your computer | Command | Quality |
|---|---|---|
| 8 GB RAM, no GPU | `ollama pull qwen2.5-coder:3b` | Basic pages |
| 16 GB RAM, or a GPU with 8 GB | `ollama pull qwen2.5-coder:7b` | Good — the default |
| 32 GB RAM, or a GPU with 16 GB+ | `ollama pull qwen2.5-coder:14b` | Better |
| GPU with 24 GB+ | `ollama pull qwen2.5-coder:32b` | Best local results |

Newer coding models appear often — check [ollama.com/library](https://ollama.com/library) and pick the strongest one that fits.

3. In Forge, open **Settings** (gear icon, bottom left) → **Ollama** → type the model name → **Test connection** → **Save**.

**Want much stronger results?** Choose **API** in Settings and use a cloud model through any OpenAI-compatible service (OpenAI, OpenRouter, Groq, Together…). Paste the service's URL, model id and API key. Your key stays in `settings.json` on your computer.

## How it works

```
 You ──► Chat UI ──► server.py ──► model (Ollama / API)
                        │               │ streams back code in <file path="…"> blocks
                        ▼               ▼
                  writes files ──► static checks (missing files, syntax, cut-off output)
                        │
                        ▼
                  live preview ──► catches runtime errors in the page
                        │
             problems? ─┴─► sent back to the model as an [auto-fix] message (max 3 tries)
```

| File | What it does |
|---|---|
| `server.py` | The whole backend: model streaming, file writing, checks, preview server. Standard library only. |
| `forge_prompt.py` | The system prompt — Forge's "personality" and design standards. Edit this to change how it builds. |
| `static/index.html` | The interface. |
| `projects/` | Your projects: `chat.json` (history) and `site/` (the files). |
| `finetune/` | Train the model on your best results. See `finetune/README.md`. |

## Put it online (Render, free)

Online, Forge can't reach the Ollama on your laptop, so it uses a cloud model API instead. It's protected by a password so nobody else can spend your API credits.

1. Get an API key from a model provider. [OpenRouter](https://openrouter.ai) gives one key for many models, including free ones.
2. Sign in to [render.com](https://render.com) with GitHub → **New** → **Blueprint** → pick this repository.
3. Fill in the values Render asks for:
   - `FORGE_PASSWORD` — the password you'll type to open Forge (make it long)
   - `FORGE_MODEL` — a model id from your provider, e.g. from [openrouter.ai/models](https://openrouter.ai/models)
   - `FORGE_API_KEY` — your API key
4. Click **Apply**. After a minute or two you get a link like `https://forge-xxxx.onrender.com`.

Things to know about the free plan:
- It sleeps after 15 minutes without visitors; the first visit after that takes about a minute to wake up.
- **Projects are wiped when the service restarts or redeploys.** Download sites you want to keep with the ⬇ button. For permanent storage, add a Render disk (paid) mounted at `/opt/render/project/src/projects`.
- To try it without an API key first, set `FORGE_PROVIDER` to `demo` in Render's Environment tab.

Other hosts: the included `Dockerfile` runs on Railway, Fly.io or any VPS. Always set `FORGE_PASSWORD` — Forge refuses to start on a public address without one.

| Setting | What it does |
|---|---|
| `FORGE_PASSWORD` | Turns on the sign-in screen |
| `FORGE_PROVIDER` | `ollama`, `openai` (any OpenAI-compatible API) or `demo` |
| `FORGE_BASE_URL` / `FORGE_MODEL` / `FORGE_API_KEY` | Default model connection (the key is never written to disk) |
| `FORGE_HOST` / `PORT` | Where the server listens |

## Making it better

1. **Use the biggest model your computer can run.** This matters more than anything else.
2. **Tune `forge_prompt.py`.** Add your preferred fonts, colors, or rules ("always include a footer", "use Tailwind from CDN").
3. **Fine-tune** on your best sites — see `finetune/README.md`.

## Troubleshooting

- **"Could not reach the model server"** — Ollama isn't running. Open the Ollama app or run `ollama serve`.
- **"The model was not found"** — run `ollama pull <model name>` with the exact name from Settings.
- **Replies stop mid-file** — the model ran out of output space. Forge detects this and asks it to redo the file. If it happens a lot, ask for smaller pieces ("first build the header and hero").
- **Very slow replies** — the model is too big for your computer. Use a smaller one.
- **Port already in use** — run with another port: `FORGE_PORT=8800 python server.py` (Windows: `set FORGE_PORT=8800` then `python server.py`).
- **Use it from your phone on the same Wi-Fi** — start with `FORGE_HOST=0.0.0.0` and open `http://<your-computer's-IP>:8765`. Only do this on a network you trust: there is no login.
