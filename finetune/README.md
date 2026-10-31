# Fine-tuning your Forge model

Fine-tuning teaches an open model **your** style and Forge's exact output format, so it makes fewer mistakes and needs fewer auto-fixes. It does not turn a 7B model into a frontier model. Expect it to get noticeably more consistent, not dramatically smarter.

## The loop

```
use Forge  →  build_dataset.py  →  train_lora.py  →  merge_and_export.py  →  ollama create  →  benchmark.py
   ↑                                                                                                 │
   └──────────────────────────── use your improved model, collect more examples ─────────────────────┘
```

## 1. Collect examples

Quality beats quantity. 200–1,000 excellent examples is a good target.

- **Your Forge history.** Every reply that passed the checks without needing a fix is used automatically. Bad replies are skipped.
- **Hand-made examples.** Add a folder to `examples/` like `examples/saas-pricing-page/`: a `prompt.txt`, a `site/` folder with the ideal files, and optional `reply_intro.txt` / `reply_summary.txt`.
- **Distillation (biggest win).** Temporarily switch Forge to a much stronger cloud model (Settings → API, e.g. through OpenRouter) and build many sites with it. Those good replies become training data that teaches your small local model to imitate the stronger one. Check the provider's terms of service first — some don't allow using their outputs to train other models.

```bash
python build_dataset.py
```

This writes `data/train.jsonl` and `data/eval.jsonl`.

## 2. Train

You need an NVIDIA GPU. Pick the model size by GPU memory:

| GPU memory | Base model | Command |
|---|---|---|
| 8 GB | Qwen2.5-Coder-3B-Instruct | `python train_lora.py --base Qwen/Qwen2.5-Coder-3B-Instruct --qlora --max-len 6144` |
| 12–16 GB (free Colab T4) | Qwen2.5-Coder-7B-Instruct | `python train_lora.py --base Qwen/Qwen2.5-Coder-7B-Instruct --qlora --max-len 6144` |
| 24 GB | Qwen2.5-Coder-14B-Instruct | `python train_lora.py --base Qwen/Qwen2.5-Coder-14B-Instruct --qlora` |
| 48–80 GB (rented) | Qwen2.5-Coder-32B-Instruct | `python train_lora.py --base Qwen/Qwen2.5-Coder-32B-Instruct --qlora` |

Any Hugging Face chat model with Llama/Qwen/Mistral-style layers works; check for newer coding models before you start. Use the **same model family** you run in Ollama.

```bash
pip install -r requirements.txt
python train_lora.py --base Qwen/Qwen2.5-Coder-7B-Instruct --qlora
```

Rough time: 500 examples × 2 epochs on a 7B model takes 1–3 hours on a T4, under an hour on an A100. Watch the `loss` in the log: it should fall and then level off. If `eval_loss` starts rising while `loss` keeps falling, the model is memorizing — train fewer epochs.

**No GPU?** Use Google Colab (Runtime → Change runtime type → T4 GPU). Upload the `finetune` folder plus `forge_prompt.py` one level above it, then run the same commands in a cell prefixed with `!`. Or rent a GPU by the hour from RunPod, Vast.ai, Lambda or similar.

## 3. Export to Ollama

```bash
python merge_and_export.py --name forge-coder
```

It prints the exact commands to convert to GGUF with llama.cpp and run `ollama create`. Then pick `forge-coder` in Forge's Settings.

## 4. Check it actually improved

```bash
python benchmark.py --model qwen2.5-coder:7b     # before
python benchmark.py --model forge-coder          # after
```

Compare the pass rate and the number of problems. Add `--keep` to open the generated sites in Forge and judge the design by eye — the benchmark can't measure taste.

## Tips

- Bad examples teach bad habits. Delete projects you didn't like before running `build_dataset.py`.
- Variety matters: landing pages, dashboards, forms, games, multi-page sites, edits to existing sites, and fixes.
- Edit requests ("make the header sticky", "add a contact page") are as important as builds from scratch.
- If you change `forge_prompt.py`, rebuild the dataset so training matches what the app sends.
