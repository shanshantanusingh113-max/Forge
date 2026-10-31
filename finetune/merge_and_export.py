#!/usr/bin/env python3
"""Merge your trained LoRA adapter into the base model and prepare it for Ollama.

    python merge_and_export.py                       # uses output/adapter
    python merge_and_export.py --name forge-coder    # name of the Ollama model

It saves a full Hugging Face model to output/merged, writes an Ollama Modelfile,
and prints the exact commands to convert it to GGUF and load it into Ollama.
"""
import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=os.path.join(HERE, "output", "adapter"))
    ap.add_argument("--base", default=None, help="defaults to the base model used for training")
    ap.add_argument("--out", default=os.path.join(HERE, "output", "merged"))
    ap.add_argument("--name", default="forge-coder")
    args = ap.parse_args()

    base = args.base
    if not base:
        with open(os.path.join(args.adapter, "forge_base.json")) as f:
            base = json.load(f)["base"]
    print(f"Loading {base} (full precision, this needs RAM roughly 2x the model size)…")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="cpu")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)

    gguf = os.path.join(os.path.dirname(args.out), f"{args.name}.gguf")
    modelfile = os.path.join(os.path.dirname(args.out), "Modelfile")
    with open(modelfile, "w") as f:
        f.write(f"FROM ./{args.name}.gguf\n"
                "PARAMETER temperature 0.3\n"
                "PARAMETER num_ctx 16384\n")
    out_dir = os.path.dirname(args.out)
    print(f"""
Merged model saved to {args.out}

Now turn it into an Ollama model:

  1. Get llama.cpp's converter (one time):
       git clone https://github.com/ggml-org/llama.cpp
       pip install -r llama.cpp/requirements.txt

  2. Convert to GGUF (q8_0 keeps quality; ~8 GB for a 7B model):
       python llama.cpp/convert_hf_to_gguf.py "{args.out}" --outfile "{gguf}" --outtype q8_0

     Optional, smaller and faster (~4.5 GB for 7B): build llama.cpp, then
       llama-quantize "{gguf}" "{gguf.replace('.gguf', '-q4.gguf')}" Q4_K_M
     and change the FROM line in the Modelfile to that file.

  3. Load it into Ollama:
       cd "{out_dir}"
       ollama create {args.name} -f Modelfile

  4. In Forge, open Settings → Ollama → Model: {args.name}

If replies come out garbled, your Ollama version did not pick up the chat template.
Copy the TEMPLATE block from the base model into the Modelfile:
       ollama show qwen2.5-coder:7b --modelfile
""")


if __name__ == "__main__":
    main()
