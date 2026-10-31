#!/usr/bin/env python3
"""Fine-tune an open coding model on your Forge dataset with LoRA / QLoRA.

Only a small "adapter" (a few hundred MB) is trained on top of the base model,
so this fits on one consumer GPU or a free Google Colab T4.

Examples:
    # 7B model, 4-bit QLoRA (needs ~12-16 GB of GPU memory)
    python train_lora.py --base Qwen/Qwen2.5-Coder-7B-Instruct --qlora

    # small model for an 8 GB GPU
    python train_lora.py --base Qwen/Qwen2.5-Coder-3B-Instruct --qlora --max-len 6144

Afterwards run merge_and_export.py to turn the result into an Ollama model.
"""
import argparse
import json
import math
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments)

HERE = os.path.dirname(os.path.abspath(__file__))
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Hugging Face model id or local path")
    ap.add_argument("--data", default=os.path.join(HERE, "data", "train.jsonl"))
    ap.add_argument("--eval-data", default=os.path.join(HERE, "data", "eval.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "output"))
    ap.add_argument("--qlora", action="store_true", help="load the base model in 4-bit (much less memory)")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32, help="LoRA rank: 16-64; higher learns more, needs more memory")
    ap.add_argument("--max-len", type=int, default=8192, help="max tokens per example")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def chat_ids(tok, messages, add_generation_prompt=False):
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return list(ids)


def build_encoder(tok, max_len, stats):
    def encode(ex):
        msgs = ex["messages"]
        prompt = chat_ids(tok, msgs[:-1], add_generation_prompt=True)
        full = chat_ids(tok, msgs)
        # Train only on the assistant's answer; the system prompt and request are context.
        labels = [-100] * len(prompt) + full[len(prompt):]
        if len(full) > max_len:
            stats["truncated"] += 1
            full, labels = full[:max_len], labels[:max_len]
        return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}
    return encode


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        def pad(key, value):
            return torch.tensor([b[key] + [value] * (n - len(b[key])) for b in batch])
        return {"input_ids": pad("input_ids", self.pad_id),
                "attention_mask": pad("attention_mask", 0),
                "labels": pad("labels", -100)}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        print("WARNING: no NVIDIA GPU found. Training on CPU will be extremely slow. "
              "Use Google Colab or a rented GPU (see README).")
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    files = {"train": args.data}
    if os.path.isfile(args.eval_data) and os.path.getsize(args.eval_data) > 0:
        files["eval"] = args.eval_data
    ds = load_dataset("json", data_files=files)
    stats = {"truncated": 0}
    ds = ds.map(build_encoder(tok, args.max_len, stats), remove_columns=ds["train"].column_names)
    ds = ds.filter(lambda e: any(l != -100 for l in e["labels"]))
    print(f"Examples: {len(ds['train'])} train" + (f", {len(ds['eval'])} eval" if "eval" in ds else ""))
    if stats["truncated"]:
        print(f"Note: {stats['truncated']} examples were longer than --max-len and were cut off.")

    quant = None
    if args.qlora:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=quant,
                                                 torch_dtype=dtype, device_map="auto")
    model.config.use_cache = False
    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    total_steps = math.ceil(len(ds["train"]) / (args.batch * args.grad_accum)) * args.epochs
    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.03 * total_steps)),
        logging_steps=5, save_strategy="epoch", save_total_limit=2,
        eval_strategy="epoch" if "eval" in ds else "no",
        bf16=bf16, fp16=torch.cuda.is_available() and not bf16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if args.qlora else "adamw_torch",
        report_to="none", seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds["train"],
                      eval_dataset=ds.get("eval"), data_collator=Collator(tok.pad_token_id))
    trainer.train()

    adapter_dir = os.path.join(args.out, "adapter")
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    with open(os.path.join(adapter_dir, "forge_base.json"), "w") as f:
        json.dump({"base": args.base}, f)
    print(f"\nDone. Adapter saved to {adapter_dir}\nNext: python merge_and_export.py")


if __name__ == "__main__":
    main()
