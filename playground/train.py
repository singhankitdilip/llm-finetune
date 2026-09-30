"""TRAIN a LoRA adapter on your own data.

    python train.py --data data/mydata.jsonl --name myrun
    python train.py --data data/mydata.jsonl --name myrun --epochs 12 --r 8 --lr 1e-4

Data format: one JSON per line, chat style:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
(an optional first {"role": "system", ...} message is fine too)

Output goes to runs/<name>/ : the adapter + loss_log.json.
"""
import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="path to your .jsonl file")
ap.add_argument("--name", required=True, help="run name -> runs/<name>/")
ap.add_argument("--base", default="LiquidAI/LFM2.5-230M", help="HF model id to start from")
ap.add_argument("--epochs", type=int, default=10, help="passes over the data")
ap.add_argument("--lr", type=float, default=2e-4, help="learning rate (LoRA: 1e-4..3e-4)")
ap.add_argument("--r", type=int, default=16, help="LoRA rank: adapter size/capacity")
ap.add_argument("--batch", type=int, default=8, help="examples per step (lower it if out of memory)")
ap.add_argument("--max-len", type=int, default=512, help="longest example in tokens")
ap.add_argument("--val-frac", type=float, default=0.1, help="share of data held out for validation")
args = ap.parse_args()

out_dir = f"runs/{args.name}"
os.makedirs(out_dir, exist_ok=True)

# 1. Tokenizer + model (bf16 = half the memory of float32; the 3060 Ti supports it).
tok = AutoTokenizer.from_pretrained(args.base)
model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)

# 2. Data -> split into train / validation. The model NEVER trains on validation;
#    it only measures "how well do I do on examples I haven't memorized?"
ds = load_dataset("json", data_files=args.data)["train"]
n_val = max(1, int(len(ds) * args.val_frac))
split = ds.train_test_split(test_size=n_val, seed=0, shuffle=True)
print(f"examples: {len(split['train'])} train / {len(split['test'])} val")

# 3. LoRA: freeze the base, train small add-on matrices.
lora = LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05,
                  target_modules="all-linear", task_type="CAUSAL_LM")

# 4. Training settings.
cfg = SFTConfig(
    output_dir=out_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_steps=0.1,            # float < 1 = ratio of total steps
    bf16=True,
    logging_steps=1,
    eval_strategy="epoch",       # validation loss after every epoch
    save_strategy="no",
    max_length=args.max_len,
    report_to="none",
)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=split["train"],
                     eval_dataset=split["test"], peft_config=lora, processing_class=tok)
trainer.model.print_trainable_parameters()
trainer.train()

# 5. Save adapter + a loss log you can inspect.
adapter_dir = f"{out_dir}/adapter"
trainer.model.save_pretrained(adapter_dir)
with open(f"{out_dir}/meta.json", "w") as f:
    json.dump({"base": args.base, **{k: v for k, v in vars(args).items() if k != "base"}}, f, indent=2)

log = trainer.state.log_history
with open(f"{out_dir}/loss_log.json", "w") as f:
    json.dump(log, f, indent=1)

train_loss = [l["loss"] for l in log if "loss" in l]
val = [(round(l["epoch"]), round(l["eval_loss"], 3)) for l in log if "eval_loss" in l]
best = min(val, key=lambda x: x[1])
print("\n===== SUMMARY =====")
print(f"final train loss : {train_loss[-1]:.3f}")
print(f"val loss by epoch: {val}")
print(f"best val loss    : {best[1]} at epoch {best[0]}  (of {args.epochs})")
if best[0] < args.epochs - 2:
    print("hint: val loss got worse after its best epoch -> overfitting; try fewer epochs or more data")
print(f"adapter saved    : {adapter_dir}")
