"""TRAIN a LoRA adapter on your own data.

    python train.py --data data/mydata.jsonl --name myrun
    python train.py --data data/mydata.jsonl --name myrun --epochs 12 --r 8 --lr 1e-4
    python train.py --data data/chat.jsonl data/tools.jsonl --name combo   # MIX several files in one run
    python train.py --data data/chat_orbit.jsonl:3 data/tools_main.jsonl --name combo   # ":3" = use that file 3x

Data format: one JSON per line, chat style:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
(an optional first {"role": "system", ...} message is fine too)

Tool-calling examples add a "tools" list and use tool_calls / role "tool" messages (see data/template_tools.jsonl).

Output goes to runs/<name>/ : the adapter + loss_log.json.
"""
import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, nargs="+", help="one or more .jsonl files, mixed and shuffled. Add :N to repeat a file N times in TRAIN only (balance the mix)")
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

# 2. Data. Two things happen here, both on purpose:
#
#  (a) We render each example to text in plain Python, not through the `datasets` library (Arrow).
#      Arrow gives all rows one shared schema, so get_weather(city=..) silently becomes
#      get_weather(city=.., expression=None) when another row's tool has an "expression" argument.
#      The model would then learn junk arguments.
#
#  (b) ASSISTANT-ONLY LOSS. We mark every token that is NOT inside an assistant turn with label -100
#      ("ignore"). The model then learns only from what the assistant says: the answer, the
#      <|tool_call_start|>...<|tool_call_end|> call, and the final reply. Instruct models never
#      predict user/system/tool-list text, so training on it only adds noise (for tool calling the
#      tool schema alone is ~100 tokens per example).
import re

ASSISTANT = re.compile(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>)", re.S)


def encode(row):
    msgs = json.loads(json.dumps(row["messages"]))          # deep copy
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            a = tc["function"]["arguments"]
            if isinstance(a, str):                          # allow arguments stored as a JSON string
                tc["function"]["arguments"] = json.loads(a)
    text = tok.apply_chat_template(msgs, tools=row.get("tools"), tokenize=False)
    spans = [m.span(1) for m in ASSISTANT.finditer(text)]   # character ranges of assistant output
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)  # template already has <|startoftext|>
    ids = enc["input_ids"]
    labels = [t if any(s <= st < e for s, e in spans) else -100
              for t, (st, _) in zip(ids, enc["offset_mapping"])]
    return {"input_ids": ids[: args.max_len], "labels": labels[: args.max_len], "text": text}


rows, weights = [], []
for spec in args.data:
    path, n_rep = spec, 1
    head, sep, tail = spec.rpartition(":")          # "file.jsonl:3" -> use this file 3x in train
    if sep and tail.isdigit():
        path, n_rep = head, int(tail)
    with open(path) as f:
        part = [json.loads(line) for line in f if line.strip()]
    print(f"  {path}: {len(part)} examples" + (f"  (x{n_rep} in train)" if n_rep > 1 else ""))
    rows += part
    weights += [n_rep] * len(part)

enc_rows = [(encode(r), w) for r, w in zip(rows, weights)]
enc_rows = [(e, w) for e, w in enc_rows if any(l != -100 for l in e["labels"])]   # need >=1 trainable token
n_tok = sum(len(e["labels"]) * w for e, w in enc_rows)
n_learn = sum(sum(l != -100 for l in e["labels"]) * w for e, w in enc_rows)
print(f"tokens: {n_tok} total, {n_learn} trained on ({100 * n_learn / n_tok:.0f}%, assistant-only)")

ds = Dataset.from_dict({"input_ids": [e["input_ids"] for e, _ in enc_rows],
                        "labels": [e["labels"] for e, _ in enc_rows],
                        "w": [w for _, w in enc_rows]})
# Split FIRST (on unique examples), THEN repeat only the train part -> validation never contains a
# copy of a training example. The model NEVER trains on validation; it measures
# "how well do I do on examples I haven't memorized?"
n_val = max(1, int(len(ds) * args.val_frac))
split = ds.train_test_split(test_size=n_val, seed=0, shuffle=True)
tr = split["train"]
train_ds = Dataset.from_dict({
    "input_ids": [x for x, w in zip(tr["input_ids"], tr["w"]) for _ in range(w)],
    "labels": [x for x, w in zip(tr["labels"], tr["w"]) for _ in range(w)]})
val_ds = split["test"].remove_columns("w")
split = {"train": train_ds, "test": val_ds}
print(f"examples: {len(train_ds)} train / {len(val_ds)} val")

# Show one example with the trained part in [[double brackets]] so you can SEE what is learned.
ex = split["train"][0]
shown, inside = "", False
for t, l in zip(ex["input_ids"], ex["labels"]):
    if (l != -100) != inside:
        shown += "[[" if l != -100 else "]]"
        inside = l != -100
    shown += tok.decode([t])
print("--- sample (trained tokens in [[ ]]) ---")
print(shown + ("]]" if inside else ""))
print("----------------------------------------")

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
