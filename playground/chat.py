"""CHAT with the base model or one of your trained runs.

    python chat.py                      # base model, interactive
    python chat.py --run myrun          # base + runs/myrun/adapter, interactive
    python chat.py --run myrun --ask "question 1" "question 2"   # one-shot questions
    python chat.py --system "You are a helpful assistant."        # optional system prompt

Type 'exit' to quit. Uses greedy decoding by default (repeatable); add --temp 0.7 for variety.
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--run", help="run name under runs/ (omit for the untouched base model)")
ap.add_argument("--base", default=None, help="base model id (default: the one the run was trained from)")
ap.add_argument("--ask", nargs="*", help="ask these questions and exit")
ap.add_argument("--system", default=None)
ap.add_argument("--temp", type=float, default=0.0)
ap.add_argument("--max-new", type=int, default=200)
args = ap.parse_args()

base = args.base or "LiquidAI/LFM2.5-230M"
if args.run and not args.base:
    base = json.load(open(f"runs/{args.run}/meta.json"))["base"]

tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to("cuda")
if args.run:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, f"runs/{args.run}/adapter")  # stack adapter on base
model.eval()
print(f"[model: {base}" + (f" + runs/{args.run}" if args.run else " (base, no training)") + "]")


def reply(history):
    ids = tok.apply_chat_template(history, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to("cuda")
    kw = dict(do_sample=True, temperature=args.temp) if args.temp > 0 else dict(do_sample=False)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=args.max_new, **kw)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


sys_msg = [{"role": "system", "content": args.system}] if args.system else []

if args.ask:
    for q in args.ask:
        print(f"\nQ: {q}\nA: {reply(sys_msg + [{'role': 'user', 'content': q}])}")
else:
    history = list(sys_msg)
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("exit", "quit"):
            break
        if not q:
            continue
        history.append({"role": "user", "content": q})
        a = reply(history)
        history.append({"role": "assistant", "content": a})
        print(f"model> {a}")
