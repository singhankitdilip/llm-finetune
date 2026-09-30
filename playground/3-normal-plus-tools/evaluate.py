"""SCORE models on held-out prompts so you compare numbers, not vibes.

    python evaluate.py                                 # base model only
    python evaluate.py --runs chat-v1 tools-v1 combo-v1   # base + each run, side by side

Two tests:
  TOOLS   data/tools_test.jsonl  (phrasings NEVER used in training). Per prompt we check
          - a call is made when one is needed, and NOT made when none is needed
          - the right tool name, and exactly the right arguments
  PERSONA 3 unseen questions; the answer must mention "Orbit" (only meaningful for the Orbit recipe)
"""
import argparse
import ast
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="*", default=[], help="run names under runs/ (base model is always included)")
ap.add_argument("--test", default="data/tools_test.jsonl")
ap.add_argument("--base", default="LiquidAI/LFM2.5-230M")
ap.add_argument("--only", choices=["tools", "persona", "both"], default="both",
                help="which test to run (persona = Orbit recipe, tools = tool-calling recipe)")
ap.add_argument("--show", action="store_true", help="print every wrong tool answer")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.base)
tests = [json.loads(l) for l in open(args.test) if l.strip()] if args.only != "persona" else []
PERSONA_Q = ["Who am I talking to?", "Are you ChatGPT?", "Hey, what exactly are you?"]


def parse_call(text):
    """'<|tool_call_start|>[fn(a='x', n=5)]<|tool_call_end|>' -> ('fn', {'a': 'x', 'n': 5}); None if no/garbled call.
    Accepts both quote styles: Python-style 'x' (training data) and "x" (what the base model writes)."""
    m = re.search(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", text, re.S)
    if not m:
        return None
    try:
        node = ast.parse(m.group(1).strip().strip("[]").strip(), mode="eval").body
        return node.func.id, {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    except Exception:
        return ("<garbled>", {})


def same(a, b):
    norm = lambda d: {k: (v.replace(" ", "") if isinstance(v, str) else v) for k, v in d.items()}
    return norm(a) == norm(b)


def load(run):
    m = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).to("cuda")
    if run:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, f"runs/{run}/adapter")
    return m.eval()


def gen(model, messages, tools=None):
    ids = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=80, do_sample=False)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)


results = {}
for run in [None] + args.runs:
    label = run or "base"
    model = load(run)
    need = [t for t in tests if t["expect"]]
    nocall = [t for t in tests if not t["expect"]]
    do_tools, do_persona = args.only != "persona", args.only != "tools"
    name_ok = args_ok = 0
    wrong = []
    for t in (need if do_tools else []):
        out = gen(model, [{"role": "user", "content": t["prompt"]}], t["tools"])
        c = parse_call(out)
        if c and c[0] == t["expect"]["name"]:
            name_ok += 1
            if same(c[1], t["expect"]["args"]):
                args_ok += 1
                continue
        wrong.append((t["prompt"], t["expect"], out.replace("<|im_end|>", "").strip()))
    nocall_ok = sum(parse_call(gen(model, [{"role": "user", "content": t["prompt"]}], t["tools"])) is None
                    for t in (nocall if do_tools else []))
    persona = sum("orbit" in gen(model, [{"role": "user", "content": q}]).lower()
                  for q in (PERSONA_Q if do_persona else []))
    results[label] = dict(name=name_ok / len(need) if do_tools else None, args=args_ok / len(need) if do_tools else None,
                          nocall=nocall_ok / len(nocall) if do_tools else None,
                          persona=persona / len(PERSONA_Q) if do_persona else None)
    if args.show:
        for p, e, o in wrong:
            print(f"  [{label}] {p!r}\n     expected {e}\n     got      {o[:120]}")
    del model
    torch.cuda.empty_cache()

pct = lambda v: "-" if v is None else f"{v:.0%}"
print(f"\n{'model':<14}{'right tool':>12}{'exact args':>12}{'no-call ok':>12}{'persona':>10}")
print(f"{'':<14}{'(n=%d)' % len(need):>12}{'(n=%d)' % len(need):>12}{'(n=%d)' % len(nocall):>12}{'(n=3)':>10}")
for k, v in results.items():
    print(f"{k:<14}{pct(v['name']):>12}{pct(v['args']):>12}{pct(v['nocall']):>12}{pct(v['persona']):>10}")
