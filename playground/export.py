"""EXPORT a trained run to a .gguf file for llama.cpp.

    python export.py --run myrun                 # -> ../<run>-Q4_K_M.gguf
    python export.py --run myrun --quant Q8_0    # other quantization level

Steps: merge adapter into base -> convert to GGUF (f16) -> quantize.
NOTE: do NOT `pip install -r llama.cpp/requirements/...` in this venv: it swaps in a
CPU-only torch and downgrades transformers. Everything the converter needs is already installed.
"""
import argparse
import json
import os
import subprocess
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--quant", default="Q4_K_M", help="Q4_K_M (small), Q5_K_M, Q8_0 (bigger, closer to original)")
args = ap.parse_args()

run = f"runs/{args.run}"
base = json.load(open(f"{run}/meta.json"))["base"]
merged, f16 = f"{run}/merged", f"{run}/model-f16.gguf"
final = f"../{args.run}-{args.quant}.gguf"
llama = "../llama.cpp"

# 1. MERGE: fold the adapter weights into the base weights -> one normal model.
print("1/3 merging adapter into base ...")
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, f"{run}/adapter").merge_and_unload()
model.save_pretrained(merged)
AutoTokenizer.from_pretrained(base).save_pretrained(merged)

# 2. CONVERT: HF format -> GGUF (still 16-bit).
print("2/3 converting to GGUF ...")
subprocess.run([sys.executable, f"{llama}/convert_hf_to_gguf.py", merged,
                "--outfile", f16, "--outtype", "f16"], check=True, stdout=subprocess.DEVNULL)

# 3. QUANTIZE: shrink weights to ~4.5 bits each.
print(f"3/3 quantizing to {args.quant} ...")
subprocess.run([f"{llama}/build/bin/llama-quantize", f16, final, args.quant],
               check=True, stderr=subprocess.DEVNULL)

os.remove(f16)  # the f16 file is only an intermediate
print(f"\ndone -> {os.path.abspath(final)}  ({os.path.getsize(final) / 1e6:.0f} MB)")
print(f"run it: {llama}/build/bin/llama-cli -m {final}")
