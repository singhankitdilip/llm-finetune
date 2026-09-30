# LFM2.5-230M Fine-tuning Guide

**Language:** English (this file) | [Hinglish](README.hinglish.md)

This guide explains what is in this repo, how it works, and how to fine-tune the model for any new task.
Setup is done once; after that you only change your data and settings.

---

## 0. Setting up on a new computer (after cloning from GitHub)

```bash
git clone <your-repo-url> models && cd models
bash setup.sh          # .venv + libraries (requirements.txt) + llama.cpp build
```
The repo contains only code and docs. `.venv`, `llama.cpp`, `.gguf` files and `playground/runs/` are **not in git**
(they are large); `setup.sh` recreates them. Base model weights download automatically from HuggingFace on the first
train/chat run. Put your own data in `playground/data/` (it is git-ignored so it stays private; remove its line from
`.gitignore` if you want to push it).

`setup.sh` needs: `git`, `cmake`, `g++`, and [`uv`](https://docs.astral.sh/uv/).

---

## 1. What is in the folder

```
models/
├── LFM2.5-230M-Q4_K_M.gguf   Original model (untrained). Run-only; GGUF files cannot be trained.
├── .venv/                    Python 3.12 environment (GPU PyTorch + training libraries)
├── llama.cpp/                Tools to convert / quantize / run GGUF (already built)
├── README.md / README.hinglish.md   This guide (English / Hinglish)
├── requirements.txt          Pinned library versions
├── setup.sh                  One-time setup script
└── playground/
    ├── data/template.jsonl   Template showing the data format
    ├── train.py              Training
    ├── chat.py               Talk to the model (base or trained)
    ├── export.py             Merge + GGUF + quantize in one command
    ├── runs/                 One folder per training run (runs/<name>/)
    └── README.md             Short quick reference
```

**Machine:** RTX 3060 Ti (8 GB VRAM), Python 3.12 (via uv), Linux.
**Base model:** `LiquidAI/LFM2.5-230M` (HuggingFace, ~460 MB in bf16, 230M parameters).

---

## 2. Key concepts (read these first)

| Term | Meaning |
|---|---|
| **Weights** | The numbers inside the model (~230 million here). Training changes them. |
| **Token** | A small piece of text. An LLM only ever predicts the *next token*. |
| **Loss** | A score for how wrong the model was. Lower is better. |
| **Epoch** | One pass over the whole dataset. |
| **Learning rate** | How much the weights change per update. Big = fast but unstable, small = slow. |
| **Batch size** | How many examples are processed at once. |
| **Fine-tuning** | Training an existing model a bit more on your examples. It learns style/task, not lots of new facts. |
| **LoRA** | Freeze the original weights and train only small add-on matrices (an adapter). Cheap, fast, safe. |
| **r (rank)** | Size of the LoRA adapter. Bigger r = more capacity. Default 16. |
| **Adapter** | The LoRA result (~15 MB). It does not run alone; it sits on top of the base model. |
| **Merge** | Permanently folding the adapter into the base weights. |
| **GGUF** | The format llama.cpp/Ollama *run*. It cannot be trained. |
| **Quantization (Q4_K_M)** | Compressing weights from 16 bits to ~4.5 bits. Size ~460 MB -> ~150 MB, slightly lower quality. |
| **Validation set** | Data the model never trains on. It shows whether the model is *learning* or just *memorizing*. |
| **Overfitting** | Train loss keeps falling while validation loss rises = the model is memorizing. |

**Whole pipeline:**
```
HF original weights -> LoRA train -> adapter (15 MB) -> merge -> GGUF (f16) -> quantize -> .gguf -> llama.cpp
```
A GGUF file cannot be trained directly, so training happens on the original HuggingFace weights.

---

## 3. Installed libraries (`.venv`)

| Library | Job |
|---|---|
| **torch** `2.14.0+cu130` | Engine: math, GPU, backpropagation |
| **transformers** `5.17.0` | Load model + tokenizer, `generate()` |
| **datasets** | Read `.jsonl` data |
| **peft** | LoRA |
| **trl** | Training loop (`SFTTrainer`) |
| **accelerate** | GPU/memory management behind the scenes |

Extra: **llama.cpp** (a separate tool, not a Python library): `llama-cli`, `llama-quantize`, `convert_hf_to_gguf.py`.
It is a **CPU-only** build (`llama-server` is not built; build it separately if you want a web UI).

---

## 4. Steps to train a new task

Run every command from the `playground/` folder:
```bash
cd ~/Desktop/llm-models/models/playground
source ../.venv/bin/activate
```

### Step 1 - Make your data
Put a `.jsonl` file in `data/`. One line = one conversation:
```json
{"messages": [{"role": "user", "content": "input"}, {"role": "assistant", "content": "ideal output"}]}
```
An optional system message can come first: `{"role": "system", "content": "..."}`.

Data rules:
- **Quality > quantity.** 200 clean examples beat 5000 messy ones.
- Start with **100-500 varied** examples.
- Do not repeat the same answer many times, or validation loss will look better than it really is.
- Validation data should be somewhat *different* from training data, not near-copies.
- Fine-tuning teaches **style / format / task**. A small model does not reliably memorize lots of new *facts*.

### Step 2 - Take a baseline (before training)
```bash
python chat.py --ask "a question from your task" "another question"
```
Note the answers so you can compare later.

### Step 3 - Train
```bash
python train.py --data data/mydata.jsonl --name v1
```
All settings:

| Flag | Default | Meaning |
|---|---|---|
| `--data` | (required) | Your `.jsonl` file |
| `--name` | (required) | Run name -> `runs/<name>/` |
| `--epochs` | 10 | Passes over the data |
| `--lr` | 2e-4 | Learning rate (1e-4 to 3e-4 for LoRA) |
| `--r` | 16 | LoRA rank |
| `--batch` | 8 | Examples per step (lower it if you run out of memory) |
| `--max-len` | 512 | Longest example, in tokens |
| `--val-frac` | 0.1 | Share of data held out for validation |
| `--base` | LFM2.5-230M | Start from a different HF model |

At the end a **SUMMARY** prints: final train loss, validation loss per epoch, the best epoch, and a hint if it
overfit. `runs/<name>/` gets `adapter/`, `meta.json` and `loss_log.json`.

### Step 4 - Test
```bash
python chat.py --run v1                          # interactive chat (type "exit" to quit)
python chat.py --run v1 --ask "a question"       # one-shot questions
python chat.py --run v1 --temp 0.7               # a bit of randomness
python chat.py --run v1 --system "system prompt"
```
Also ask **new questions** that were not in your data, not only ones from training.

### Step 5 - Make a GGUF
```bash
python export.py --run v1                        # -> ../v1-Q4_K_M.gguf  (~150 MB)
python export.py --run v1 --quant Q8_0           # bigger, closer to the original
```
This does 3 things: merge -> convert (f16) -> quantize. It never overwrites the original `.gguf`.

### Step 6 - Run it in llama.cpp
```bash
cd ~/Desktop/llm-models/models
llama.cpp/build/bin/llama-cli -m v1-Q4_K_M.gguf                              # interactive chat
llama.cpp/build/bin/llama-cli -m v1-Q4_K_M.gguf -p "a question" --single-turn   # one question
llama.cpp/build/bin/llama-cli -m v1-Q4_K_M.gguf --temp 0 -n 100              # repeatable, max 100 tokens
```
Quit with `Ctrl+C` or `/exit`.

---

## 5. Memory (8 GB GPU)

| Item | Memory |
|---|---|
| LoRA training (r=16) | ~2-3 GB |
| Full fine-tune (all weights) | ~4-5 GB (also fits) |
| LoRA trainable weights | 3,883,008 (1.66% of the model) |
| Raising `r` | The LoRA part grows linearly, but total memory barely changes |
| The real memory cost | Grows with batch size and `--max-len` |

**If you get out-of-memory:** lower `--batch` 8 -> 4 -> 2 first, then `--max-len`.

**Training does not make the model file bigger:** 230M weights stay 230M. File size depends on bits per weight
(bf16 ~460 MB, Q4_K_M ~150 MB). The adapter is a separate ~15 MB.

---

## 6. What we learned (experiment: 65 examples, r=16)

| Epochs | Train loss | Val loss (final) | Best val loss |
|---|---|---|---|
| 1 | 3.96 | 4.18 | 4.18 |
| 5 | 0.62 | 1.49 | 1.49 |
| 20 | 0.11 | 0.73 | 0.70 (epoch 13) |
| 50 | 0.16 | 0.76 | 0.64 (epoch 13) |

- 1 epoch is not enough; 5 was also too few; ~10-15 epochs works better on this small dataset.
- After the best epoch, validation loss slowly rises = overfitting. Stop around there.
- LoRA did not erase old skills ("capital of France?" still answered "Paris").
- Small model (230M) + small data = **it learned the style/persona, but facts were not reliable.**
- The validation set must differ from the training data, or the loss looks falsely good.

---

## 7. Problems we hit and their fixes

**1. `Your setup doesn't support bf16/gpu` / `torch.cuda.is_available()` is False**
Cause: `pip install -r llama.cpp/requirements/...` replaced torch with a **CPU-only** build and downgraded
transformers. **Never run that command.** `export.py` does not need it; everything is already installed.
Fix:
```bash
cd ~/Desktop/llm-models/models && source .venv/bin/activate
uv pip install "torch==2.14.0" "transformers==5.17.0"
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"   # 2.14.0+cu130 True
```

**2. `Tokenizer class TokenizersBackend does not exist`**
Cause: old transformers 4.x cannot read the 5.x tokenizer. Fix: the same `transformers==5.17.0` fix above.

**3. `SFTConfig got an unexpected keyword argument 'warmup_ratio'`**
Renamed in the newer transformers. The scripts now use `warmup_steps=0.1` (a float below 1 means a ratio).

**4. Very slow download (KB/s)**
Cause: WiFi/ISP (ping was 500 ms). Plugging in ethernet gave 50+ MB/s. Note: if you plug in ethernet mid-download,
old connections stay stuck on WiFi; stop the download and restart it.

**5. A long install gets killed in the background**
Give it a long `timeout`, or run it as a background command.

**6. Warnings from new transformers (safe to ignore)**
`torch_dtype is deprecated`; `causal_conv1d ... falling back` (a fast kernel is missing, no real effect on a small
model); `driver.c` compiler notes.

**7. System Python is 3.14**
Too new for PyTorch, so a separate Python 3.12 `.venv` is made with `uv`. Do not use the system Python.

---

## 8. Cleanup / housekeeping

```bash
rm -rf playground/runs/<name>          # delete one run
rm <name>-Q4_K_M.gguf                  # delete one exported GGUF
du -sh ~/.cache/huggingface ~/.cache/uv   # how big the caches are (base weights live in the HF cache; keep them)
```
`runs/<name>/merged/` (~450 MB) stays after export. Delete it to save space; it can be rebuilt from the adapter.

---

## 9. What next

1. Pick a task (JSON extraction, classification, style/tone, rewrite/translation...).
2. Make 100-500 varied examples -> `playground/data/`.
3. Run steps 2-6 with `--name v1`, `v2`... to compare versions.
4. Watch validation loss; if it overfits, lower `--epochs` or add more data.
