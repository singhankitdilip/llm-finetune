# LFM2.5-230M Fine-tuning Guide (Hinglish)

**Language:** [English](README.md) | Hinglish (ye file)

Ye guide batati hai ki is folder mein kya hai, kaise kaam karta hai, aur kisi bhi naye task ke liye model ko
kaise train karna hai. Ek baar setup ho gaya hai, ab sirf data aur settings badalni hain.

---

## 0. Naye computer par setup (GitHub se clone ke baad)

```bash
git clone <tumhara-repo-url> models && cd models
bash setup.sh          # .venv + libraries (requirements.txt) + llama.cpp build
```
Repo mein sirf code aur guide hai. `.venv`, `llama.cpp`, `.gguf` files aur `playground/runs/` **git mein nahi** hain
(badi hain), `setup.sh` unhe dobara bana deta hai. Base model weights pehli training par HuggingFace se apne aap
download hote hain. Apna data `playground/data/` mein rakho (`.gitignore` ke kaaran private rehta hai; push karna
ho to `.gitignore` mein uski line hatao).

---

## 1. Folder mein kya hai

```
models/
├── LFM2.5-230M-Q4_K_M.gguf   Original model (bina train kiya). Sirf chalane ke liye, isko train nahi kar sakte.
├── .venv/                    Python 3.12 environment (PyTorch GPU + training libraries)
├── llama.cpp/                GGUF convert/quantize/run karne ke tools (build ho chuke hain)
├── README.md / README.hinglish.md   Guide (English / Hinglish)
└── playground/
    ├── data/template.jsonl   Data ka format dekhne ke liye template
    ├── train.py              Training
    ├── chat.py               Model se baat (base ya trained)
    ├── export.py             Merge + GGUF + quantize ek command mein
    ├── runs/                 Har training ka alag folder (runs/<naam>/)
    └── README.md             Chhota quick reference
```

**Machine:** RTX 3060 Ti (8 GB VRAM), Python 3.12 (uv se), Linux.
**Base model:** `LiquidAI/LFM2.5-230M` (HuggingFace, ~460 MB bf16, 230M parameters).

---

## 2. Zaroori concept (pehle ye samjho)

| Term | Matlab |
|---|---|
| **Weights** | Model ke andar ke numbers (yahan ~23 crore). Training inko badalti hai. |
| **Token** | Text ka chhota tukda. LLM har baar sirf *agla token* predict karta hai. |
| **Loss** | Model kitna galat tha, uska score. Kam = accha. |
| **Epoch** | Poore data par ek baar training. |
| **Learning rate** | Har update mein weights kitne badle. Bada = tez par unstable, chhota = dheema. |
| **Batch size** | Ek baar mein kitne examples. |
| **Fine-tuning** | Bane-banaye model ko apne examples par thoda aur train karna. Style/task seekhta hai, bahut saare naye facts nahi. |
| **LoRA** | Original weights freeze karke sirf chhote add-on matrices (adapter) train karna. Sasta, tez, safe. |
| **r (rank)** | LoRA adapter ka size. Bada r = zyada capacity. Default 16. |
| **Adapter** | LoRA ka result (~15 MB). Akela nahi chalta, base model ke upar lagta hai. |
| **Merge** | Adapter ko base weights mein hamesha ke liye milana. |
| **GGUF** | llama.cpp/Ollama ka *chalane wala* format. Isko train nahi kar sakte. |
| **Quantization (Q4_K_M)** | Weights ko 16 bit se ~4.5 bit mein compress karna. Size ~460 MB -> ~150 MB, quality thodi kam. |
| **Validation set** | Data ka wo hissa jis par model kabhi train nahi hota. Isse pata chalta hai model *seekh* raha hai ya *ratta* maar raha hai. |
| **Overfitting** | Train loss girta rahe par validation loss badhne lage = model ratta maar raha hai. |

**Poora pipeline:**
```
HF original weights -> LoRA train -> adapter (15 MB) -> merge -> GGUF (f16) -> quantize -> .gguf -> llama.cpp
```
GGUF file seedhi train nahi hoti, isliye training HuggingFace wale original weights par hoti hai.

---

## 3. Libraries jo install hain (`.venv`)

| Library | Kaam |
|---|---|
| **torch** `2.14.0+cu130` | Engine: math, GPU, backpropagation |
| **transformers** `5.17.0` | Model + tokenizer load karna, `generate()` |
| **datasets** | `.jsonl` data padhna |
| **peft** | LoRA |
| **trl** | Training loop (`SFTTrainer`) |
| **accelerate** | Piche ka GPU/memory management |

Extra: **llama.cpp** (alag tool, Python library nahi) - `llama-cli`, `llama-quantize`, `convert_hf_to_gguf.py`.
Ye **CPU-only** build hua hai (`llama-server` build nahi hua; chahiye to alag se build karna hoga).

---

## 4. Naya task train karne ke steps

Har command `playground/` folder se chalao:
```bash
cd ~/Desktop/llm-models/models/playground
source ../.venv/bin/activate
```

### Step 1 - Data banao
`data/` mein `.jsonl` file rakho. Har line = ek baatcheet:
```json
{"messages": [{"role": "user", "content": "input"}, {"role": "assistant", "content": "ideal output"}]}
```
Optional system message pehle aa sakta hai: `{"role": "system", "content": "..."}`.

Data ke niyam:
- **Quality > quantity.** 200 saaf examples, 5000 ganday se behtar.
- Shuru mein **100-500 alag-alag** examples.
- Ek hi jawab ko bahut baar repeat mat karo, validation loss jhootha accha dikhega.
- Validation data training jaisa copy nahi, thoda alag hona chahiye.
- Fine-tuning **style/format/task** seekhati hai. Bahut saare naye *facts* chhote model mein pakke nahi hote.

### Step 2 - Baseline lo (training se pehle)
```bash
python chat.py --ask "apne task ka sawal 1" "sawal 2"
```
Jawab note kar lo, baad mein compare karne ke liye.

### Step 3 - Train karo
```bash
python train.py --data data/mydata.jsonl --name v1
```
Saari settings:

| Flag | Default | Matlab |
|---|---|---|
| `--data` | (zaroori) | Tumhari `.jsonl` file |
| `--name` | (zaroori) | Run ka naam -> `runs/<name>/` |
| `--epochs` | 10 | Data par kitne pass |
| `--lr` | 2e-4 | Learning rate (LoRA ke liye 1e-4 se 3e-4) |
| `--r` | 16 | LoRA rank |
| `--batch` | 8 | Examples per step (memory kam pade to ghatao) |
| `--max-len` | 512 | Sabse lamba example (tokens mein) |
| `--val-frac` | 0.1 | Validation ke liye alag data ka hissa |
| `--base` | LFM2.5-230M | Kisi aur HF model se shuru karna ho to |

Ant mein **SUMMARY** aata hai: final train loss, har epoch ka validation loss, best epoch, aur overfitting ho
to hint. `runs/<name>/` mein `adapter/`, `meta.json`, `loss_log.json` bante hain.

### Step 4 - Test karo
```bash
python chat.py --run v1                          # interactive chat (exit likhkar band)
python chat.py --run v1 --ask "sawal"            # ek-ek sawal
python chat.py --run v1 --temp 0.7               # thoda randomness
python chat.py --run v1 --system "system prompt"
```
Test mein **naye sawal** (jo data mein nahi the) bhi poochho, sirf training wale nahi.

### Step 5 - GGUF banao
```bash
python export.py --run v1                        # -> ../v1-Q4_K_M.gguf  (~150 MB)
python export.py --run v1 --quant Q8_0           # bada, original ke kareeb
```
Ye 3 kaam karta hai: merge -> convert (f16) -> quantize. Original `.gguf` kabhi overwrite nahi hoti.

### Step 6 - llama.cpp mein chalao
```bash
cd ~/Desktop/llm-models/models
llama.cpp/build/bin/llama-cli -m v1-Q4_K_M.gguf                              # interactive chat
llama.cpp/build/bin/llama-cli -m v1-Q4_K_M.gguf -p "sawal" --single-turn     # ek sawal
llama.cpp/build/bin/llama-cli -m v1-Q4_K_M.gguf --temp 0 -n 100              # repeatable, max 100 tokens
```
Band karne ke liye `Ctrl+C` ya `/exit`.

---

## 5. Memory (8 GB GPU)

| Cheez | Memory |
|---|---|
| LoRA training (r=16) | ~2-3 GB |
| Full fine-tune (saare weights) | ~4-5 GB (bhi aa jata hai) |
| LoRA trainable weights | 3,883,008 (model ka 1.66%) |
| `r` badhane par | LoRA wala hissa seedha badhta hai, par total par zyada farak nahi |
| Asli memory kharch | Batch size aur `--max-len` se badhta hai |

**Out of memory aaye to:** pehle `--batch` 8 -> 4 -> 2, phir `--max-len` ghatao.

**Model file size training se nahi badhta:** 230M weights train ke baad bhi 230M hi. Size bits-per-weight se tay hota
hai (bf16 ~460 MB, Q4_K_M ~150 MB). Adapter alag ~15 MB.

---

## 6. Jo hamne seekha (experiment, 65 examples, r=16)

| Epochs | Train loss | Val loss (aakhir) | Best val loss |
|---|---|---|---|
| 1 | 3.96 | 4.18 | 4.18 |
| 5 | 0.62 | 1.49 | 1.49 |
| 20 | 0.11 | 0.73 | 0.70 (epoch 13) |
| 50 | 0.16 | 0.76 | 0.64 (epoch 13) |

- 1 epoch kaafi nahi; 5 bhi kam the; is chhote data par ~10-15 epochs behtar.
- Best epoch ke baad validation loss dheere badhta hai = overfitting. Wahin ruk jao.
- LoRA se model purani skills nahi bhoola ("capital of France?" -> Paris sahi raha).
- Chhota model (230M) + chhota data = **style/persona seekha, facts pakke nahi hue.**
- Validation set training se alag tarah ka hona chahiye, warna loss jhootha accha dikhta hai.

---

## 7. Dikkatein jo aayi thi aur unke hal

**1. `Your setup doesn't support bf16/gpu` / `torch.cuda.is_available()` False**
Kaaran: `pip install -r llama.cpp/requirements/...` ne torch ko **CPU-only** version se badal diya aur transformers
downgrade kar diya. **Ye command kabhi mat chalana.** `export.py` ko iski zaroorat nahi, sab pehle se install hai.
Hal:
```bash
cd ~/Desktop/llm-models/models && source .venv/bin/activate
uv pip install "torch==2.14.0" "transformers==5.17.0"
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"   # 2.14.0+cu130 True
```

**2. `Tokenizer class TokenizersBackend does not exist`**
Kaaran: transformers 4.x (purana) 5.x ka tokenizer nahi padh sakta. Hal: upar wala transformers==5.17.0 wala fix.

**3. `SFTConfig got an unexpected keyword argument 'warmup_ratio'`**
Naye transformers mein naam badal gaya. Ab `warmup_steps=0.1` (float <1 = ratio) use hota hai. Scripts mein theek hai.

**4. Download bahut dheema (KB/s)**
Kaaran: WiFi/ISP. Ping 500 ms tha. Ethernet lagane se 50+ MB/s mila. Ek baat: agar download ke beech mein
ethernet lagao, to purane connections WiFi par atke rehte hain; download rokkar dobara chalao.

**5. Install bahut der chale to background mein kat jaye**
Lamba `timeout` do ya command background mein chalao.

**6. Naye transformers ke warnings (ignore kar sakte ho)**
`torch_dtype is deprecated`, `causal_conv1d ... falling back` (ek tez kernel nahi hai, chhote model par farak nahi),
`driver.c` compiler chetavni.

**7. Python 3.14 system mein hai**
PyTorch ke liye bahut naya. Isliye `uv` se alag Python 3.12 `.venv` banaya gaya hai. System Python mat use karo.

---

## 8. Safai / housekeeping

```bash
rm -rf playground/runs/<naam>          # ek run hatao
rm <naam>-Q4_K_M.gguf                  # ek exported GGUF hatao
du -sh ~/.cache/huggingface ~/.cache/uv   # cache kitna bhara hai (base weights HF cache mein hain, mat hatao)
```
`runs/<naam>/merged/` (~450 MB) export ke baad bhi rehta hai. Jagah chahiye to hata do, adapter se dobara ban jayega.

---

## 9. Agla kya

1. Apna task chuno (JSON extraction, classification, style/tone, rewrite/translation...).
2. 100-500 alag examples banao -> `playground/data/`.
3. Steps 2-6 chalao, `--name v1`, `v2`... se versions compare karo.
4. Validation loss dekho, overfitting ho to `--epochs` ghatao ya data badhao.
