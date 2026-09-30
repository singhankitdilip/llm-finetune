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
Repo mein sirf code aur guide hai. `.venv`, `llama.cpp`, `.gguf` files aur `playground/*/runs/` **git mein nahi** hain
(badi hain), `setup.sh` unhe dobara bana deta hai. Base model weights pehli training par HuggingFace se apne aap
download hote hain. Apna data `playground/<folder>/data/` mein rakho (`.gitignore` ke kaaran private rehta hai; push karna
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
    ├── README.md             Overview + teeno recipes ke results
    ├── 1-normal/             Normal chat training (Orbit persona)
    ├── 2-tools/              Tool-calling training
    └── 3-normal-plus-tools/  Dono skills ek model mein
        (har folder mein: make_data.py, data/, train.py, chat.py, evaluate.py, export.py,
         run.sh, runs/ = trained adapters, exports/ = .gguf files, README.md)
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

Apni recipe ka folder chuno (section 11) aur har command uske andar se chalao. Har folder ka
`bash run.sh <command>` neeche ke commands chhote kar deta hai (uski README dekho):
```bash
cd ~/Desktop/llm-models/models/playground/<folder>     # 1-normal | 2-tools | 3-normal-plus-tools
source ../../.venv/bin/activate
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
python export.py --run v1                        # -> exports/v1-Q4_K_M.gguf  (~150 MB)
python export.py --run v1 --quant Q8_0           # bada, original ke kareeb
```
Ye 3 kaam karta hai: merge -> convert (f16) -> quantize. Original `.gguf` kabhi overwrite nahi hoti.

### Step 6 - llama.cpp mein chalao
```bash
cd ~/Desktop/llm-models/models      # repo root
llama.cpp/build/bin/llama-cli -m playground/<folder>/exports/v1-Q4_K_M.gguf                              # interactive chat
llama.cpp/build/bin/llama-cli -m playground/<folder>/exports/v1-Q4_K_M.gguf -p "sawal" --single-turn     # ek sawal
llama.cpp/build/bin/llama-cli -m playground/<folder>/exports/v1-Q4_K_M.gguf --temp 0 -n 100              # repeatable, max 100 tokens
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
rm -rf playground/<folder>/runs/<naam>          # ek run hatao
rm playground/<folder>/exports/<naam>-Q4_K_M.gguf                  # ek exported GGUF hatao
du -sh ~/.cache/huggingface ~/.cache/uv   # cache kitna bhara hai (base weights HF cache mein hain, mat hatao)
```
`runs/<naam>/merged/` (~450 MB) export ke baad bhi rehta hai. Jagah chahiye to hata do, adapter se dobara ban jayega.

---

## 9. Agla kya

1. Apna task chuno (JSON extraction, classification, style/tone, rewrite/translation...).
2. 100-500 alag examples banao -> `playground/<folder>/data/`.
3. Steps 2-6 chalao, `--name v1`, `v2`... se versions compare karo.
4. Validation loss dekho, overfitting ho to `--epochs` ghatao ya data badhao.

---

## 10. Tool calling, aur normal training ke saath mix karna

### Model kya seekhta hai
Model koi function chalata nahi. Woh ek **structured call likhna** seekhta hai, phir result ko use karna:
```
user: "Weather in Paris?"
model -> <|tool_call_start|>[get_weather(city='Paris')]<|tool_call_end|>
tumhari app get_weather chalati hai -> "18C, cloudy"     (role "tool" message)
model -> "It's 18C and cloudy in Paris."
```
Teen skills: (1) tool call karna hai ya nahi, (2) sahi naam aur arguments bharna, (3) result ka use karna.
Base model pehle se simple calls kar leta hai (hamne test kiya), isliye training se woh TUMHARE tools par zyada
bharosemand hota hai.

### Data format
Normal `.jsonl` jaisa hi, bas har example mein `tools` list. `playground/2-tools/data/template_tools.jsonl` dekho.
Shamil karo: normal calls, **bina tool wale** examples (hello, general sawal), **argument gayab** wale cases (model
poochhe "Which city?"), aur kai tools jahan sirf ek sahi ho. Chhote model (230M) ke liye tools 3-6 hi rakho.
300-1000 alag-alag examples ka target rakho.

### `train.py` ye kaise karta hai (aur kyun)
- **Assistant-only loss.** Sirf wahi tokens train hote hain jo assistant bolta hai (tool call aur final jawab).
  Tool list, user ka text aur tool ka result mask hote hain (`-100`). Instruct model inhe kabhi predict nahi karta,
  aur sirf tool schema har example mein ~100 tokens hai. Masking se pehle loss 8-10 tha (zyadatar prompt tokens),
  baad mein ~1. Har run ek sample print karta hai jisme trained tokens `[[ ]]` mein hote hain, taaki tum check kar sako.
- **Python mein render, Arrow mein nahi.** `datasets` library saari rows ko ek schema mein milati hai aur
  `get_weather(city='Paris')` ko `get_weather(city='Paris', expression=None)` bana deti. Hum isse bachte hain.
- **Ek hi BOS token** (template aur tokenizer dono ek-ek lagate hain; hum sirf ek rakhte hain).

### Normal training aur tool training ek hi model par kaise milayein
| Tareeka | Kaise | Nateeja |
|---|---|---|
| **A. Ek run mein mix** | `python train.py --data data/chat.jsonl data/tools.jsonl --name combo` | **Recommended.** Dono skills saath seekhta hai, kuch bhoolta nahi. |
| B. Ek ke baad ek | Pehle chat train, phir uske upar tools (`--base` = merged model ka path) | Risky: doosra stage pehla bhula sakta hai (catastrophic forgetting). |
| C. Alag adapters | Har skill ka alag run, jo chahiye woh load karo | Saaf hai, par `chat.py` ek baar mein ek adapter load karta hai aur export ke liye ek merged model chahiye. |

Mix ke tips: lagbhag **60-70% normal + 30-40% tools** rakho; bina-tool wale examples tools file mein rakho, warna
model har cheez par tool call karega; agar ek skill kharab ho jaye to uske zyada examples daalkar dobara train karo.

### Tools test karna
```bash
python chat.py --tools data/tools_example.json --ask "What's the weather in Paris?" "Hi there"      # base
python chat.py --run combo --tools data/tools_example.json --ask "What's the weather in Paris?"      # trained
```
Dekho: sahi tool, sahi arguments, small talk par call nahi. Training data Python-style quotes (`city='Paris'`) mein
render hota hai jabki base model `city="Paris"` likhta hai, isliye apna parser dono quotes samjhe.

### Limits
230M model single-tool, simple arguments wali calls accha karta hai. Nested arguments, lambi tool list aur
ambiguous requests par kamzor hoga. Held-out set ko automatically score karo (sahi tool? sahi args? bina zaroorat
ke call nahi?), aankh se mat judge karo.

---

## 11. Recipes: har training type ka alag folder

`playground/` mein teen **apne aap mein poore** folders hain. Har ek mein apna data, scripts, `runs/` (adapters),
`exports/` (.gguf files), `run.sh` aur README hai. Jo chahiye wahi chuno; steps sab mein same hain.

| Folder | Kya train karta hai | Kab use karein | `--data` |
|---|---|---|---|
| `playground/1-normal/` | Normal chat (Orbit persona) | Sirf chat/Q&A/style, tools nahi | `chat_orbit.jsonl` |
| `playground/2-tools/` | Sirf tool calling | App functions call karti hai; behtar tool accuracy | `tools_main.jsonl tools_nocall.jsonl:2` |
| `playground/3-normal-plus-tools/` | Dono ek model mein | Style BHI aur tools BHI | `chat_orbit.jsonl:2 tools_main.jsonl tools_nocall.jsonl:2` |

```bash
cd ~/Desktop/llm-models/models/playground/2-tools      # ya 1-normal / 3-normal-plus-tools
bash run.sh data        # data banao        bash run.sh train     # train -> runs/<naam>/
bash run.sh baseline    # bina train model  bash run.sh eval      # base vs trained score
bash run.sh chat        # baat karo         bash run.sh export    # -> exports/<naam>-Q4_K_M.gguf
```

`file.jsonl:N` ka matlab hai "yeh file sirf training hisse mein N baar use karo" (validation mein copy nahi aati), taaki
mix balance ho. `evaluate.py` se held-out prompts par nateeje (chhote tests: trend samjho, pakka number nahi):

| Model | sahi tool (n=32) | exact args | bina zaroorat call nahi (n=4) | Orbit persona (n=3) |
|---|---|---|---|---|
| base (untrained) | 38% | 38% | 100% | 0% |
| `1-normal` | 28% | 28% | 100% | 67% |
| `2-tools` | **97%** | **97%** | 100% | 0% |
| `3-normal-plus-tools` | 91% | 91% | 100% | 67% |

Sirf normal training se tool calling base se **kharab** ho gayi (38% -> 28%); sirf tools se tool score sabse accha par
persona nahi; mixed run mein dono bache. Teeno default 10 epochs par overfit hue (validation loss epoch 2-7 par best
tha), isliye kam epochs (`--epochs 4`) try karo aur `bash run.sh eval` se compare karo.
Scripts har folder mein same copy hain; ek mein sudhaar karo to baaki mein bhi copy karo.
