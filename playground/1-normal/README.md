# 1-normal - Orbit persona (normal chat training)

Teaches the model a small persona called **Orbit** from ~72 chat examples. The base model has never heard of Orbit, so if it answers as Orbit, it learned from your data. Use this for style/persona/Q&A models without tools.

Model ko ek chhota persona **Orbit** sikhata hai (~72 chat examples). Base model Orbit ko nahi jaanta, isliye Orbit bole to matlab data se seekha. Tools ke bina style/persona/Q&A ke liye.

## Files
| File | What it is |
|---|---|
| `make_data.py` | Builds `data/chat_orbit.jsonl` (the Orbit persona) |
| `data/template.jsonl` | Template of the chat data format |
| `train.py` | Training (LoRA). Loss only on the assistant's answers |
| `chat.py` | Talk to base or trained model |
| `evaluate.py` | Persona score (answer must mention "Orbit" on 3 unseen questions) |
| `export.py` | Merge + GGUF + quantize -> `exports/` |
| `run.sh` | Short commands for all of the above |
| `runs/` | Trained adapters, one folder per run |
| `exports/` | Your `.gguf` files |

## Steps
```bash
cd ~/Desktop/llm-models/models/playground/1-normal
bash run.sh data        # 1. build the data into data/
bash run.sh baseline    # 2. see what the untrained model says
bash run.sh train       # 3. train  ->  runs/<name>/adapter
bash run.sh eval        # 4. score base vs trained on held-out tests
bash run.sh chat        # 5. talk to it (type "exit" to quit)
bash run.sh export      # 6. make exports/<name>-Q4_K_M.gguf  (run it with llama.cpp)
```
The training command inside `run.sh`:
```bash
python train.py --data data/chat_orbit.jsonl --name chat-v1
```
Run the exported file:
```bash
cd ../..   # repo root
llama.cpp/build/bin/llama-cli -m playground/1-normal/exports/chat-v1-Q4_K_M.gguf
```

## Result we measured
Data mein har jawab kai sawalon ke saath repeat hota hai, isliye validation loss asli se achha dikhta hai; `eval` aur `chat` se judge karo. Validation loss epoch 7 ke aas-paas best tha, phir badha (overfitting); `--epochs 6` try karo. 230M model mein facts pakke nahi hote, style/persona achha seekhta hai.

## Notes
| model | Orbit persona (n=3 unseen questions) | right tool (n=32, for reference) |
|---|---|---|
| base | 0% | 38% |
| chat-v1 | 67% | **28%** |

The persona was learned, but tool ability got **worse** than the base model (38% -> 28%) because this run never sees
tools. If you ever need tools too, use `3-normal-plus-tools/`.

**Hinglish:** - The data repeats each answer for several questions, so validation loss looks better than it really is. Judge by `eval`/`chat`, not loss alone.
- Validation loss was best around epoch 7 of 10 and then rose (overfitting). Try `NAME=chat-v2 bash run.sh train --epochs 6`.
- Facts are not reliable in a 230M model; style/persona is what it learns well.
