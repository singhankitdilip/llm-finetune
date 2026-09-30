# 3-normal-plus-tools - Orbit persona + tool calling in one model

Trains **both skills at once**: the Orbit persona and tool calling, in a single run. This avoids the forgetting you get when you train one skill after the other. Use it when one model must chat in your style AND call your tools.

**Dono skills ek saath** train karta hai: Orbit persona aur tool calling, ek hi run mein. Ek ke baad ek train karne par jo bhoolna hota hai, isse woh nahi hota. Jab ek model ko tumhare style mein chat bhi karna ho aur tools bhi call karne ho.

## Files
| File | What it is |
|---|---|
| `make_orbit_data.py` | Builds `data/chat_orbit.jsonl` (normal part) |
| `make_tools_data.py` | Builds `tools_main`, `tools_nocall`, `tools_test` (tools part) |
| `data/` | Both datasets + templates + `tools_example.json` |
| `train.py` | Training with several data files mixed (`file:N` = repeat N times to balance) |
| `chat.py` | Talk to base or trained model (`--tools` gives it the tool list) |
| `evaluate.py` | Tool scores AND persona score |
| `export.py` | Merge + GGUF + quantize -> `exports/` |
| `run.sh` | Short commands for all of the above |
| `runs/` , `exports/` | Trained adapters / `.gguf` files |

## Steps
```bash
cd ~/Desktop/llm-models/models/playground/3-normal-plus-tools
bash run.sh data        # 1. build the data into data/
bash run.sh baseline    # 2. see what the untrained model says
bash run.sh train       # 3. train  ->  runs/<name>/adapter
bash run.sh eval        # 4. score base vs trained on held-out tests
bash run.sh chat        # 5. talk to it (type "exit" to quit)
bash run.sh export      # 6. make exports/<name>-Q4_K_M.gguf  (run it with llama.cpp)
```
The training command inside `run.sh`:
```bash
python train.py --data data/chat_orbit.jsonl:2 data/tools_main.jsonl data/tools_nocall.jsonl:2 --name combo-v1 --r 32
```
Run the exported file:
```bash
cd ../..   # repo root
llama.cpp/build/bin/llama-cli -m playground/3-normal-plus-tools/exports/combo-v1-Q4_K_M.gguf
```

## Result we measured
**Balance zaroori hai.** Tools ka data bahut bada hai, isliye Orbit ko `:2` se 2 baar rakha; warna persona kamzor seekhta hai. Validation loss **epoch 2** par best tha aur train loss ~0 ho gaya, yaani 10 epochs mein bahut overfit hota hai. Pehle `--epochs 4` try karo aur `eval` se compare karo. Ek skill kharab ho to uske examples badhao (ya `:N` badhao) aur dobara train karo.

## Notes
| model | right tool (n=32) | exact args (n=32) | no call when none needed (n=4) | Orbit persona (n=3) |
|---|---|---|---|---|
| base | 38% | 38% | 100% | 0% |
| combo-v1 | **91%** | **91%** | 100% | **67%** |

Both skills are present in one model. Compared with `2-tools` (97% tools, no persona) it gives up ~6 points of tool
accuracy (2 prompts out of 32, within noise) to gain the persona.

**Hinglish:** - **Balance matters.** `chat_orbit.jsonl:2` repeats the 72 Orbit rows twice because the tools data is much bigger; without it the persona is learned weakly. Training tokens: ~14% are trained on (assistant-only).
- Validation loss was best at **epoch 2 of 10** and training loss went to ~0: this run overfits badly at 10 epochs. First thing to try: `NAME=combo-v2 bash run.sh train --epochs 4`, then `bash run.sh eval` to compare.
- If one skill gets worse, add more examples of that skill (or raise its `:N`) and retrain.
