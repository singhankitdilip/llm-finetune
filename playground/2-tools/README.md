# 2-tools - tool calling only

Trains the model to call functions: it writes `<|tool_call_start|>[get_weather(city='Paris')]<|tool_call_end|>`, your app runs the function, and the model uses the result. Starter toolset: `get_weather`, `calculate`, `set_timer`, `web_search`. Replace them with your own tools in `make_data.py`.

Model ko functions call karna sikhata hai: woh `[get_weather(city='Paris')]` likhta hai, tumhari app function chalati hai, model result use karta hai. Starter tools: `get_weather`, `calculate`, `set_timer`, `web_search`. `make_data.py` mein apne tools daal sakte ho.

## Files
| File | What it is |
|---|---|
| `make_data.py` | Builds `tools_main` (calls), `tools_nocall` (no-tool + missing-argument rows) and `tools_test` (held-out phrasings) |
| `data/template_tools.jsonl` | Template of the tool-calling data format |
| `data/tools_example.json` | Tool definitions to pass to `chat.py --tools` |
| `train.py` | Training. Loss only on the assistant's tool call + final reply |
| `chat.py` | Talk to base or trained model (`--tools` gives it the tool list) |
| `evaluate.py` | Right tool? exact arguments? no call when none needed? (held-out prompts) |
| `export.py` | Merge + GGUF + quantize -> `exports/` |
| `run.sh` | Short commands for all of the above |
| `runs/` , `exports/` | Trained adapters / `.gguf` files |

## Steps
```bash
cd ~/Desktop/llm-models/models/playground/2-tools
bash run.sh data        # 1. build the data into data/
bash run.sh baseline    # 2. see what the untrained model says
bash run.sh train       # 3. train  ->  runs/<name>/adapter
bash run.sh eval        # 4. score base vs trained on held-out tests
bash run.sh chat        # 5. talk to it (type "exit" to quit)
bash run.sh export      # 6. make exports/<name>-Q4_K_M.gguf  (run it with llama.cpp)
```
The training command inside `run.sh`:
```bash
python train.py --data data/tools_main.jsonl data/tools_nocall.jsonl:2 --name tools-v1 --r 32
```
Run the exported file:
```bash
cd ../..   # repo root
llama.cpp/build/bin/llama-cli -m playground/2-tools/exports/tools-v1-Q4_K_M.gguf
# (tool calls through llama.cpp need llama-server --jinja, which is not built yet)
```

## Result we measured
`tools_nocall.jsonl:2` ka matlab hai file train mein 2 baar use ho, taaki model small talk par tool call na kare aur argument gayab ho to pooche. Training data Python quotes (`city='Paris'`) mein hai, base `city="Paris"` likhta hai; apna parser dono samjhe. Tests chhote hain (n=32/n=4), asli prompts se bhi check karo. Validation loss epoch 6 ke aas-paas best tha; kam epochs try karo.

## Notes
| model | right tool (n=32) | exact args (n=32) | no call when none needed (n=4) |
|---|---|---|---|
| base | 38% | 38% | 100% |
| tools-v1 | **97%** | **97%** | 100% |

The base model often refuses on new phrasings ("I don't have access to real-time data..."); training fixed that.
The 32 test prompts use phrasings that are NOT in the training data.

**Hinglish:** - `tools_nocall.jsonl:2` means "use that file twice in train" so the model learns to NOT call tools for small talk and to ask when an argument is missing.
- Training data prints calls with Python quotes (`city='Paris'`); the base model writes `city="Paris"`. Make your parser accept both (`evaluate.py` does).
- The tests are small (n=32 / n=4). Expect a few points of noise, and check with your own real prompts before trusting it.
- Validation loss was best around epoch 6 of 10. Try fewer epochs.
