# Playground: three training recipes / Teen training recipes

Each folder is **self-contained**: its own data, scripts, `runs/` (trained adapters), `exports/` (.gguf files) and README.
Pick the folder that matches what you need. / Har folder apne aap mein poora hai; jo chahiye wahi folder chuno.

| Folder | Trains | Pick it when / Kab chuno |
|---|---|---|
| [`1-normal/`](1-normal/) | Normal chat: the **Orbit** persona | Pure chat/Q&A/style, no tools / Sirf chat, tools nahi |
| [`2-tools/`](2-tools/) | **Tool calling** only | App calls functions; best tool accuracy / App functions call karti hai |
| [`3-normal-plus-tools/`](3-normal-plus-tools/) | **Normal + tools** together | One model for both / Ek model dono ke liye |

## Results (same held-out tests, base vs trained)
| Model | right tool (n=32) | exact args (n=32) | no call when none needed (n=4) | Orbit persona (n=3) |
|---|---|---|---|---|
| base (untrained) | 38% | 38% | 100% | 0% |
| `1-normal` (chat-v1) | 28% | 28% | 100% | 67% |
| `2-tools` (tools-v1) | **97%** | **97%** | 100% | 0% |
| `3-normal-plus-tools` (combo-v1) | 91% | 91% | 100% | 67% |

Read this as a **trend, not a precise measurement**: the tests are tiny (a difference of 2 prompts = 6 points).
The pattern is clear, though: normal-only training made tool calling *worse* than the base (38% -> 28%), tools-only
gave the best tool score but no persona, and the mixed run kept both.
(Tool numbers come from the tools starter dataset: 4 tools. Persona from 3 unseen questions.)

## Same steps in every folder
```bash
cd ~/Desktop/llm-models/models/playground/<folder>
bash run.sh data        # build the data
bash run.sh baseline    # ask the untrained model
bash run.sh train       # train   (NAME=v2 bash run.sh train --epochs 5  for a new version)
bash run.sh eval        # score base vs trained
bash run.sh chat        # talk to it
bash run.sh export      # -> exports/<name>-Q4_K_M.gguf
```
`run.sh` only shortens the commands; the real scripts (`train.py`, `chat.py`, `export.py`, `evaluate.py`) are in each
folder and take more options (`python train.py --help`). The scripts are identical copies in all three folders, so if
you improve one, copy the change to the others.

Full guide (concepts, settings, troubleshooting): [English](../README.md) | [Hinglish](../README.hinglish.md)
