# Training playground - quick reference / Quick reference

Full guide / Poori guide: [English](../README.md) | [Hinglish](../README.hinglish.md)

```bash
cd ~/Desktop/llm-models/models/playground
source ../.venv/bin/activate
```

| Step | Command | English | Hinglish |
|---|---|---|---|
| 1 | put data in `data/*.jsonl` (see `data/template.jsonl`) | One JSON per line: `{"messages":[user, assistant]}` | Har line ek JSON: `{"messages":[user, assistant]}` |
| 2 | `python chat.py --ask "question"` | Baseline before training | Training se pehle baseline |
| 3 | `python train.py --data data/mydata.jsonl --name v1` | Train (knobs: `--epochs --lr --r --batch`) | Train karo (settings: `--epochs --lr --r --batch`) |
| 4 | `python chat.py --run v1` | Chat with your model | Apne model se baat |
| 5 | `python export.py --run v1` | Make `../v1-Q4_K_M.gguf` | `../v1-Q4_K_M.gguf` banao |

Val loss rising while train loss falls = overfitting. / Val loss badhe aur train loss ghate = overfitting.
Each run lives in `runs/<name>/`. / Har run `runs/<naam>/` mein alag rehta hai.
