#!/usr/bin/env bash
# Shortcuts for this folder.   Usage:  bash run.sh <command> [extra train.py flags]
#   data      build the training data
#   baseline  ask the UNTRAINED model a few questions
#   train     train (run name = $NAME, default combo-v1);  e.g.  NAME=v2 bash run.sh train --epochs 5
#   chat      talk to the trained model
#   eval      score base + trained model on held-out tests
#   export    make a .gguf in exports/
cd "$(dirname "$0")"
source ../../.venv/bin/activate
NAME=${NAME:-combo-v1}
case "${1:-}" in
  data)     python make_orbit_data.py && python make_tools_data.py ;;
  baseline) python chat.py --tools data/tools_example.json --ask "Who are you?" "What's the weather in Paris?" "Hello" ;;
  train)    python train.py --data data/chat_orbit.jsonl:2 data/tools_main.jsonl data/tools_nocall.jsonl:2 --name "$NAME" --r 32 "${@:2}" ;;
  chat)     python chat.py --run "$NAME" --tools data/tools_example.json ;;
  eval)     python evaluate.py --runs "$NAME" ;;
  export)   python export.py --run "$NAME" ;;
  *) sed -n '2,9p' "$0" ;;
esac
