#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
llm prepare-data --data-config configs/data_small.yaml --vocab-size 8192
python scripts/prepare_preference.py data/processed data/tokenizer 4000
llm train --config configs/tiny.yaml
llm finetune --config configs/tiny_sft.yaml
llm finetune --config configs/tiny_dpo.yaml
llm evaluate --checkpoint checkpoints/tiny_sft --output experiments/tiny_sft/evaluation.json
