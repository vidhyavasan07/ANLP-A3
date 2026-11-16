set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DATA_DIR="${DATA_DIR:-data/part1}"
OUTPUT_DIR="output/part1/training"
MODELS="${MODELS:-transformer mamba deltanet}"
export WANDB_MODE="${WANDB_MODE:-offline}"

for model in $MODELS; do
  uv run python src/train.py \
    --data-dir "$DATA_DIR" \
    --model "$model" \
    --output-dir "$OUTPUT_DIR/$model" \
    --epochs 1 \
    --steps-per-epoch 50 \
    --batch-size 8 \
    --max-seq-len 256 \
    --max-validation-samples 50
done