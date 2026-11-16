set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

ROLL_NUMBER="${ROLL_NUMBER:2026701016}"
HF_REPO="${HF_REPO:vidhyavasan/anlp-a3}"
BEST_MODEL="${BEST_MODEL:-mamba}"
OUTPUT_DIR="output/part1"
MODE="test"

for arg in "$@"; do
  case "$arg" in
    --heldout) MODE="heldout" ;;
    *) echo "unknown argument: $arg" >&2; exit 1 ;;
  esac
done

mkdir -p "$OUTPUT_DIR"

if [ "$MODE" = "heldout" ]; then
  uv run python src/evaluate.py \
    --mode heldout \
    --data data/part1/heldout \
    --models "$BEST_MODEL" \
    --hf-repo-id "$HF_REPO" \
    --output "$OUTPUT_DIR/heldout_result_${ROLL_NUMBER}.jsonl"
else
  uv run python src/evaluate.py \
    --mode test \
    --data data/part1/test \
    --models transformer mamba deltanet hybrid \
    --hf-repo-id "$HF_REPO" \
    --output-dir "$OUTPUT_DIR"
fi