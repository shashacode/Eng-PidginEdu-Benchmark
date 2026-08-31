#!/usr/bin/env bash
# ============================================================
# Run the full benchmark sweep, cheapest model first.
#
#   ./run_benchmark.sh                      # every model, skipping finished ones
#   ./run_benchmark.sh mt5 nllb             # only these
#   FORCE=1 ./run_benchmark.sh              # re-run models that already finished
#   EPOCHS=3 ./run_benchmark.sh             # fewer epochs for a dry pass
#
# A model that fails does not stop the sweep -- its status is recorded and
# the next one starts. Re-running the script picks up where it left off.
# ============================================================

set -uo pipefail          # deliberately no -e: a failed model must not abort the sweep

# Portable by construction -- resolves to wherever this repo was cloned.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

cd "$PROJECT_DIR"

# All 12 models that support full fine-tuning under this pipeline's
# hardware (2x32GB GPU), ordered smallest to largest: cheap failures
# surface before expensive ones. madlad3b/t5-v1_1-xl are NOT included
# here -- they hit a genuine memory ceiling under full fine-tuning
# (BENCHMARK_REPORT.md §9.3) and only support LoRA; run those with
# `./run_train.sh madlad3b --lora` / `./run_train.sh t5v11xl --lora`
# instead, or see README.md for the full 14-model reproduction guide.
#
# There is no predetermined "flagship" model in this list -- the
# benchmark's own methodology (§9.5) picks the flagship (PidginEdu-LLM)
# empirically from whichever model scores best after every model here
# has actually been trained and evaluated, not in advance. As finalized
# in BENCHMARK_REPORT.md, that turned out to be mt5_large.
ALL_MODELS=(
    afriteva            # 229M
    m2m100               # 418M
    mt5                  # 580M
    afrimt5              # 580M
    nllb                 # 600M
    mbart50              # 680M
    afriteva_v2_large    # 1B
    mt5_large            # 1.2B
    cheetah              # 1.2B
    toucan               # 1.2B
    seamless             # 1.37B
    m2m100_1.2b          # 1.24B
)

if [[ $# -gt 0 ]]; then
    MODELS=("$@")
else
    MODELS=("${ALL_MODELS[@]}")
fi

EPOCHS="${EPOCHS:-5}"
FORCE="${FORCE:-0}"

STATUS_FILE="$PROJECT_DIR/benchmark_status.tsv"

if [[ ! -f "$STATUS_FILE" ]]; then
    printf "model\tstatus\tseconds\tfinished\n" > "$STATUS_FILE"
fi

record() {
    printf "%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$(date -Iseconds)" >> "$STATUS_FILE"
}

echo "################################################################"
echo " benchmark sweep: ${MODELS[*]}"
echo " epochs: $EPOCHS | force: $FORCE"
echo "################################################################"

for model in "${MODELS[@]}"; do

    OUT_DIR="$PROJECT_DIR/output_${model}"

    # A finished run is one that produced its glossary report.
    if [[ -f "$OUT_DIR/glossary_report.json" && "$FORCE" != "1" ]]; then
        echo ""
        echo ">>> $model: already finished, skipping (FORCE=1 to redo)"
        record "$model" "skipped" 0
        continue
    fi

    echo ""
    echo "################################################################"
    echo ">>> $model  ($(date +%H:%M:%S))"
    echo "################################################################"

    START=$SECONDS

    ./run_train.sh "$model" --epochs "$EPOCHS"
    CODE=$?

    ELAPSED=$((SECONDS - START))

    if [[ $CODE -eq 0 ]]; then
        echo ">>> $model: done in ${ELAPSED}s"
        record "$model" "ok" "$ELAPSED"
    else
        echo ">>> $model: FAILED (exit $CODE) after ${ELAPSED}s -- continuing"
        record "$model" "failed_$CODE" "$ELAPSED"
    fi

    # Let the driver settle before the next model claims the GPUs.
    sleep 10

done

echo ""
echo "################################################################"
echo " sweep complete -- aggregating"
echo "################################################################"

"$PYTHON" aggregate_results.py

echo ""
echo "status log: $STATUS_FILE"
