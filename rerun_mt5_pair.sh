#!/usr/bin/env bash
# ============================================================
# Re-run the mT5 pair with the glossing fix.
#
# Both models translated well at the original settings but never learned
# to emit terminology glosses (29 and 42 glosses produced against 2726
# expected, vs afriteva's 1914). Two changes:
#
#   learning rate 5e-5/3e-5 -> 1e-4   (set in train.py MODEL_CONFIGS)
#   early-stopping patience 4 -> 6    (mt5 previously stopped at step
#                                      3500 of 6560, best checkpoint 1500)
#
# The originals are preserved in baseline_runs/ and their metrics in
# results_baseline_lr3e5/, so the comparison survives.
# ============================================================

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

for model in mt5 afrimt5; do

    echo "################################################################"
    echo ">>> $model  ($(date +%H:%M:%S))"
    echo "################################################################"

    START=$SECONDS

    ./run_train.sh "$model" --early-stopping-patience 6

    echo ">>> $model finished (exit $?) in $((SECONDS - START))s"

    sleep 10

done

echo "################################################################"
echo " re-run complete -- aggregating"
echo "################################################################"

"${PYTHON:-python}" aggregate_results.py