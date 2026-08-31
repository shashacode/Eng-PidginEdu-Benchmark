#!/usr/bin/env bash
# ============================================================
# Launch the English -> Nigerian Pidgin finetuning run.
#
#   ./run_train.sh                    # afrimt5 on every visible GPU
#   ./run_train.sh mt5                # pick a different model
#   ./run_train.sh nllb --epochs 3    # extra flags go to train.py
#
# Env overrides:
#   NUM_GPUS=1 ./run_train.sh         # force a GPU count
#   CUDA_VISIBLE_DEVICES=0 ./run_train.sh
# ============================================================

set -euo pipefail

# Portable by construction: resolves to wherever this repo was cloned,
# not a path specific to the machine this was developed on. Override
# PYTHON/TORCHRUN if your virtualenv's bin/ isn't already on PATH.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"

MODEL_KEY="${1:-afrimt5}"
shift || true          # anything left over is forwarded to train.py

cd "$PROJECT_DIR"

# ------------------------------------------------------------
# How many GPUs are we actually allowed to use?
# ------------------------------------------------------------
if [[ -z "${NUM_GPUS:-}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        # Count the comma-separated entries the user pinned.
        NUM_GPUS=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
    elif command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS=$(nvidia-smi -L | wc -l)
    else
        NUM_GPUS=0
    fi
fi

# ------------------------------------------------------------
# Runtime environment
# ------------------------------------------------------------

# Each rank gets its own dataloader workers; without this OpenMP grabs
# every core per process and the ranks fight over the CPU.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# The fast tokenizers are already used inside forked dataloader workers.
export TOKENIZERS_PARALLELISM=false

# Long fp32 runs fragment the allocator; expandable segments reclaim it.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Set NCCL_DEBUG=INFO before invoking this script to debug rank hangs.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# torch 2.13 dispatches some ops (hit during generation, not training) to
# Triton, which JIT-compiles a C helper and therefore needs the CPython
# headers. On the machine this was developed on, python3.12-dev wasn't
# installed and there was no root access, so the headers were extracted
# from the .deb into ~/.local/include and gcc was pointed at them via
# CPATH (REPRODUCE.md has the exact extraction steps). If your system
# already has python3-dev / python3.12-dev installed normally, none of
# this is needed -- this block is a best-effort fallback, not a
# requirement, and only warns rather than failing if the headers aren't
# found in this specific fallback location. Without SOME source of
# Python.h, training runs fine and then dies at the first evaluation
# with "Python.h: No such file or directory".
PYHEADERS="${PYHEADERS:-$HOME/.local/include}"
PY_MINOR_HEADER_DIR="python$(${PYTHON} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

if [[ -f "$PYHEADERS/$PY_MINOR_HEADER_DIR/Python.h" ]]; then
    export CPATH="${CPATH:+$CPATH:}$PYHEADERS:$PYHEADERS/$PY_MINOR_HEADER_DIR"
elif echo '#include <Python.h>' | gcc -E - >/dev/null 2>&1; then
    : # already resolvable via the normal system include path -- nothing to do
else
    echo "WARNING: Python.h not found via the system include path or"
    echo "         $PYHEADERS/$PY_MINOR_HEADER_DIR -- Triton JIT compilation"
    echo "         will fail at the first evaluation. Install python3-dev"
    echo "         (or your distro's equivalent) or see REPRODUCE.md for"
    echo "         the no-root fallback used during development."
fi

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${MODEL_KEY}_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================"
echo " model    : $MODEL_KEY"
echo " GPUs     : $NUM_GPUS"
echo " extra    : ${*:-<none>}"
echo " log      : $LOG_FILE"
echo "================================================================"

# Log everything from here on without a pipeline, so the trainer's own
# exit status is what this script reports.
exec > >(tee "$LOG_FILE") 2>&1

# ------------------------------------------------------------
# Shutdown
# ------------------------------------------------------------
# torchrun's ranks and their dataloader workers do not die with the
# parent shell. On Ctrl-C or a timeout that strands them on the GPUs,
# still holding their memory. setsid puts the whole run in its own
# process group so a single negative-PID kill takes all of it down.

TRAIN_PID=""

cleanup() {

    if [[ -z "$TRAIN_PID" ]]; then
        return
    fi

    if kill -0 "$TRAIN_PID" 2>/dev/null; then

        echo ""
        echo "shutting down run (pgid $TRAIN_PID)..."

        kill -TERM -- "-$TRAIN_PID" 2>/dev/null || true

        # Give the ranks a moment to release their CUDA context.
        for _ in {1..10}; do
            kill -0 "$TRAIN_PID" 2>/dev/null || break
            sleep 1
        done

        kill -KILL -- "-$TRAIN_PID" 2>/dev/null || true
    fi
}

trap cleanup INT TERM EXIT

# ------------------------------------------------------------
# Launch
# ------------------------------------------------------------
# Always launched through torchrun, even for a single GPU: plain `python
# train.py` crashes with "Default process group has not been initialized"
# under the installed accelerate/transformers versions, because
# accelerate's PartialState reaches a distributed-only code path
# whenever CUDA is available, but only torchrun sets LOCAL_RANK, which
# is what actually gates the corresponding init_process_group() call.
# See BENCHMARK_REPORT.md §11.1. --standalone picks a free rendezvous
# port on this host either way.
setsid "$TORCHRUN" \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS:-1}" \
    train.py --model "$MODEL_KEY" "$@" &

TRAIN_PID=$!

STATUS=0
wait "$TRAIN_PID" || STATUS=$?

exit "$STATUS"
