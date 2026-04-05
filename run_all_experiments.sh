#!/usr/bin/env bash
# run_all_experiments.sh
# Runs all 10 experiments sequentially using `uv run` in WSL.
# Usage: bash run_all_experiments.sh

set -euo pipefail

# Resolve the project root to the directory containing this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOG_DIR/run_all_${TIMESTAMP}.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

# ── Config list ────────────────────────────────────────────────────────────────
CONFIGS=(
    # DeepSeek-R1-Distill-Qwen-1.5B  (max_new_tokens = 8192)
    "config/DS_cot.json"
    "config/DS_consistency.json"
    "config/DS_refine.json"
    "config/DS_ltm.json"
    "config/DS_autocot.json"
    # Qwen2.5-Math-1.5B              (max_new_tokens = 3072)
    "config/Qwen_cot.json"
    "config/Qwen_consistency.json"
    "config/Qwen_refine.json"
    "config/Qwen_ltm.json"
    "config/Qwen_autocot.json"
)

TOTAL=${#CONFIGS[@]}
PASS=0
FAIL=0

log "Starting $TOTAL experiments. Master log: $MASTER_LOG"
log "======================================================"

# ── Main loop ──────────────────────────────────────────────────────────────────
for i in "${!CONFIGS[@]}"; do
    CFG="${CONFIGS[$i]}"
    IDX=$((i + 1))
    NAME="$(basename "$CFG" .json)"
    EXP_LOG="$LOG_DIR/${NAME}_${TIMESTAMP}.log"

    log "[$IDX/$TOTAL] Starting: $CFG"

    START=$(date +%s)
    if uv run python scripts/run_experiment.py --config "$CFG" \
            > >(tee -a "$EXP_LOG") \
            2> >(tee -a "$EXP_LOG" >&2); then
        END=$(date +%s)
        ELAPSED=$(( END - START ))
        log "[$IDX/$TOTAL] DONE: $NAME  (${ELAPSED}s)"
        PASS=$((PASS + 1))
    else
        END=$(date +%s)
        ELAPSED=$(( END - START ))
        log "[$IDX/$TOTAL] FAILED: $NAME  (${ELAPSED}s)  — see $EXP_LOG"
        FAIL=$((FAIL + 1))
        # Continue with remaining experiments even if one fails.
    fi

    log "------------------------------------------------------"
done

# ── Summary ────────────────────────────────────────────────────────────────────
log "======================================================"
log "All done.  Passed: $PASS / $TOTAL   Failed: $FAIL / $TOTAL"
log "Logs saved to: $LOG_DIR/"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
