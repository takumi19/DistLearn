#!/usr/bin/env bash
# run_selected.sh — run a curated experiment list in order.
#
# Each experiment runs via orchestrate --only, which already:
#   pulls results → git-pushes to decentr-results → cleans nodes → returns.
# So this driver just sequences them. One failing run does NOT stop the rest
# (orchestrate marks it timeout and moves on); check campaign-state/ afterwards.
#
# Usage: ./run_selected.sh            # run the list below
#        ./run_selected.sh --list     # print the list and exit
set -uo pipefail
cd "$(dirname "$0")"

# Curated set (user-selected). Revised 2026-06-09: reliability_aware_graph dropped
# (now enabled:false in suite); added topology(star) + frequency(low_comm_100) +
# Phase-3 payload (delta + top-k compression). orchestrate auto-[skip]s already-done
# experiments (campaign-state) and disabled ones, so listing them here is safe.
EXPERIMENTS=(
  # ── done this campaign (auto-skip via campaign-state) ──
  #   random_fanout_1   top_k_fastest   top_k_reliable
  #   ping_aware                 # running in the live driver
  #   reliability_aware_graph    # REMOVED for now → enabled:false in suite
  # ── remaining plan ──
  dynamic_graph              # Phase 3: rebuild active graph every 5 windows
  star_topology              # topology: hub/star — bottleneck + single point of failure
  low_comm_100               # frequency: push every 100 steps (10x sparser)
  delta_exchange             # Phase 3: lossless weight deltas (payload shrinks late)
  compress_topk              # Phase 3: top-k 10% sparsified (~10x) — aggressive
)

if [[ "${1:-}" == "--list" ]]; then
  printf '%s\n' "${EXPERIMENTS[@]}"
  exit 0
fi

SUITE=suites/cifar10_wan_campaign.yaml
LOG=/tmp/run_selected.log

echo "Refreshing YC token…"
export YC_TOKEN=$(~/yandex-cloud/bin/yc iam create-token 2>/dev/null)

echo "Driver started $(date)  |  ${#EXPERIMENTS[@]} experiments" | tee "$LOG"
for exp in "${EXPERIMENTS[@]}"; do
  echo "" | tee -a "$LOG"
  echo "═══ $(date '+%H:%M:%S')  RUN: $exp ═══" | tee -a "$LOG"
  # Refresh token before each (a run can outlive the 12h token; harmless if early).
  export YC_TOKEN=$(~/yandex-cloud/bin/yc iam create-token 2>/dev/null)
  PYTHONUNBUFFERED=1 python3 orchestrate.py run --pool pool.yaml \
    --suite "$SUITE" --only "$exp" --max-runtime-hours 9 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "═══ Driver done $(date) ═══" | tee -a "$LOG"
python3 orchestrate.py status --suite "$SUITE" | tee -a "$LOG"
