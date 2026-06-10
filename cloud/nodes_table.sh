#!/usr/bin/env bash
# nodes_table.sh — per-node training status table
# Usage: ./nodes_table.sh [run_id]
#        ./nodes_table.sh --init       deploy read_test_acc.py to all nodes (run once)

set -uo pipefail
cd "$(dirname "$0")"

KEY=~/.ssh/decentr_id_ed25519
BASTION=84.201.157.237
BASTION_USER=decentr
TIMEOUT=25
HELPER_LOCAL="../patches/read_test_acc.py"
HELPER_REMOTE="/opt/decentr/read_test_acc.py"

SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=$TIMEOUT -o BatchMode=yes -o LogLevel=ERROR"

# ── --init: deploy read_test_acc.py to all 20 nodes ──────────────────────────
if [[ "${1:-}" == "--init" ]]; then
  echo "Deploying $HELPER_LOCAL to all nodes at $HELPER_REMOTE ..."
  scp -q $SSH_OPTS "$HELPER_LOCAL" "${BASTION_USER}@${BASTION}:${HELPER_REMOTE}"
  echo "  decentr-01 (bastion) ✓"

  FOLLOWER_IPS=$(python3 -c "
import yaml
with open('pool.yaml') as f:
    pool = yaml.safe_load(f)
for n in pool['nodes'][1:]:
    print(n['id'] + ' ' + n['ssh_host'])
")
  ssh $SSH_OPTS "${BASTION_USER}@${BASTION}" "
KEY=~/.ssh/decentr_id_ed25519
$(while IFS=' ' read -r nid ip; do
  echo "scp -q -i \$KEY -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=8 $HELPER_REMOTE decentr@$ip:$HELPER_REMOTE && echo '  $nid ✓' || echo '  $nid ✗' &"
done <<< "$FOLLOWER_IPS")
wait" 2>/dev/null
  echo "Done."
  exit 0
fi

# Clean up any stale temp files from killed/crashed previous runs
rm -f /tmp/nodes_table.*.sh 2>/dev/null

# ── RUN_ID ────────────────────────────────────────────────────────────────────
if [ -n "${1:-}" ]; then
  RUN_ID="$1"
else
  RUN_ID=$(ssh $SSH_OPTS "${BASTION_USER}@${BASTION}" \
    "ls -t /tmp/dc_boot_*.log 2>/dev/null | head -1 | sed 's|/tmp/dc_boot_||;s|\.log||'" 2>/dev/null)
fi
[ -z "$RUN_ID" ] && echo "No run found." && exit 1

# ── Follower list ─────────────────────────────────────────────────────────────
FOLLOWER_LIST=$(python3 -c "
import yaml
with open('pool.yaml') as f:
    pool = yaml.safe_load(f)
for n in pool['nodes'][1:]:
    print(n['id'] + ' ' + n['ssh_host'])
")

TOTAL_EPOCHS=$(grep -A5 'sync_static' "$(dirname "$0")/suites/cifar10_wan_campaign.yaml" \
  | grep 'epochs:' | head -1 | awk '{print $2}')
TOTAL_EPOCHS=${TOTAL_EPOCHS:-50}

echo "Run: $RUN_ID"
date

# ── Build bastion script ──────────────────────────────────────────────────────
TMP_LOCAL=$(mktemp "/tmp/nodes_table.XXXXXX.sh")
trap 'rm -f "$TMP_LOCAL"' EXIT

# Inject RUN_ID and node list (simple strings, safe to expand locally)
cat > "$TMP_LOCAL" << HEADER
#!/bin/bash
RUN_ID="${RUN_ID}"
NODES=(
HEADER

while IFS=' ' read -r nid ip; do
  printf '  "%s %s"\n' "$nid" "$ip" >> "$TMP_LOCAL"
done <<< "$FOLLOWER_LIST"

# Single-quoted marker: no local expansion — all $ belong to bastion runtime
cat >> "$TMP_LOCAL" << 'BODY'
)
KEY=~/.ssh/decentr_id_ed25519
SOPTS="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 -o BatchMode=yes -o LogLevel=ERROR"
ART=/opt/decentr/DistLearn/Decentr_my_own/artifacts/checkpoints
HELPER=/opt/decentr/read_test_acc.py

check_follower() {
  local node_id="$1" ip="$2"
  local out rc
  out=$(ssh $SOPTS "decentr@${ip}" "
    # grep python first: the ssh/bash wrapper running THIS command also has
    # '(start|join)-run' in its argv — match the actual trainer interpreter only.
    proc=\$(ps aux | grep python | grep -E '(start|join)-run' | grep -v grep | head -1)
    if [ -z \"\$proc\" ]; then printf 'DEAD|0|0|0|-\n'; exit 0; fi
    cpu=\$(echo \"\$proc\" | awk '{print \$3}')
    mem=\$(echo \"\$proc\" | awk '{print \$4}')
    ckpt=\$(find ${ART}/${RUN_ID}/${node_id} -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')
    test_acc=\$(python3 ${HELPER} ${ART}/${RUN_ID}/${node_id} 2>/dev/null || echo -)
    printf 'RUNNING|%s|%s|%s|%s\n' \"\$cpu\" \"\$mem\" \"\$ckpt\" \"\$test_acc\"
  " 2>/dev/null)
  rc=$?
  if [[ $rc -ne 0 || -z "$out" ]]; then
    echo "${node_id}|${ip}|UNREACHABLE|0|0|0|-"
  else
    echo "${node_id}|${ip}|${out}"
  fi
}

# Bootstrap runs locally on bastion.
proc=$(ps aux | grep python | grep -E "(start|join)-run" | grep -v grep | head -1)
if [ -n "$proc" ]; then
  cpu=$(echo "$proc" | awk '{print $3}')
  mem=$(echo "$proc" | awk '{print $4}')
  ckpt=$(find ${ART}/${RUN_ID}/decentr-01 -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')
  test_acc=$(python3 ${HELPER} ${ART}/${RUN_ID}/decentr-01 2>/dev/null || echo -)
  echo "decentr-01|192.168.199.14|RUNNING|${cpu}|${mem}|${ckpt}|${test_acc}"
else
  echo "decentr-01|192.168.199.14|DEAD|0|0|0|-"
fi

for entry in "${NODES[@]}"; do
  read -r node_id ip <<< "$entry"
  check_follower "$node_id" "$ip" &
done
wait
BODY

# ── Deploy and run ────────────────────────────────────────────────────────────
TMP_REMOTE="/tmp/nodes_table_$$.sh"
scp -q $SSH_OPTS "$TMP_LOCAL" "${BASTION_USER}@${BASTION}:${TMP_REMOTE}"

RAW=$(ssh $SSH_OPTS "${BASTION_USER}@${BASTION}" "bash $TMP_REMOTE; rm -f $TMP_REMOTE" 2>&1)

# ── Print table ───────────────────────────────────────────────────────────────
echo ""
printf '%-13s  %-11s  %-6s  %-6s  %-9s  %-9s  %s\n' \
  "Node" "Status" "CPU%" "Mem%" "Epoch" "TestAcc" "Phase"
printf '%s\n' "$(printf '─%.0s' {1..70})"

render_row() {
  local node_id="$1"
  local line status cpu mem epoch test_acc epoch_str phase sym

  line=$(echo "$RAW" | grep "^${node_id}|" || true)
  if [ -z "$line" ]; then
    printf '%-13s  %-11s  %-6s  %-6s  %-9s  %-9s  %s\n' \
      "$node_id" "? NOREPLY" "-" "-" "-/-" "-" "-"
    return
  fi

  IFS='|' read -r _nid _ip status cpu mem epoch test_acc <<< "$line"

  case "$status" in
    RUNNING)
      sym="✓"
      epoch_str="${epoch}/${TOTAL_EPOCHS}"
      if awk -v c="$cpu" 'BEGIN{exit !(c+0 > 50)}'; then phase="computing"
      else phase="syncing"; fi
      ;;
    DEAD)
      sym="✗"; epoch_str="-/-"; phase="stopped"; cpu="-"; mem="-"; test_acc="-";;
    UNREACHABLE)
      sym="?"; epoch_str="-/-"; phase="unreachable"; cpu="-"; mem="-"; test_acc="-";;
    *)
      sym="?"; epoch_str="-/-"; phase="$status"; test_acc="${test_acc:--}";;
  esac

  printf '%-13s  %s %-9s  %-6s  %-6s  %-9s  %-9s  %s\n' \
    "$node_id" "$sym" "$status" "$cpu" "$mem" "$epoch_str" "$test_acc" "$phase"
}

render_row "decentr-01"
while IFS=' ' read -r nid _ip; do
  render_row "$nid"
done <<< "$FOLLOWER_LIST"

echo ""
echo "Legend: ✓ running  ✗ dead  ? unreachable   Phase: CPU>50% = computing / else = syncing"
echo "TestAcc: test-set accuracy from latest checkpoint (run --init once to enable)"
