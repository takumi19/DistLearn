#!/usr/bin/env bash
# kill_cluster.sh — kill all decentr-my-own processes on every cluster node.
# Usage: ./kill_cluster.sh [--verify]
#   --verify  only check process counts, don't kill

set -uo pipefail

cd "$(dirname "$0")"

KEY="${HOME}/.ssh/decentr_id_ed25519"
BASTION="84.201.157.237"
USER="decentr"
BASTION_KEY="~/.ssh/decentr_id_ed25519"
TIMEOUT=25

VERIFY=0
[[ "${1:-}" == "--verify" ]] && VERIFY=1

# Parse pool.yaml → "node_id ssh_host is_follower(0|1)" per line
NODE_LIST=$(python3 - <<'PY'
import yaml
with open("pool.yaml") as f:
    data = yaml.safe_load(f)
for n in data["nodes"]:
    is_follower = int("ssh_jump" in n)
    print(n["id"], n.get("ssh_host", n.get("host")), is_follower)
PY
)

# Build the follower kill/check script locally, then scp it to the bastion and
# run it there (one SSH per follower, fired in parallel from the bastion side).
TMP_LOCAL=$(mktemp "/tmp/kill_cluster.XXXXXX.sh")
trap 'rm -f "$TMP_LOCAL"' EXIT

cat > "$TMP_LOCAL" << SHEOF
#!/bin/bash
SSH_OPTS="-i $BASTION_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=$TIMEOUT -o BatchMode=yes -o LogLevel=ERROR"
KILL_CMD='pkill -9 -f "decentr-my-own (start|join)-run" 2>/dev/null; sleep 0.3; ps aux | grep -E "decentr-my-own.*(start|join)-run" | grep -v grep | wc -l'
CHECK_CMD='ps aux | grep -E "decentr-my-own.*(start|join)-run" | grep -v grep | wc -l'
SHEOF

if [[ $VERIFY -eq 1 ]]; then
  echo 'CMD=$CHECK_CMD' >> "$TMP_LOCAL"
else
  echo 'CMD=$KILL_CMD' >> "$TMP_LOCAL"
fi

# Add kill command for each follower
while read -r node ip is_follower; do
  [[ "$is_follower" == "0" ]] && continue
  cat >> "$TMP_LOCAL" << SHEOF
( out=\$(ssh \$SSH_OPTS decentr@$ip "\$CMD" 2>&1); rc=\$?
  if [[ \$rc -ne 0 ]]; then echo "$node $ip UNREACHABLE"
  else n=\$(echo "\$out" | tail -1 | tr -d '[:space:]'); echo "$node $ip \$n"
  fi ) &
SHEOF
done <<< "$NODE_LIST"

echo 'wait' >> "$TMP_LOCAL"

# ── Kill bootstrap directly (local SSH) ──────────────────────────────────────
if [[ $VERIFY -eq 1 ]]; then
  BOOTSTRAP_CMD='ps aux | grep -E "decentr-my-own.*(start|join)-run" | grep -v grep | wc -l'
else
  BOOTSTRAP_CMD='pkill -9 -f "decentr-my-own (start|join)-run" 2>/dev/null; sleep 0.3; ps aux | grep -E "decentr-my-own.*(start|join)-run" | grep -v grep | wc -l'
fi

boot_out=$(ssh -i "$KEY" \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout="$TIMEOUT" \
  -o BatchMode=yes \
  -o LogLevel=ERROR \
  "${USER}@${BASTION}" "$BOOTSTRAP_CMD" 2>&1)
boot_rc=$?

if [[ $boot_rc -ne 0 ]]; then
  BOOT_STATUS="✗ UNREACHABLE"
else
  n=$(echo "$boot_out" | tail -1 | tr -d '[:space:]')
  if [[ $VERIFY -eq 1 ]]; then
    [[ "$n" == "0" ]] && BOOT_STATUS="✓ clean" || BOOT_STATUS="✗ ${n} procs"
  else
    [[ "$n" == "0" ]] && BOOT_STATUS="✓ killed" || BOOT_STATUS="⚠ ${n} procs left"
  fi
fi

# ── Run the follower kill script on bastion ───────────────────────────────────
TMP_REMOTE="/tmp/kill_cluster_$$.sh"
scp -q -i "$KEY" \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout="$TIMEOUT" \
  -o BatchMode=yes \
  -o LogLevel=ERROR \
  "$TMP_LOCAL" "${USER}@${BASTION}:${TMP_REMOTE}"

RAW=$(ssh -i "$KEY" \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout="$TIMEOUT" \
  -o BatchMode=yes \
  -o LogLevel=ERROR \
  "${USER}@${BASTION}" "bash $TMP_REMOTE; rm -f $TMP_REMOTE" 2>&1)

# ── Format and print results in pool order ───────────────────────────────────
printf '%-14s  %-18s  %s\n' "Node" "SSH host" "Status"
printf '%s\n' "$(printf '─%.0s' {1..55})"
printf '%-14s  %-18s  %s\n' "decentr-01" "$BASTION" "$BOOT_STATUS"

while read -r node ip is_follower; do
  [[ "$is_follower" == "0" ]] && continue
  # Find this node's result line
  result_line=$(echo "$RAW" | grep "^$node " || true)
  if [[ -z "$result_line" ]]; then
    printf '%-14s  %-18s  ✗ no output\n' "$node" "$ip"
    continue
  fi
  n=$(echo "$result_line" | awk '{print $3}')
  if [[ "$n" == "UNREACHABLE" ]]; then
    printf '%-14s  %-18s  ✗ UNREACHABLE\n' "$node" "$ip"
  elif [[ $VERIFY -eq 1 ]]; then
    [[ "$n" == "0" ]] \
      && printf '%-14s  %-18s  ✓ clean\n' "$node" "$ip" \
      || printf '%-14s  %-18s  ✗ %s procs\n' "$node" "$ip" "$n"
  else
    [[ "$n" == "0" ]] \
      && printf '%-14s  %-18s  ✓ killed\n' "$node" "$ip" \
      || printf '%-14s  %-18s  ⚠ %s procs left\n' "$node" "$ip" "$n"
  fi
done <<< "$NODE_LIST"

echo ""
echo "Done."
