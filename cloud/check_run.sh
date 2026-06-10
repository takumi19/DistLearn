#!/usr/bin/env bash
# Usage: ./check_run.sh [run_id]
# If run_id omitted — picks latest sync_static run automatically.

KEY=~/.ssh/decentr_id_ed25519
cd "$(dirname "$0")"
BASTION=$(python3 -c "import yaml; d=yaml.safe_load(open('pool.yaml')); print(d['nodes'][0]['ssh_host'])")
BOOTSTRAP_IP=$(python3 -c "import yaml; d=yaml.safe_load(open('pool.yaml')); print(d['nodes'][0]['host'])")

ssh_b() { ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
  -o ProxyCommand="ssh -i $KEY -W %h:%p -o StrictHostKeyChecking=no decentr@$BASTION" \
  "decentr@$1" "${@:2}" 2>/dev/null; }

# ── Find run ID ────────────────────────────────────────────────────────────────
if [ -n "$1" ]; then
  RUN_ID="$1"
else
  # Pick the latest run of ANY experiment (sync OR async), not just sync_static.
  RUN_ID=$(ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "decentr@$BASTION" \
    "ls -t /tmp/dc_boot_*.log 2>/dev/null | head -1 | \
     sed 's|/tmp/dc_boot_||;s|\.log||'" 2>/dev/null)
fi
[ -z "$RUN_ID" ] && echo "No run found." && exit 1
echo "Run: $RUN_ID"
echo "$(date)"
echo "────────────────────────────────────────────"

# ── Bootstrap ─────────────────────────────────────────────────────────────────
BOOT=$(ssh_b $BOOTSTRAP_IP \
  "ps aux | grep python | grep start-run | grep -v grep | awk '{print \$3\"% cpu  \"\$4\"% mem  elapsed=\"\$10}'")
[ -n "$BOOT" ] && echo "Bootstrap:  $BOOT" || echo "Bootstrap:  DEAD ✗"

# ── Coordination ──────────────────────────────────────────────────────────────
# Only the sync algorithm runs a central participant barrier on :50052.
# Async gossip has no central barrier (port 50052 serves config, not a
# participant list), so hitting /sync/participants there just 404s.
case "$RUN_ID" in
  *__sync_*) MODE=sync ;;
  *)         MODE=async ;;
esac

if [ "$MODE" = sync ]; then
  COORD=$(ssh_b $BOOTSTRAP_IP \
    "curl -s -o /tmp/_cp.json -w '%{http_code}' http://localhost:50052/sync/participants 2>/dev/null")
  if [ "$COORD" = "200" ]; then
    COUNT=$(ssh_b $BOOTSTRAP_IP "python3 -c \"import json; d=json.load(open('/tmp/_cp.json')); print(len(d['participants']))\"" 2>/dev/null)
    echo "Coordination: ✓ decided — $COUNT participants"
  elif [ "$COORD" = "202" ]; then
    echo "Coordination: ⏳ collecting registrations (202)"
  else
    echo "Coordination: ? (http=$COORD)"
  fi
else
  echo "Coordination: — async gossip (no central barrier)"
fi

# ── Followers ─────────────────────────────────────────────────────────────────
FOLLOWER_IPS=$(python3 -c "
import yaml
with open('$(dirname "$0")/pool.yaml') as f:
    pool = yaml.safe_load(f)
print(' '.join(n['ssh_host'] for n in pool['nodes'][1:]))")

COUNTS=$(ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=15 "decentr@$BASTION" "
KEY=/home/decentr/.ssh/decentr_id_ed25519
r=0; d=0
for IP in $FOLLOWER_IPS; do
  (res=\$(ssh -i \$KEY -o StrictHostKeyChecking=no -o ConnectTimeout=4 -o BatchMode=yes decentr@\$IP \
    'pgrep -f python.*join-run > /dev/null && echo R || echo D' 2>/dev/null < /dev/null)
   echo \${res:-D}) &
done | { while read s; do [ \"\$s\" = R ] && r=\$((r+1)) || d=\$((d+1)); done; echo \"\$r \$d\"; }
" 2>/dev/null)
RUNNING=$(echo $COUNTS | awk '{print $1}')
DEAD=$(echo $COUNTS | awk '{print $2}')
TOTAL=$(echo "$FOLLOWER_IPS" | wc -w)
[ "$DEAD" = "0" ] \
  && echo "Followers:  ✓ $RUNNING/$TOTAL running" \
  || echo "Followers:  ✗ $RUNNING/$TOTAL running  ($DEAD dead)"

# ── Training artifacts ────────────────────────────────────────────────────────
CKPT_INFO=$(ssh_b $BOOTSTRAP_IP "
  CKPT_DIR=/opt/decentr/DistLearn/Decentr_my_own/artifacts/checkpoints/${RUN_ID}/decentr-01
  FILES=\$(find \$CKPT_DIR -name '*.pt' 2>/dev/null | sort)
  COUNT=\$(echo \"\$FILES\" | grep -c '.pt' 2>/dev/null || echo 0)
  LAST=\$(echo \"\$FILES\" | tail -1)
  FIRST=\$(echo \"\$FILES\" | head -1)
  [ -z \"\$LAST\" ] && exit 0
  LAST_NUM=\$(basename \$LAST | grep -oE '[0-9]+')
  FIRST_MTIME=\$(stat -c %Y \$FIRST 2>/dev/null)
  LAST_MTIME=\$(stat -c %Y \$LAST 2>/dev/null)
  echo \"\$COUNT \$LAST_NUM \$FIRST_MTIME \$LAST_MTIME\"
" 2>/dev/null)

if [ -n "$CKPT_INFO" ]; then
  COUNT=$(echo $CKPT_INFO | awk '{print $1}')
  LAST_NUM=$(echo $CKPT_INFO | awk '{print $2}')
  FIRST_MTIME=$(echo $CKPT_INFO | awk '{print $3}')
  LAST_MTIME=$(echo $CKPT_INFO | awk '{print $4}')
  TOTAL_EPOCHS=$(grep -A5 'sync_static' "$(dirname "$0")/suites/cifar10_wan_campaign.yaml" \
    | grep 'epochs:' | head -1 | awk '{print $2}')
  TOTAL_EPOCHS=${TOTAL_EPOCHS:-50}

  # Time per round and ETA
  ETA_STR=""
  if [ -n "$FIRST_MTIME" ] && [ -n "$LAST_MTIME" ] && [ "$COUNT" -gt 1 ]; then
    ELAPSED=$(( LAST_MTIME - FIRST_MTIME ))
    SEC_PER_ROUND=$(( ELAPSED / (COUNT - 1) ))
    ROUNDS_LEFT=$(( TOTAL_EPOCHS - LAST_NUM ))
    ETA_SEC=$(( ROUNDS_LEFT * SEC_PER_ROUND ))
    ETA_MIN=$(( ETA_SEC / 60 ))
    ETA_STR="  ~${SEC_PER_ROUND}s/round  ETA ${ETA_MIN}min"
  fi

  # Match both sync_run_summary.json and async_run_summary.json.
  DONE_NODES=$(ssh_b $BOOTSTRAP_IP \
    "find /opt/decentr/DistLearn/Decentr_my_own/artifacts/logs/${RUN_ID}/ -name '*run_summary.json' 2>/dev/null | wc -l")
  echo "Training:   ✓ epoch ${LAST_NUM}/${TOTAL_EPOCHS}${ETA_STR}  ($DONE_NODES nodes finished)"
else
  echo "Training:   ⏳ not started yet (no checkpoints)"
fi

# ── Errors in follower logs ───────────────────────────────────────────────────
ERRORS=$(ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=15 "decentr@$BASTION" "
KEY=/home/decentr/.ssh/decentr_id_ed25519
for IP in $FOLLOWER_IPS; do
  ssh -i \$KEY -o StrictHostKeyChecking=no -o ConnectTimeout=4 -o BatchMode=yes decentr@\$IP \
    \"grep -l 'TimeoutError\|Traceback\|Error' /tmp/dc_fol_${RUN_ID}_*.log 2>/dev/null\" < /dev/null 2>/dev/null &
done; wait" 2>/dev/null)
[ -n "$ERRORS" ] \
  && echo "Errors:     ✗ found in: $(echo "$ERRORS" | wc -l) node log(s)" \
  || echo "Errors:     ✓ none"
echo "────────────────────────────────────────────"
