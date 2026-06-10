#!/usr/bin/env bash
# deploy_patches.sh — push patched source files from ../patches/ to every node.
#
# Files in ../patches/ override the git-cloned DistLearn package on each node
# (which is `pip install -e`, so editing the clone takes effect on next run).
#
# Usage:
#   ./deploy_patches.sh             # deploy the active SET (see below), verify md5
#   ./deploy_patches.sh --verify    # only compare md5 local vs remote, don't copy
#
# SAFE during a running experiment: overwriting .py files does NOT affect already
# running Python processes (modules are imported into memory at start). New code
# only takes effect on the NEXT start-run/join-run.

set -uo pipefail
cd "$(dirname "$0")"

KEY="${HOME}/.ssh/decentr_id_ed25519"
BASTION="84.201.157.237"
USER="decentr"
BASTION_KEY="~/.ssh/decentr_id_ed25519"
PATCH_DIR="../patches"
PKG="/opt/decentr/DistLearn/Decentr_my_own/decentr_my_own"
TIMEOUT=25

# ── Patch set: "<local_basename> <remote_path_relative_to_PKG>" ───────────────
# Active set = Phase 2 (CommunicationPolicy). Add lines to extend.
PATCHES=(
  "communication_policy.py algorithms/communication_policy.py"
  "async_gossip.py         algorithms/async_gossip.py"
  "config_models.py        config/models.py"
)

VERIFY=0
[[ "${1:-}" == "--verify" ]] && VERIFY=1

SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=$TIMEOUT -o BatchMode=yes -o LogLevel=ERROR"

# Local md5 for each patch (for verification)
declare -A LOCAL_MD5
for entry in "${PATCHES[@]}"; do
  src="${entry%% *}"
  if [[ ! -f "$PATCH_DIR/$src" ]]; then
    echo "ERROR: missing $PATCH_DIR/$src"; exit 1
  fi
  LOCAL_MD5["$src"]=$(md5 -q "$PATCH_DIR/$src" 2>/dev/null || md5sum "$PATCH_DIR/$src" | awk '{print $1}')
done

# Follower IPs from pool.yaml (skip bootstrap = first node)
NODE_LIST=$(python3 - <<'PY'
import yaml
with open("pool.yaml") as f:
    data = yaml.safe_load(f)
for n in data["nodes"]:
    is_follower = int("ssh_jump" in n)
    print(n["id"], n.get("ssh_host", n.get("host")), is_follower)
PY
)

# ── Step 1: copy patch files to bastion /tmp ─────────────────────────────────
if [[ $VERIFY -eq 0 ]]; then
  echo "Copying $((${#PATCHES[@]})) patch file(s) to bastion…"
  for entry in "${PATCHES[@]}"; do
    src="${entry%% *}"
    scp -q $SSH_OPTS "$PATCH_DIR/$src" "${USER}@${BASTION}:/tmp/patch_$src" || {
      echo "ERROR: scp $src to bastion failed"; exit 1; }
  done
fi

# ── Step 2: build a bastion-side script that installs locally (bootstrap) and
#            fans out to every follower, then reports md5 per node. ───────────
TMP_LOCAL=$(mktemp "/tmp/deploy_patches.XXXXXX.sh")
trap 'rm -f "$TMP_LOCAL"' EXIT

{
  echo '#!/bin/bash'
  echo "PKG=$PKG"
  echo "VERIFY=$VERIFY"
  echo 'BKEY=~/.ssh/decentr_id_ed25519'
  echo 'SOPTS="-i $BKEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o BatchMode=yes -o LogLevel=ERROR"'
  # Patch table
  echo 'declare -a MAP=('
  for entry in "${PATCHES[@]}"; do
    src="${entry%% *}"; dst="${entry##* }"
    echo "  \"$src $dst\""
  done
  echo ')'
  # Follower IP list
  echo 'declare -a FOLLOWERS=('
  while read -r nid ip isf; do
    [[ "$isf" == "1" ]] && echo "  \"$nid $ip\""
  done <<< "$NODE_LIST"
  echo ')'
  cat <<'BASTION_BODY'
# Install on bootstrap (this host) + report md5
for entry in "${MAP[@]}"; do
  src="${entry%% *}"; dst="${entry##* }"
  if [[ $VERIFY -eq 0 ]]; then
    mkdir -p "$(dirname "$PKG/$dst")"
    cp "/tmp/patch_$src" "$PKG/$dst"
  fi
  m=$(md5sum "$PKG/$dst" 2>/dev/null | awk '{print $1}')
  echo "decentr-01 $src ${m:-MISSING}"
done
# Fan out to followers in parallel
for fentry in "${FOLLOWERS[@]}"; do
  read -r nid ip <<< "$fentry"
  (
    for entry in "${MAP[@]}"; do
      src="${entry%% *}"; dst="${entry##* }"
      if [[ $VERIFY -eq 0 ]]; then
        scp -q $SOPTS "/tmp/patch_$src" "decentr@$ip:/tmp/patch_$src" 2>/dev/null
        ssh $SOPTS "decentr@$ip" "mkdir -p \$(dirname $PKG/$dst) && cp /tmp/patch_$src $PKG/$dst" 2>/dev/null
      fi
      m=$(ssh $SOPTS "decentr@$ip" "md5sum $PKG/$dst 2>/dev/null | awk '{print \$1}'" 2>/dev/null)
      echo "$nid $src ${m:-UNREACHABLE}"
    done
  ) &
done
wait
BASTION_BODY
} > "$TMP_LOCAL"

TMP_REMOTE="/tmp/deploy_patches_run_$$.sh"
scp -q $SSH_OPTS "$TMP_LOCAL" "${USER}@${BASTION}:${TMP_REMOTE}"
RAW=$(ssh $SSH_OPTS "${USER}@${BASTION}" "bash $TMP_REMOTE; rm -f $TMP_REMOTE" 2>&1)

# ── Step 3: verify md5 matches local for every node/file ─────────────────────
echo ""
echo "Verifying md5 (local vs each node)…"
TOTAL=0; OK=0; BAD=0
while read -r nid src remote_md5; do
  [[ -z "$nid" ]] && continue
  TOTAL=$((TOTAL+1))
  want="${LOCAL_MD5[$src]:-?}"
  if [[ "$remote_md5" == "$want" ]]; then
    OK=$((OK+1))
  else
    BAD=$((BAD+1))
    printf '  ✗ %-12s %-26s remote=%s want=%s\n' "$nid" "$src" "${remote_md5:0:8}" "${want:0:8}"
  fi
done <<< "$RAW"

echo ""
echo "Match: $OK/$TOTAL   Mismatch: $BAD"
[[ $BAD -eq 0 ]] && echo "✓ All nodes have the patched files." || echo "✗ Some nodes are out of sync — re-run."
exit $(( BAD > 0 ? 1 : 0 ))
