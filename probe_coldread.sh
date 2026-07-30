#!/usr/bin/env bash
# probe_coldread.sh — prove the v4 cold store can RECONSTRUCT a KV cache from disk.
#
# RUN THIS FIRST THING AFTER A SERVER RESTART, before anything else touches the
# endpoint. The whole test rests on the in-memory tier being EMPTY: every entry
# in the cold store got there via memory, so while a process lives, memory always
# wins the longest-prefix match and the payload is never read. Verified
# 2026-07-30 — a fresh session id does NOT force a cold read, because matching is
# longest-prefix across the store and is not partitioned by session key.
#
# A restart is therefore not a convenience here. It is the only condition under
# which this path is reachable, and it is also the real use case: a Kindled
# resuming after the server has been bounced.
#
# The payload is the exact prompt persisted at 18:54 on 2026-07-30 (10,386
# tokens, 953 MB of blocks on the T7 store).
set -euo pipefail
PORT="${1:-8890}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SID="coldread-$(date +%H%M%S)"

echo "session: $SID   port: $PORT"
START=$(python3 -c 'import time;print(time.time())')
curl -s --max-time 300 -X POST "http://localhost:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" -H "X-Session-Id: ${SID}" \
  --data-binary "@${HERE}/probe_coldread_payload.json" > /tmp/coldread_resp.json
python3 - "$START" <<'PY'
import json, sys, time
d = json.load(open('/tmp/coldread_resp.json'))
u = d.get('usage', {})
cached = u.get('prompt_tokens_details', {}).get('cached_tokens')
total  = u.get('prompt_tokens')
secs   = time.time() - float(sys.argv[1])
print(f"  cached_tokens : {cached} of {total}")
print(f"  wall seconds  : {secs:.1f}")
print()
if cached and total and cached > total * 0.9:
    print("  PASS on the client side — but this alone does NOT prove a cold read.")
    print("  Confirm in the log that the SSD tier actually loaded (see below).")
else:
    print("  NO ADOPTION. Either the cold store did not find the manifest, or")
    print("  the payload read failed. The log lines below say which.")
PY
cat <<'MSG'

Now confirm the TIER in the server log — the client cannot distinguish them:

  L=$(ls -t ~/mlxcel_logs/*.log | head -1)
  sed 's/\x1b\[[0-9;]*m//g' "$L" | grep -aE "lookup attempt|MATCH|LOADED|NOT LOADED" | tail -5

WANT (cold read happened):
  prompt-cache: ... store_entries=0            <- memory empty, the precondition
  prompt-cache: SSD cold-store LOADED ...      <- the payload read

DO NOT ACCEPT (test was void, not passed):
  in-memory longest-prefix MATCH ...           <- something warmed memory first
  SSD cold-store NOT LOADED ... none beat the in-memory match

A high cached_tokens with an in-memory MATCH means the run proved nothing about
the cold store. Restart and run this before anything else touches the endpoint.
MSG
