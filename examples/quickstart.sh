#!/usr/bin/env bash
# Copy-paste session against a running instance.
#
#   ./examples/quickstart.sh                 # text only
#   ./examples/quickstart.sh photo.jpg       # also stores and searches an image
#
# Reads the token from $MEMORYSUM_TOKEN, .secrets/api_token, or the first arg
# of MEMORYSUM_TOKEN_FILE.
set -euo pipefail

BASE="${MEMORYSUM_BASE_URL:-http://127.0.0.1:8003}"
TOKEN="${MEMORYSUM_TOKEN:-}"
IMAGE="${1:-}"

if [ -z "$TOKEN" ]; then
  token_file="${MEMORYSUM_TOKEN_FILE:-.secrets/api_token}"
  [ -f "$token_file" ] && TOKEN="$(cat "$token_file")"
fi
[ -n "$TOKEN" ] || { echo "no token: set MEMORYSUM_TOKEN or create .secrets/api_token" >&2; exit 1; }

auth=(-H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json')

echo "== healthz"
curl -s "$BASE/healthz"; echo

echo "== add (text)"
curl -s "${auth[@]}" -X POST "$BASE/v1/add" -d '{
  "request_id": "demo-req-1",
  "user_id": "demo-user",
  "session_id": "demo-session",
  "messages": [
    {"role": "user", "content": "The spare apartment key lives in the blue kitchen drawer.",
     "timestamp": 1746000000000},
    {"role": "assistant", "content": "Noted: spare key → blue drawer."}
  ]}'; echo

if [ -n "$IMAGE" ]; then
  echo "== add (image: $IMAGE)"
  b64="$(base64 -w0 "$IMAGE")"
  curl -s "${auth[@]}" -X POST "$BASE/v1/add" -d @- <<JSON
{"request_id":"demo-req-2","user_id":"demo-user","session_id":"demo-session-photo",
 "messages":[{"role":"user","content":[
   {"type":"text","text":"What is on this desk?"},
   {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,$b64"}}]}]}
JSON
  echo
fi

echo "== search (text)"
curl -s "${auth[@]}" -X POST "$BASE/v1/search" -d '{
  "user_id": "demo-user",
  "query": "where are the keys?",
  "top_k": 5}' | head -c 1200
echo

if [ -n "$IMAGE" ]; then
  echo "== search (image only)"
  b64="$(base64 -w0 "$IMAGE")"
  curl -s "${auth[@]}" -X POST "$BASE/v1/search" -d @- <<JSON | head -c 600
{"user_id":"demo-user","top_k":5,"query":[
  {"type":"text","text":"what is this?"},
  {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,$b64"}}]}
JSON
  echo
fi
