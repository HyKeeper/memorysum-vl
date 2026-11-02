# HTTP API

Base URL: `http://127.0.0.1:8003` — all memory endpoints live under
`MEMORYSUM_API_PREFIX` (default `/v1`).

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `POST` | `/v1/add` | Bearer | Append messages (text and/or images) to a user's memory |
| `POST` | `/v1/search` | Bearer | Retrieve the `top_k` memories most relevant to a query |
| `GET` | `/healthz` | none | Liveness probe — always unauthenticated |

Authentication: `Authorization: Bearer <token>`, where the token comes from
`MEMORYSUM_AUTH_TOKEN` or `MEMORYSUM_AUTH_TOKEN_FILE`. A missing/incorrect token
returns `401` with `WWW-Authenticate: Bearer`. Generate one with
`python -c "import secrets; print(secrets.token_urlsafe(24))"`.

---

## POST /v1/add

```json
{
  "request_id": "req-0001",
  "user_id": "u_42",
  "session_id": "s_2025_05_01",
  "messages": [
    {"role": "user", "content": "Here is the desk this afternoon.",
     "timestamp": 1746000000000},
    {"role": "user", "content": [
      {"type": "text", "text": "Note the monitor stand."},
      {"type": "image_url",
       "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSk..."}}
    ]},
    {"role": "assistant", "content": "Got it — comparing with the earlier photo."}
  ]
}
```

| Field | Type | Notes |
|---|---|---|
| `request_id` | string, non-empty | **Idempotency key.** Re-sending the same `request_id` replaces the previous write for that request |
| `user_id` | string, non-empty | Tenant / memory owner; also the vector partition key |
| `session_id` | string, non-empty | Conversation grouping; rendered as `Session N (date)` in results |
| `messages[].role` | `user` \| `assistant` | |
| `messages[].content` | string **or** list of parts | parts are `{"type":"text"}` or `{"type":"image_url"}` |
| `messages[].timestamp` | int (ms epoch), optional | Used for ordering and the rendered date |

Images must be **inline data URIs** (`image/jpeg`, `image/png`, `image/webp`),
≤ 10 MB each and ≤ 30 MB per request. URLs are rejected — nothing is fetched
from the network, by design.

Response `200`:

```json
{"success": true, "request_id": "req-0001", "user_id": "u_42", "session_id": "s_2025_05_01"}
```

`success: true` means **durably written and immediately searchable** — captions
and vectors are computed synchronously. If a model server was down, the write
still succeeds and the affected units are flagged (`caption_state` /
`embedding_state` = `missing`) rather than failing the request.

---

## POST /v1/search

```json
{
  "user_id": "u_42",
  "query": "Where did I leave the tweezers in the centrifuge station?",
  "top_k": 20,
  "options": null
}
```

| Field | Type | Notes |
|---|---|---|
| `user_id` | string, non-empty | Scope of the search |
| `query` | string **or** list of parts | May contain images — for "what is this?" style questions |
| `top_k` | int, `1..100` | Number of memories returned |
| `options` | list of strings, optional | Multiple-choice options; each option becomes an extra lexical query |

Response `200`:

```json
{
  "data": [
    {
      "id": "mem_61d4a7559c3ab7ac5154",
      "content": [
        {"type": "text", "text": "Session 12 (2025-08-29)\nuser: Here is a close up of the right side ..."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ],
      "score": 0.9230700512476664,
      "created_at": "2025-08-29T00:00:00Z"
    }
  ]
}
```

* `content` is rendered for a downstream answer model: role label, `Session N
  (date)` prefix, the message text, and the image description. Adjacent messages
  of the same session may be appended (`MEMORYSUM_NEIGHBOR_EXPAND`).
* `score` is a **sigmoid of the reranker logit** — a calibrated confidence, not a
  normalised rank. A top hit at `0.02` means "nothing really matched", which is
  exactly the signal a caller needs to abstain. With reranking disabled the score
  is the normalised fused score.
* Original images are returned only when `MEMORYSUM_RETURN_IMAGES=true`; they are
  inlined as data URIs (re-encoded above `MEMORYSUM_RESPONSE_IMAGE_PASSTHROUGH_BYTES`),
  capped by `MEMORYSUM_MAX_RETURNED_IMAGES`. A 100-hit multimodal response can be
  tens of MiB — most callers that only need text should turn this off.
* `data: []` is a legitimate answer: the abstention path returns an empty list
  when the question's content words appear nowhere in the user's corpus.

---

## GET /healthz

```json
{"status": "ok", "track": "multimodal"}
```

Never requires authentication, never touches the models — safe for load balancers
and container probes.

---

## Errors

Every failure uses one envelope, and `code` is stable enough to branch on:

```json
{"error": {"code": "unauthorized", "message": "missing or invalid bearer token"}}
```

| Status | `code` | When |
|---|---|---|
| `401` | `unauthorized` | Missing/invalid bearer token |
| `422` | `validation_error` | Body failed validation (empty `user_id`, `top_k` out of range, non-inline image, oversize images…) |
| `415` | `unsupported_content` | Content type the track does not accept |
| `501` | `unsupported_track` | Endpoint not implemented for the configured `MEMORYSUM_TRACK` |
| `500` | `internal_error` | Unexpected failure — details are in the log, not in the response |

Every response carries `x-request-id` (echoed if you send one), which is the same
id printed in every log line of that request — grep the log by it to see the full
retrieval trace.

---

## Examples

`examples/quickstart.sh` (curl) and `examples/client.py` (Python, stdlib only)
contain runnable versions of both calls, including a base64 image.

```bash
export MEMORYSUM_TOKEN=$(cat .secrets/api_token)

curl -s -X POST http://127.0.0.1:8003/v1/add \
  -H "Authorization: Bearer $MEMORYSUM_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"req-1","user_id":"u_42","session_id":"s_1",
       "messages":[{"role":"user","content":"the spare key is in the blue drawer"}]}'

curl -s -X POST http://127.0.0.1:8003/v1/search \
  -H "Authorization: Bearer $MEMORYSUM_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"u_42","query":"where are the keys?","top_k":5}'
```
