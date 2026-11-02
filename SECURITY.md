# Security

## Reporting a vulnerability

Open a private security advisory on GitHub
(<https://github.com/HyKeeper/memorysum-vl/security/advisories/new>) rather than a
public issue. Please include the version/commit, a minimal reproduction, and what
you expected versus what happened.

## Threat model

`memorysum-vl` is designed to run **on your own machine or inside your own
network**, next to your own model servers. It is not a hardened public service.

* **Authentication is a single shared bearer token.** There is no per-user
  identity: any caller holding the token can read any `user_id`'s memory by asking
  for that `user_id`. Do not hand the token to tenants that should not see each
  other's data.
* **The token is the only secret.** It comes from `MEMORYSUM_AUTH_TOKEN_FILE`
  (preferred, `chmod 600`) or `MEMORYSUM_AUTH_TOKEN`. Comparison is
  constant-time (`hmac.compare_digest`), and the token is read once at startup —
  rotate it by restarting the service.
* **No outbound network calls except the model servers.** Images must be inline
  data URIs; the service never fetches a URL you pass in. That removes SSRF as a
  class, and means a compromised payload cannot make the service phone home.
* **Uploaded content is untrusted.** Images are re-encoded/limit-checked before
  they reach the models (`core/imaging.py`, `core/media.py`), and text is
  sanitised before it lands in a log line. Prompt injection inside a stored
  screenshot is still possible: the captioner will happily *describe* a picture
  that says "ignore previous instructions". The service does not interpret model
  output as instructions, but a downstream answer model might.
* **Logs contain user data.** `info.log` records request and response bodies
  (truncated). Set `MEMORYSUM_LOG_REQUEST_BODY_LIMIT` /
  `MEMORYSUM_LOG_RESPONSE_BODY_LIMIT` to small values, or `MEMORYSUM_LOG_FILE=` to
  keep logs on stderr only, if that is not acceptable for your data.
* **Images are returned inline by default** (`MEMORYSUM_RETURN_IMAGES=true`), so
  a caller with the token gets the original bytes back. Turn it off if your
  corpus contains sensitive screenshots.

## Hardening checklist

- [ ] `MEMORYSUM_AUTH_TOKEN_FILE` set, file mode `0600`, unique per deployment
- [ ] bind to `127.0.0.1` and put a TLS-terminating proxy in front
- [ ] `MEMORYSUM_LOG_*_BODY_LIMIT` small (e.g. `2048`), log rotation in place
- [ ] `MEMORYSUM_RETURN_IMAGES=false` if the corpus is sensitive
- [ ] model servers bound to loopback only — they have **no authentication** of
      their own and will happily describe/embed anything asked of them
- [ ] back up `data/` with `sqlite3 ... ".backup"`, and protect it like a database
      of screenshots (because it is one)
