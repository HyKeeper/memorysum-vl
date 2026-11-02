#!/usr/bin/env bash
# Start/stop the memory service as a plain background process (no systemd).
#
#   ./run.sh start      start uvicorn in the background (default :8003)
#   ./run.sh stop       stop it
#   ./run.sh restart    stop + start
#   ./run.sh status     pid + /healthz
#
# Environment:
#   HOST / PORT         bind address (default 0.0.0.0:8003)
#   UVICORN_OUT         uvicorn's own stderr/stdout log (default /tmp/memorysum.uvicorn.log)
#
# Application logs go to MEMORYSUM_LOG_FILE (info.log by default). You must run
# this from the repository root: Settings reads .env from the working directory.
set -uo pipefail

cd "$(dirname "$0")" || exit 1

# Bracket trick: the pattern text itself must not match, or pgrep would report the
# shell that is running this script (and "start" would think we are already up).
PATTERN='[u]vicorn memorysum_vl\.main'
UVICORN_OUT="${UVICORN_OUT:-/tmp/memorysum.uvicorn.log}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8003}"
PYTHON="${PYTHON:-.venv/bin/uvicorn}"

# Only real python/uvicorn processes: a transient shell whose command line happens
# to mention the module must not look like a running service.
pids() {
  pgrep -f "$PATTERN" 2>/dev/null | while read -r pid; do
    case "$(cat /proc/"$pid"/comm 2>/dev/null)" in
      python* | uvicorn) printf '%s\n' "$pid" ;;
    esac
  done
}

case "${1:-status}" in
  start)
    if [ -n "$(pids)" ]; then
      echo "already running: pid $(pids | tr '\n' ' ')"
      exit 0
    fi
    nohup setsid "$PYTHON" memorysum_vl.main:create_app --factory \
      --app-dir src --host "$HOST" --port "$PORT" --no-access-log \
      >> "$UVICORN_OUT" 2>&1 < /dev/null &
    sleep 2
    if [ -n "$(pids)" ]; then
      echo "started: pid $(pids | tr '\n' ' '), port $PORT, log ${MEMORYSUM_LOG_FILE:-info.log}"
    else
      echo "failed to start — see $UVICORN_OUT" >&2
      tail -20 "$UVICORN_OUT" >&2
      exit 1
    fi
    ;;
  stop)
    if [ -z "$(pids)" ]; then
      echo "not running"
      exit 0
    fi
    kill $(pids) 2>/dev/null
    for _ in 1 2 3 4 5; do
      [ -z "$(pids)" ] && break
      sleep 1
    done
    [ -n "$(pids)" ] && kill -9 $(pids) 2>/dev/null
    echo "stopped"
    ;;
  restart)
    "$0" stop
    "$0" start
    ;;
  status)
    if [ -n "$(pids)" ]; then
      echo "running: pid $(pids | tr '\n' ' ')"
    else
      echo "not running"
    fi
    curl -s -m 3 "http://127.0.0.1:$PORT/healthz" || echo "health check failed"
    echo
    ;;
  *)
    echo "usage: $0 {start|stop|restart|status}" >&2
    exit 1
    ;;
esac
