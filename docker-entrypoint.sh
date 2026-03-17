#!/usr/bin/env bash
#───────────────────────────────────────────────────────────────────────────────
#  Script : docker-entrypoint.sh
#  Purpose: Container entrypoint that allows switching between HTTP servers
#
#  Environment Variables:
#    HTTP_SERVER : Which HTTP server to use (default: gunicorn)
#                  - gunicorn : Python-based with Uvicorn workers (default)
#                  - granian  : Rust-based HTTP server (alternative)
#
#  Usage:
#    # Run with Gunicorn (default)
#    docker run -e HTTP_SERVER=gunicorn mcpgateway
#
#    # Run with Granian
#    docker run -e HTTP_SERVER=granian mcpgateway
#───────────────────────────────────────────────────────────────────────────────

set -euo pipefail

HTTP_SERVER="${HTTP_SERVER:-gunicorn}"
EXPERIMENTAL_RUST_LLM_GATEWAY_ENABLED="${EXPERIMENTAL_RUST_LLM_GATEWAY_ENABLED:-false}"

is_true() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

resolve_http_server_command() {
    case "${HTTP_SERVER}" in
        granian)
            echo "./run-granian.sh"
            ;;
        gunicorn)
            echo "./run-gunicorn.sh"
            ;;
        *)
            echo "ERROR: Unknown HTTP_SERVER value: ${HTTP_SERVER}"
            echo "Valid options: granian, gunicorn"
            exit 1
            ;;
    esac
}

HTTP_SERVER_CMD="$(resolve_http_server_command)"

if ! is_true "${EXPERIMENTAL_RUST_LLM_GATEWAY_ENABLED}"; then
    case "${HTTP_SERVER}" in
        granian)
            echo "Starting ContextForge with Granian (Rust-based HTTP server)..."
            ;;
        gunicorn)
            echo "Starting ContextForge with Gunicorn + Uvicorn..."
            ;;
    esac
    exec "${HTTP_SERVER_CMD}" "$@"
fi

LLM_GATEWAY_BIN="${LLM_GATEWAY_BIN:-./llm_gateway}"
if [[ ! -x "${LLM_GATEWAY_BIN}" ]]; then
    echo "ERROR: Experimental Rust LLM Gateway is enabled but binary not found at ${LLM_GATEWAY_BIN}"
    echo "Build the container with ENABLE_RUST=true or disable EXPERIMENTAL_RUST_LLM_GATEWAY_ENABLED"
    exit 1
fi

PORT="${PORT:-4444}"
SSL="${SSL:-false}"
export LLM_GATEWAY_BIND="${LLM_GATEWAY_BIND:-127.0.0.1:8011}"
export LLM_GATEWAY_REQUEST_TIMEOUT_SECONDS="${LLM_GATEWAY_REQUEST_TIMEOUT_SECONDS:-${EXPERIMENTAL_RUST_LLM_GATEWAY_TIMEOUT_SECONDS:-120}}"
if [[ -n "${EXPERIMENTAL_RUST_LLM_GATEWAY_INTERNAL_SECRET:-}" ]] && [[ -z "${LLM_GATEWAY_INTERNAL_SECRET:-}" ]]; then
    export LLM_GATEWAY_INTERNAL_SECRET="${EXPERIMENTAL_RUST_LLM_GATEWAY_INTERNAL_SECRET}"
fi
if [[ -z "${LLM_GATEWAY_CORE_URL:-}" ]]; then
    if is_true "${SSL}"; then
        export LLM_GATEWAY_CORE_URL="https://127.0.0.1:${PORT}"
        export LLM_GATEWAY_CORE_INSECURE_TLS="${LLM_GATEWAY_CORE_INSECURE_TLS:-true}"
    else
        export LLM_GATEWAY_CORE_URL="http://127.0.0.1:${PORT}"
    fi
fi

echo "Starting experimental Rust LLM Gateway at ${LLM_GATEWAY_BIND} (core: ${LLM_GATEWAY_CORE_URL})..."
"${LLM_GATEWAY_BIN}" &
LLM_GATEWAY_PID=$!

case "${HTTP_SERVER}" in
    granian)
        echo "Starting ContextForge with Granian (Rust-based HTTP server)..."
        ;;
    gunicorn)
        echo "Starting ContextForge with Gunicorn + Uvicorn..."
        ;;
esac
"${HTTP_SERVER_CMD}" "$@" &
APP_PID=$!

terminate_children() {
    kill -TERM "${APP_PID}" "${LLM_GATEWAY_PID}" 2>/dev/null || true
}

trap terminate_children INT TERM

set +e
wait -n "${APP_PID}" "${LLM_GATEWAY_PID}"
EXIT_STATUS=$?
set -e

terminate_children
wait "${APP_PID}" 2>/dev/null || true
wait "${LLM_GATEWAY_PID}" 2>/dev/null || true
exit "${EXIT_STATUS}"
