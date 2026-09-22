#!/usr/bin/env bash
set -euo pipefail

# Qwen3.8-27B with DFlash 2 on SGLang (DGX Spark / GB10, aarch64)
# Default launcher for this recipe.
#
# Prereqs:
#   1. Ensure .env is present (use .env.sample as template).
#   2. NVIDIA DGX Spark (GB10 / SM121) with NVIDIA Container Toolkit.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load optional .env overrides
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  while IFS='=' read -r key value || [[ -n "${key}" ]]; do
    key="${key%$'\r'}"; value="${value%$'\r'}"
    key="${key#"${key%%[![:space:]]*}"}"; key="${key%"${key##*[![:space:]]}"}"
    [[ -z "${key}" || "${key}" == \#* ]] && continue
    # Strip surrounding single or double quotes
    value="${value#\"}" ; value="${value%\"}"
    value="${value#\'}" ; value="${value%\'}"
    if [[ -z "${!key:-}" ]]; then
      export "${key}=${value}"
    fi
  done < "${SCRIPT_DIR}/.env"
fi

DFLASH_IMAGE="${DFLASH_IMAGE:-lmsysorg/sglang@sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe}"
TARGET_MODEL="${TARGET_MODEL:-0xWhiteMage/Qwen3.8-27B-Kearuga}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b-sglang}"

# Drafter profile: kearuga (default; Kearuga-distilled NVFP4 drafter, K=12,
# vendored 64K draft-head overlay + token map, hash-verified at launch) or
# stock (the v0.5.0 recipe: z-lab BF16 drafter, K=10, no overlay).
DRAFTER_PROFILE="${DRAFTER_PROFILE:-kearuga}"
case "${DRAFTER_PROFILE}" in
  kearuga)
    _DEF_DFLASH_MODEL="0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2"
    _DEF_DFLASH_REV="4a109695465b5e85af452ed2db4866eb367402d7"
    _DEF_DRAFT_TOKENS="12"
    _DEF_DRAFT_QUANT="modelopt_fp4"
    _DEF_OVERLAY="1"
    _DEF_DRAFT_KV_DTYPE="fp8_e4m3"
    ;;
  stock)
    _DEF_DFLASH_MODEL="z-lab/Qwen3.8-27B-DFlash2"
    _DEF_DFLASH_REV="50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
    _DEF_DRAFT_TOKENS="10"
    _DEF_DRAFT_QUANT=""
    _DEF_OVERLAY="0"
    _DEF_DRAFT_KV_DTYPE=""
    ;;
  *)
    echo "DRAFTER_PROFILE must be 'kearuga' or 'stock', got '${DRAFTER_PROFILE}'"
    exit 1
    ;;
esac

# DFLASH_REV / DFLASH_DRAFT_QUANT use ${VAR-default} (not :-) so an explicit
# empty value disables them (needed when DFLASH_MODEL is a local path).
DFLASH_MODEL="${DFLASH_MODEL:-${_DEF_DFLASH_MODEL}}"
DFLASH_REV="${DFLASH_REV-${_DEF_DFLASH_REV}}"
DFLASH_DRAFT_QUANT="${DFLASH_DRAFT_QUANT-${_DEF_DRAFT_QUANT}}"
DFLASH_OVERLAY="${DFLASH_OVERLAY:-${_DEF_OVERLAY}}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-8192}"
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-4}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-1048576}"
DFLASH_DRAFT_TOKENS="${DFLASH_DRAFT_TOKENS:-${_DEF_DRAFT_TOKENS}}"
DFLASH_DRAFT_WINDOW_SIZE="${DFLASH_DRAFT_WINDOW_SIZE:-2048}"
# Drafter KV dtype: fp8_e4m3 on the kearuga profile (v0.6.6: +13.9 % shared KV pool for ~-1 % decode);
# ${VAR-default} so an explicit empty value keeps the BF16 drafter KV (the stock/v0.5.0 recipe).
DFLASH_DRAFT_KV_DTYPE="${DFLASH_DRAFT_KV_DTYPE-${_DEF_DRAFT_KV_DTYPE}}"
DFLASH_TOKEN_MAP="${DFLASH_TOKEN_MAP:-${SCRIPT_DIR}/drafter/draft-vocab-v4-top65650.pt}"
OVERLAY_DIR="${OVERLAY_DIR:-${SCRIPT_DIR}/drafter/sglang-overlay}"
SGLANG_PKG_DIR="${SGLANG_PKG_DIR:-/sgl-workspace/sglang/python/sglang}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bf16}"
CPUSET="${CPUSET:-${CPU_AFFINITY:-}}"
HEALTH_TIMEOUT_SECS="${HEALTH_TIMEOUT_SECS:-900}"

PRIORITY_SCHEDULING="${PRIORITY_SCHEDULING:-1}"
DEFAULT_PRIORITY_VALUE="${DEFAULT_PRIORITY_VALUE:-0}"
PRIORITY_PREEMPTION_THRESHOLD="${PRIORITY_PREEMPTION_THRESHOLD:-10}"

if (( CONTEXT_LENGTH != 262144 )); then
  echo "CONTEXT_LENGTH '${CONTEXT_LENGTH}' unsupported by this qualified profile (use 262144)"
  exit 1
fi

MAMBA_SLOTS_PER_REQ=5
MAMBA_CACHE_SIZE=$(( MAX_CONCURRENT_REQUESTS * MAMBA_SLOTS_PER_REQ ))

if (( MAX_TOTAL_TOKENS < CONTEXT_LENGTH )); then
  echo "MAX_TOTAL_TOKENS (${MAX_TOTAL_TOKENS}) must be >= CONTEXT_LENGTH (${CONTEXT_LENGTH})"
  exit 1
fi

case "${PRIORITY_SCHEDULING}" in
  0|1) : ;;
  *) echo "PRIORITY_SCHEDULING must be 0 or 1, got '${PRIORITY_SCHEDULING}'"; exit 1 ;;
esac

CONTAINER_NAME="qwen3.8-27b-sglang"
PID_FILE=".sglang.pid"
LOG_FILE=".sglang.log"
WORK_DIR="$(pwd)"
HF_HOME="${WORK_DIR}/.cache/huggingface"
TRITON_CACHE_DIR="${WORK_DIR}/.cache/triton"
READY_URL="http://127.0.0.1:${PORT}/v1/models"
HEALTH_URL="http://127.0.0.1:${PORT}/health"

command -v docker >/dev/null 2>&1 || { echo "docker is not on PATH"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is not on PATH"; exit 1; }

mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}"

export HF_TOKEN="${HF_TOKEN:-}"

# Overlay pre-flight: the vendored files and the image's base files must both
# match their pinned hashes before anything is mounted.
OVERLAY_MOUNT_ARGS=()
TOKEN_MAP_ARGS=()
if [[ "${DFLASH_OVERLAY}" == "1" ]]; then
  command -v sha256sum >/dev/null 2>&1 || { echo "DFLASH_OVERLAY=1 requires sha256sum"; exit 1; }
  if ! (cd "${SCRIPT_DIR}/drafter" && sha256sum -c --quiet SHA256SUMS); then
    echo "drafter/SHA256SUMS verification failed; refusing to boot with an unverified overlay"
    exit 1
  fi
  if [[ "${OVERLAY_SKIP_BASE_CHECK:-0}" != "1" ]]; then
    if ! docker run --rm --entrypoint sha256sum \
        -v "${SCRIPT_DIR}/drafter/BASE-SHA256SUMS:/tmp/kearuga-base.sums:ro" \
        "${DFLASH_IMAGE}" -c --quiet /tmp/kearuga-base.sums; then
      echo "image does not match the overlay's base hashes; use DRAFTER_PROFILE=stock, or apply drafter/sglang-overlay/PATCH.diff to your build and set OVERLAY_SKIP_BASE_CHECK=1"
      exit 1
    fi
  fi
  for rel in \
    sglang/srt/models/dflash.py \
    sglang/srt/speculative/dflash_worker_v2.py \
    sglang/srt/speculative/dflash_utils.py \
    sglang/srt/speculative/dflash_head_utils.py \
    sglang/srt/server_args.py; do
    OVERLAY_MOUNT_ARGS+=(-v "${OVERLAY_DIR}/${rel}:${SGLANG_PKG_DIR}/${rel#sglang/}:ro")
  done
  if [[ ! -f "${DFLASH_TOKEN_MAP}" || "${DFLASH_TOKEN_MAP}" != /* ]]; then
    echo "DFLASH_TOKEN_MAP must be an existing file given as an absolute path, got '${DFLASH_TOKEN_MAP}'"
    exit 1
  fi
  TOKEN_MAP_ARGS=(-v "${DFLASH_TOKEN_MAP}:/drafter/$(basename "${DFLASH_TOKEN_MAP}"):ro")
fi

echo "Starting ${TARGET_MODEL} with DFlash 2 profile '${DRAFTER_PROFILE}': ${DFLASH_MODEL} @ ${DFLASH_REV:0:8}"
echo "Per-request context: ${CONTEXT_LENGTH} tokens (native; YaRN off)"
echo "Max concurrent requests: ${MAX_CONCURRENT_REQUESTS} (mamba pool ${MAMBA_CACHE_SIZE} slots)"
echo "KV token pool cap: ${MAX_TOTAL_TOKENS}; DFlash draft window: ${DFLASH_DRAFT_WINDOW_SIZE}; draft KV dtype: ${DFLASH_DRAFT_KV_DTYPE:-bf16 (default)}"
echo "Priority scheduling: ${PRIORITY_SCHEDULING} (default=${DEFAULT_PRIORITY_VALUE}, preemption threshold=${PRIORITY_PREEMPTION_THRESHOLD})"
echo "Spec decode: DFLASH draft=${DFLASH_DRAFT_TOKENS} rev=${DFLASH_REV:0:8} quant='${DFLASH_DRAFT_QUANT}'"
if [[ "${DFLASH_OVERLAY}" == "1" ]]; then
  echo "Draft-head overlay: ON (map: $(basename "${DFLASH_TOKEN_MAP}"); vendored files hash-verified)"
else
  echo "Draft-head overlay: OFF"
fi
echo "Mem fraction: ${MEM_FRACTION}"
echo "Image: ${DFLASH_IMAGE}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"

PIN_ARGS=()
[[ -n "${CPUSET}" ]] && PIN_ARGS=(--cpuset-cpus "${CPUSET}")
PREFILL_GRAPH_ARGS=(--disable-prefill-cuda-graph --disable-flashinfer-autotune)
# Local model dirs need --revision local so SGLang does not treat the path as
# a hub revision.
REVISION_ARGS=()
[[ "${TARGET_MODEL}" == /* ]] && REVISION_ARGS=(--revision local)
PRIORITY_ARGS=()
if [[ "${PRIORITY_SCHEDULING}" == "1" ]]; then
  PRIORITY_ARGS=(
    --enable-priority-scheduling
    --default-priority-value "${DEFAULT_PRIORITY_VALUE}"
    --priority-scheduling-preemption-threshold "${PRIORITY_PREEMPTION_THRESHOLD}"
  )
fi

# Auto bind-mount local host model paths if specified as absolute paths
MODEL_MOUNT_ARGS=()
if [[ "${TARGET_MODEL}" == /* ]]; then
  MODEL_MOUNT_ARGS+=(-v "${TARGET_MODEL}:${TARGET_MODEL}")
fi
if [[ "${DFLASH_MODEL}" == /* ]]; then
  MODEL_MOUNT_ARGS+=(-v "${DFLASH_MODEL}:${DFLASH_MODEL}")
fi

# Conditional drafter revision / quantization / token-map flags
DRAFT_REV_ARGS=()
[[ -n "${DFLASH_REV:-}" ]] && DRAFT_REV_ARGS=(--speculative-draft-model-revision "${DFLASH_REV}")
DRAFT_QUANT_ARGS=()
[[ -n "${DFLASH_DRAFT_QUANT}" ]] && DRAFT_QUANT_ARGS=(--speculative-draft-model-quantization "${DFLASH_DRAFT_QUANT}")
DRAFT_KV_ARGS=()
[[ -n "${DFLASH_DRAFT_KV_DTYPE}" ]] && DRAFT_KV_ARGS=(--speculative-draft-kv-cache-dtype "${DFLASH_DRAFT_KV_DTYPE}")
TOKEN_MAP_FLAG=()
if [[ "${DFLASH_OVERLAY}" == "1" ]]; then
  TOKEN_MAP_FLAG=(--speculative-dflash-token-map "/drafter/$(basename "${DFLASH_TOKEN_MAP}")")
fi

DOCKER_CMD=(
  docker run -d
  --name "${CONTAINER_NAME}"
  --network host
  --ipc host
  --privileged
  --cap-add IPC_LOCK
  --ulimit memlock=-1:-1
  --ulimit stack=67108864
  --gpus all
  --shm-size 32g
  "${PIN_ARGS[@]}"
  "${MODEL_MOUNT_ARGS[@]}"
  "${OVERLAY_MOUNT_ARGS[@]}"
  "${TOKEN_MAP_ARGS[@]}"
  -e HF_HOME=/root/.cache/huggingface
  -e TRITON_CACHE_DIR=/root/.triton
  -e HF_TOKEN="${HF_TOKEN:-}"
  -e PYTHONUNBUFFERED=1
  -v "${HF_HOME}:/root/.cache/huggingface"
  -v "${TRITON_CACHE_DIR}:/root/.triton"
  "${DFLASH_IMAGE}"
  python3 -m sglang.launch_server
  --model-path "${TARGET_MODEL}"
  "${REVISION_ARGS[@]}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --mem-fraction-static "${MEM_FRACTION}"
  --attention-backend flashinfer
  --chunked-prefill-size "${CHUNKED_PREFILL}"
  --max-prefill-tokens "${CHUNKED_PREFILL}"
  "${PREFILL_GRAPH_ARGS[@]}"
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
  --mamba-ssm-dtype bfloat16
  --mamba-full-memory-ratio 4.21
  --mamba-radix-cache-strategy extra_buffer
  --max-mamba-cache-size "${MAMBA_CACHE_SIZE}"
  --max-running-requests "${MAX_CONCURRENT_REQUESTS}"
  --max-total-tokens "${MAX_TOTAL_TOKENS}"
  --context-length "${CONTEXT_LENGTH}"
  --speculative-algorithm DFLASH
  --speculative-draft-model-path "${DFLASH_MODEL}"
  "${DRAFT_REV_ARGS[@]}"
  "${DRAFT_QUANT_ARGS[@]}"
  --speculative-num-draft-tokens "${DFLASH_DRAFT_TOKENS}"
  --speculative-draft-window-size "${DFLASH_DRAFT_WINDOW_SIZE}"
  "${DRAFT_KV_ARGS[@]}"
  "${TOKEN_MAP_FLAG[@]}"
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --sampling-defaults model
  --enable-metrics
  --enable-cache-report
  --cuda-graph-max-bs-decode 4
  --sleep-on-idle
  "${PRIORITY_ARGS[@]}"
  --host "${HOST}"
  --port "${PORT}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN: all pre-flight checks passed; would run:"
  printf '%q ' "${DOCKER_CMD[@]}"
  printf '\n'
  exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching SGLang container (DFlash 2)
EOF

"${DOCKER_CMD[@]}" >/dev/null

container_id="$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")"
echo "${container_id}" > "${PID_FILE}"
echo "Spawned container ${CONTAINER_NAME} (${container_id})"

log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]] && kill -0 "${log_follow_pid}" 2>/dev/null; then
    kill "${log_follow_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

docker logs -f "${CONTAINER_NAME}" 2>&1 | tee -a "${LOG_FILE}" | grep --line-buffered -v "Enabled fused SiLU+mul+FP4-quant for dense MLP down_proj input" &
log_follow_pid=$!

echo "Waiting for HTTP readiness at ${READY_URL}"
start_time="$(date +%s)"
max_wait="${HEALTH_TIMEOUT_SECS}"

while true; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo
    echo "ERROR: container exited unexpectedly during startup."
    echo "Review ${LOG_FILE} for details."
    exit 1
  fi

  if curl -s -f -o /dev/null "${READY_URL}"; then
    if curl -s -f -o /dev/null "${HEALTH_URL}"; then
      echo
      echo "Qwen3.8-27B with DFlash 2 is ready at http://${HOST}:${PORT}"
      exit 0
    fi
  fi

  now="$(date +%s)"
  elapsed=$(( now - start_time ))
  if (( elapsed > max_wait )); then
    echo
    echo "ERROR: server failed to become ready within ${max_wait}s."
    echo "Review ${LOG_FILE} for details."
    exit 1
  fi

  sleep 2
done
