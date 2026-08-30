#!/bin/bash
# MXFP4-W4A8 target + FP8 DFlash2 drafter on vLLM 0.28 — dual R9700, dockerized.
#
# This is ggz14's radiance MXFP4+dflash stack (codeberg.org/ggz14/radiance-vllm-mxfp4,
# written for stilldeadcode/vllm-radiance:0.9.3 / vLLM 0.27.1 / podman) ported to the
# vLLM 0.28 image this guide already uses, under docker, with the P2P RCCL mounts.
#
# RUN IT FROM A CLONE OF THE ggz14 REPO (it is the patch/module source) at the pinned
# commit this launcher's patch battery matches — see the README's rebuild table:
#   git clone https://codeberg.org/ggz14/radiance-vllm-mxfp4 && cd radiance-vllm-mxfp4
#   git checkout 2d72e78
#   cp <this-guide>/mxfp4-dflash/run_mxfp4_dflash.sh \
#      <this-guide>/mxfp4-dflash/patch_transformers_docstring_lint.py .
#   MODELS=$HOME/models ./run_mxfp4_dflash.sh
#
# Models needed under $MODELS (defaults $HOME/models):
#   - target : Qwen3.8-27B-MXFP4-mtpfp8 — built once from amd/Qwen3.8-27B-Quark-AWQ-MXFP4
#              with the ggz14 repo's fp8_mtp.py (AMD's release does not load as-is; its
#              bf16 mtp.* layers fall through to the mxfp4 scheme and assert)
#   - drafter: tcclaviger/Qwen3.8-27B-DFlash2-FP8 (hf download …)
#
# Port findings (why this works on 0.28.0 — all verified 2026-08-29):
#   - patch_quark_mxfp4 / patch_ar_maxbytes / patch_topk_triton_rows: already in the image (NOOP)
#   - patch_dflash_calib, patch_rmsquant_fusion, patch_verify_head, patch_kv_group_size,
#     patch_gdn_merge_inproj, patch_qwen3_thinkoff: apply cleanly
#   - patch_dflash_mxfp4_kv: SKIPPED — upstreamed in 0.28's native qwen3_dflash.py
#     (_DFLASH_DENSE guard + identity recovery); dflash itself is native (vllm PR #52816)
#   - the overlay modules import no vLLM internals (r4d.select() name registry; a missing
#     kernel degrades gracefully — you'll see "no … kernel in this r4d build, disabled")
#
# DO NOT add --enable-per-request-metrics / --enable-prompt-tokens-details: measured
# ~0.5 s periodic stream stalls (update-gap p99 523.8 ms vs 34.7 ms without).
set -euo pipefail

IMAGE=${IMAGE:-magiccodingman/vllm-radiance:1.0.11}
NAME=${NAME:-radiance-mxfp4-dflash}
PORT=${PORT:-8000}
CHUNK=${CHUNK:-8192}
R4D_ATTN=${R4D_ATTN:-1}
GDN_MERGE=${RADIANCE_GDN_MERGE_INPROJ:-1}
CACHE_SUF=""
if [ "$GDN_MERGE" = 1 ]; then CACHE_SUF="$CACHE_SUF-gdnm"; fi
# torch-2.12 caches; never share a cache dir across images/configs
CACHE=${CACHE:-$HOME/.radiance-cache-mxfp4-dflash$CACHE_SUF}
# 0.95 clears vLLM's startup free-memory gate with a little VRAM already in use (a VM
# console, a desktop). 0.97-0.98 only pass on a card that idles nearly empty — the gate
# compares util x total against CURRENT free, so ~60 MiB of idle use can make a higher
# setting unbootable. The real KV size comes from the pin below, not from this number.
GPU_UTIL=${GPU_UTIL:-0.95}
# KV pin: unset = profiled (vLLM sizes it, leaves ~1-3 GiB unused). To reclaim that,
# derive a pin for YOUR box: boot unpinned, send an 8-way load AND a >30k prefill while
# sampling /sys/class/drm/card*/device/mem_info_vram_used on the HOST, then
#   pin = "Current kv cache memory in use" + true_free_under_load - 0.3 GiB margin
# (2x R9700 32GB measured 2026-08-29: 18150000000 -> 922k KV tokens.)
KV_MEM=${KV_MEM:-}
if [ "$KV_MEM" = "0" ]; then KV_MEM=""; fi
SPEC_METHOD=${SPEC_METHOD:-dflash}
MODELS="$(realpath -m "${MODELS:-$HOME/models}")"
DRAFTER=${DRAFTER:-$MODELS/tcclaviger/Qwen3.8-27B-DFlash2-FP8}
DRAFT_ATTN=${DRAFT_ATTN:-TRITON_ATTN}
if [ "$SPEC_METHOD" = dflash ]; then SPEC=${SPEC:-7}; else SPEC=${SPEC:-4}; fi
FAST_DRAFT=${FAST_DRAFT:-1}
# 80, not 64: RADIANCE_VERIFY_HEAD needs 4x the SAMPLER's top_k (20 below) as well as
# 4x the drafter's selector_top_k (16); at 64 the verify gate silently never fires.
if [ "$SPEC_METHOD" = dflash ]; then RADIANCE_DRAFT_RERANK=${RADIANCE_DRAFT_RERANK:-80}; fi
if [ "$SPEC_METHOD" = dflash ]; then RADIANCE_VERIFY_HEAD=${RADIANCE_VERIFY_HEAD:-1}; fi
MAXLEN=${MAXLEN:-262144}
MAXSEQS=${MAXSEQS:-8}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RESTART=${RESTART:-unless-stopped}

RENDER_GID=${RENDER_GID:-$(getent group render | cut -d: -f3)}
VIDEO_GID=${VIDEO_GID:-$(getent group video | cut -d: -f3)}

# P2P RCCL (see the README: stock RCCL refuses kernel dispatch on the emulated-switch
# topology). Set RCCL_DIR="" to skip the mounts on a host with working PCIe atomics.
RCCL_DIR=${RCCL_DIR:-$HOME/rccl-build}
RCCL_MOUNTS=()
if [ -n "$RCCL_DIR" ]; then
  for f in librccl-final.so libnccldump_stub.so libnccl_devapi_stub.so; do
    [ -f "$RCCL_DIR/$f" ] || { echo "missing $RCCL_DIR/$f (see README §4, or RCCL_DIR=\"\" to skip)" >&2; exit 1; }
  done
  RCCL_MOUNTS=(-v "$RCCL_DIR/librccl-final.so":/opt/rocm/core-7.14/lib/librccl.so.1.0:ro
               -v "$RCCL_DIR/libnccldump_stub.so":/opt/rocm/core-7.14/lib/libnccldump_stub.so:ro
               -v "$RCCL_DIR/libnccl_devapi_stub.so":/opt/rocm/core-7.14/lib/libnccl_devapi_stub.so:ro)
fi

# Refuse to fight another serve for the port.
if curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" 2>/dev/null; then
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "something else already serves port $PORT -- stop it first" >&2
    exit 1
  fi
fi

# libr4d at the pinned commit + the repo's extras patch (fp8 attention legs, fused GDN
# update), built once inside the image and cached. The pin predates the repo's own
# Dockerfile bump; it is what the 2026-08 numbers were measured with.
R4D_SO=${R4D_SO:-}
R4D_PIN=${R4D_PIN:-b9e42ab}
R4D_CACHE=${R4D_CACHE:-$HOME/.cache/radiance-libr4d}
R4D_PATCH="$SCRIPT_DIR/r4d_radiance_extras.patch"
R4D_KEY="$R4D_PIN"
if [ -f "$R4D_PATCH" ]; then R4D_KEY="$R4D_PIN-rx3"; fi
if [ -z "$R4D_SO" ] && [ "${AUTO_R4D:-1}" = 1 ]; then
  if [ ! -f "$R4D_CACHE/$R4D_KEY/r4d.so" ]; then
    echo "[radiance] building libr4d $R4D_KEY in $IMAGE -- one time, a few minutes"
    rm -rf "$R4D_CACHE/.build"
    mkdir -p "$R4D_CACHE/.build"
    git clone -q https://codeberg.org/StillDeadcode/libr4d.git "$R4D_CACHE/.build"
    git -C "$R4D_CACHE/.build" checkout -q "$R4D_PIN"
    if [ "$R4D_KEY" != "$R4D_PIN" ]; then
      git -C "$R4D_CACHE/.build" apply "$R4D_PATCH"
    fi
    docker run --rm --entrypoint bash -v "$R4D_CACHE/.build":/work -w /work \
      "$IMAGE" -c ./build.sh
    mv "$R4D_CACHE/.build" "$R4D_CACHE/$R4D_KEY"
  fi
  R4D_SO="$R4D_CACHE/$R4D_KEY"
  echo "[radiance] libr4d $R4D_KEY -> $R4D_SO"
fi
MIN_M=${MIN_M:-0}
EXTRA=${EXTRA:-}

SNAP="$(realpath -m "${SNAP:-$MODELS/Qwen3.8-27B-MXFP4-mtpfp8}")"
if [ ! -f "$SNAP/config.json" ]; then
  echo "no checkpoint at $SNAP -- build it once with the ggz14 repo's fp8_mtp.py:" >&2
  echo "  hf download amd/Qwen3.8-27B-Quark-AWQ-MXFP4" >&2
  echo "  ./fp8_mtp.py <snapshot dir> $SNAP" >&2
  exit 1
fi
case "$SNAP" in
  "$MODELS"/*) CSNAP="/models/${SNAP#"$MODELS"/}" ;;
  *) echo "SNAP ($SNAP) must be under MODELS ($MODELS)" >&2; exit 1 ;;
esac

if [ "$R4D_ATTN" = "1" ]; then ATTN=R4D; else ATTN=ROCM_AITER_UNIFIED_ATTN; fi

ASYNC=${ASYNC:-0}
if [ "$ASYNC" = 1 ]; then ASYNC_FLAG="--async-scheduling"; UNPAD=false; else ASYNC_FLAG="--no-async-scheduling"; UNPAD=true; fi

if [ "$SPEC_METHOD" = dflash ]; then
  DRAFTER="$(realpath -m "$DRAFTER")"
  if [ ! -f "$DRAFTER/config.json" ]; then
    echo "no dflash drafter at $DRAFTER" >&2
    echo "  hf download tcclaviger/Qwen3.8-27B-DFlash2-FP8 --local-dir $DRAFTER" >&2
    exit 1
  fi
  case "$DRAFTER" in
    "$MODELS"/*) CDRAFTER="/models/${DRAFTER#"$MODELS"/}" ;;
    *) echo "DRAFTER ($DRAFTER) must be under MODELS ($MODELS)" >&2; exit 1 ;;
  esac
  SPEC_CFG="{\"method\":\"dflash\",\"model\":\"$CDRAFTER\",\"num_speculative_tokens\":$SPEC,\"attention_backend\":\"$DRAFT_ATTN\",\"disable_padded_drafter_batch\":$UNPAD}"
else
  SPEC_CFG="{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC,\"attention_backend\":\"$ATTN\",\"disable_padded_drafter_batch\":$UNPAD}"
fi

# Derived, not hardcoded: a prefill chunk's all-reduce is CHUNK x hidden(5120) x 2 bytes;
# a message over this cap silently falls back to RCCL (2.3x slower at this size).
AR_MAX_KB=$(( (CHUNK * 5120 * 2) / 1024 + 4096 ))

# CUDA-graph ladder: largest replayable decode batch = MAXSEQS x (SPEC+1). The default
# ladder runs far past it; every size above the ceiling is captured memory that can
# never replay (~1 GiB/GPU measured). KEEP IN SYNC if MAXSEQS or SPEC changes.
GRAPH_CEIL=$(( MAXSEQS * (SPEC + 1) ))
GRAPH_SIZES="1,2,4,8,16,24,32"
for s in 40 48 56 64 72 80 88 96 104 112 120 128; do
  [ "$s" -le "$GRAPH_CEIL" ] && GRAPH_SIZES="$GRAPH_SIZES,$s"
done

mkdir -p "$CACHE"/{vllm,inductor,triton,aiter}

echo "[run] image=$IMAGE attn=$ATTN chunk=$CHUNK ar_max_kb=$AR_MAX_KB fast_draft=$FAST_DRAFT rerank=${RADIANCE_DRAFT_RERANK:-32} vhead=${RADIANCE_VERIFY_HEAD:-0} util=$GPU_UTIL kv_mem=${KV_MEM:-profiled} graphs=$GRAPH_SIZES cache=$CACHE"

docker rm -f "$NAME" >/dev/null 2>&1 || true
exec docker run -d --name "$NAME" --restart "$RESTART" --ipc=host --network=host \
  --device /dev/kfd --device /dev/dri --group-add "$RENDER_GID" --group-add "$VIDEO_GID" \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE --cap-add SYS_NICE \
  -e ROCR_VISIBLE_DEVICES=0,1 -e HIP_VISIBLE_DEVICES=0,1 -e HF_HUB_OFFLINE=1 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
  -e VLLM_ROCM_USE_AITER_MHA=0 -e VLLM_ROCM_USE_AITER_MLA=0 -e VLLM_ROCM_USE_AITER_MOE=0 \
  -e VLLM_ROCM_USE_AITER_LINEAR=0 -e VLLM_ROCM_USE_AITER_FP8BMM=0 \
  -e VLLM_ROCM_USE_AITER_FP4BMM=0 -e VLLM_ROCM_USE_AITER_RMSNORM=0 \
  -e NCCL_PROTO=Simple \
  -e RADIANCE_USE_R4D="${RADIANCE_USE_R4D:-1}" -e RADIANCE_USE_R4D_AR="${RADIANCE_USE_R4D_AR:-1}" -e RADIANCE_USE_R4D_AR_QUANT="${RADIANCE_USE_R4D_AR_QUANT:-1}" \
  -e RADIANCE_R4D_REPORT=1 -e RADIANCE_AR_MAX_KB="$AR_MAX_KB" \
  -e RADIANCE_RUN_BWTEST="${RADIANCE_RUN_BWTEST:-0}" \
  -e RADIANCE_PRESHUFFLE="${RADIANCE_PRESHUFFLE:-1}" -e RADIANCE_FUSE_RMS_QUANT="${RADIANCE_FUSE_RMS_QUANT:-1}" \
  -e RADIANCE_MXFP4=1 -e RADIANCE_MXFP4_W4A8=1 -e RADIANCE_MXFP4_W4A8_MIN_M="$MIN_M" \
  -e RADIANCE_FAST_DRAFT="$FAST_DRAFT" -e RADIANCE_DRAFT_TAU="${RADIANCE_DRAFT_TAU:-0.20}" \
  -e RADIANCE_DRAFT_RERANK="${RADIANCE_DRAFT_RERANK:-32}" \
  -e RADIANCE_VERIFY_HEAD="${RADIANCE_VERIFY_HEAD:-0}" \
  -e RADIANCE_VERIFY_HEAD_MAX_M="${RADIANCE_VERIFY_HEAD_MAX_M:-32}" \
  -e RADIANCE_MXFP4_TN4_MIN_M="${RADIANCE_MXFP4_TN4_MIN_M:-2048}" \
  -e RADIANCE_MXFP4_DECODE_MAX_M="${RADIANCE_MXFP4_DECODE_MAX_M:-64}" \
  -e RADIANCE_MXFP4_WPERM="${RADIANCE_MXFP4_WPERM:-0}" \
  -e RADIANCE_GDN_MERGE_INPROJ="$GDN_MERGE" \
  -e R4D_ATTN_FP8="${R4D_ATTN_FP8:-3}" \
  -e RADIANCE_GDN_FUSED_UPDATE="${RADIANCE_GDN_FUSED_UPDATE:-1}" \
  -e RADIANCE_DYNAMIC_WIDTH="${RADIANCE_DYNAMIC_WIDTH:-1}" \
  -e RADIANCE_DYNW_ALPHA="${RADIANCE_DYNW_ALPHA:-0.35}" \
  -e RADIANCE_DYNW_MARGIN="${RADIANCE_DYNW_MARGIN:-2}" \
  -e RADIANCE_DYNW_MIN="${RADIANCE_DYNW_MIN:-2}" \
  -e RADIANCE_DYNW_MIN_BATCH="${RADIANCE_DYNW_MIN_BATCH:-3}" \
  -e RADIANCE_AR_QNB="${RADIANCE_AR_QNB:-96}" \
  -e RADIANCE_AR_QNT="${RADIANCE_AR_QNT:-1024}" \
  -e RADIANCE_MXFP4_EPIFAST="${RADIANCE_MXFP4_EPIFAST:-1}" \
  -e RADIANCE_TOPK_TRITON_MIN_ROWS="${RADIANCE_TOPK_TRITON_MIN_ROWS:-1}" \
  -e RADIANCE_SKINNY_GEMM="${RADIANCE_SKINNY_GEMM:-1}" \
  -e RADIANCE_GDN_PATHS="${RADIANCE_GDN_PATHS:-both}" \
  -e TRANSFORMERS_VERBOSITY=critical \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e TRITON_CACHE_DIR=/cache/triton \
  -e AITER_ROOT_DIR=/cache/aiter -e TRITON_CACHE_AUTOTUNING=1 \
  -e TORCHINDUCTOR_COMPILE_THREADS=4 \
  -v "$MODELS":/models \
  -v "$CACHE":/cache \
  -v "${PATCHES:-$SCRIPT_DIR}":/patches \
  "${RCCL_MOUNTS[@]}" \
  ${R4D_SO:+-v "$R4D_SO":/r4d} \
  ${R4D_SO:+-e R4D_SO="$R4D_SO"} \
  --entrypoint bash \
  "$IMAGE" -lc '
    set -e
    SP=/opt/vllm/lib/python3.12/site-packages
    cd /patches
    python3 patch_quark_mxfp4.py
    python3 patch_ar_maxbytes.py
    python3 patch_topk_triton_rows.py
    python3 patch_dflash_calib.py
    # patch_dflash_mxfp4_kv.py deliberately NOT run: vLLM 0.28 ships the fix natively
    # (_DFLASH_DENSE guard + identity recovery in qwen3_dflash.py).
    python3 patch_rmsquant_fusion.py
    python3 patch_verify_head.py
    python3 patch_kv_group_size.py
    python3 patch_gdn_merge_inproj.py
    python3 patch_dynwidth.py
    python3 patch_ar_geometry.py
    python3 patch_qwen3_thinkoff.py \
      || echo "[radiance] WARNING: thinkoff patch did not apply; thinking-off requests will return empty content"
    # Cosmetic: newer transformers prints [ERROR] docstring-lint lines via a raw print().
    python3 patch_transformers_docstring_lint.py \
      || echo "[radiance] WARNING: docstring-lint silencer did not apply (harmless noise remains)"
    cp mxfp4-configs/*.json "$SP"/aiter/ops/triton/configs/gemm/
    cp radiance_mxfp4.py radiance_gdn.py radiance_rmsquant.py radiance_drafthead.py \
       radiance_verifyhead.py radiance_gdnmerge.py radiance_aroverlap.py "$SP"/
    hipcc -O3 -w -std=c++17 -fPIC -shared --offload-arch=gfx1201 $(python3 -m pybind11 --includes) \
      radiance_mxfp4_fp8.hip -o "$SP"/radiance_mxfp4_fp8.so
    if [ -n "${R4D_SO:-}" ] && [ -f /r4d/r4d.so ]; then
      cp /r4d/r4d.so "$SP"/r4d.so
      echo "[radiance] using patched r4d.so from $R4D_SO"
    fi
    cd /
    exec /opt/radiance_entrypoint.sh "$@"' _ \
    "$CSNAP" --served-model-name qwen3.8-27b Qwen3.8 Qwen3.8-MXFP4 \
    --host 0.0.0.0 --port "$PORT" \
    --kv-cache-dtype fp8 --tensor-parallel-size 2 \
    --gpu-memory-utilization "$GPU_UTIL" \
    ${KV_MEM:+--kv-cache-memory "$KV_MEM"} \
    --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" --max-num-batched-tokens "$CHUNK" \
    --attention-backend "$ATTN" \
    --compilation-config "{\"cudagraph_capture_sizes\":[$GRAPH_SIZES]}" \
    --speculative-config "$SPEC_CFG" \
    $ASYNC_FLAG $EXTRA \
    --enable-prefix-caching --mamba-cache-mode align \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --trust-remote-code \
    --language-model-only \
    --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
    --default-chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "xhigh"}' \
    --chat-template "$CSNAP/chat_template.jinja"
