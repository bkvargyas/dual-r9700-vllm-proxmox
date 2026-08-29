# MXFP4-W4A8 + DFlash2-FP8 profile (vLLM 0.28)

The fast profile: **+45% single-stream decode over the FP8 profile and 4.6× the KV
capacity** (922k tokens — real multi-agent headroom at 200k+ contexts), same hardware.
Numbers and methodology: [../benchmarks/RESULTS-MXFP4-DFLASH.md](../benchmarks/RESULTS-MXFP4-DFLASH.md).

It runs [ggz14/radiance-vllm-mxfp4](https://codeberg.org/ggz14/radiance-vllm-mxfp4)'s
kernel stack — native-MXFP4 W4A8 GEMMs, int2 fast-draft + verify heads, GDN in_proj
merge, a patched libr4d — ported from its 0.27.1 image onto the vLLM 0.28 image the
rest of this guide uses (native DFlash2, no backport).

## Setup

```bash
# 1. the patch/module source — the launcher runs FROM this clone
git clone https://codeberg.org/ggz14/radiance-vllm-mxfp4
cp run_mxfp4_dflash.sh patch_transformers_docstring_lint.py radiance-vllm-mxfp4/

# 2. models (once)
hf download tcclaviger/Qwen3.8-27B-DFlash2-FP8 --local-dir ~/models/tcclaviger/Qwen3.8-27B-DFlash2-FP8
hf download amd/Qwen3.8-27B-Quark-AWQ-MXFP4
cd radiance-vllm-mxfp4
./fp8_mtp.py "$(ls -d ~/.cache/huggingface/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/*/ | head -1)" \
             ~/models/Qwen3.8-27B-MXFP4-mtpfp8

# 3. serve (first run builds libr4d ~5 min, then compiles ~10 min; warm restarts ~3 min)
MODELS=$HOME/models ./run_mxfp4_dflash.sh
docker logs -f radiance-mxfp4-dflash
```

The P2P RCCL rebuild from the main README is still required (the launcher mounts it
from `~/rccl-build`; `RCCL_DIR="" ` skips the mounts on hosts with working atomics).

## What to check in the log

- `Using RadianceMxfp4W4A8LinearKernel for MXFP4 GEMM` — the W4A8 kernel won selection
  (the stock "platform does not support native MXFP4" warning above it is a false alarm)
- `Capturing dflash2 CUDA graphs (FULL)` — the drafter kept its graph
- the `R4D kernel selection` table — `no … gemm_nt kernel` lines are expected
  (this libr4d pin doesn't ship them; the fallback is graceful and deliberate)

## Sharp edges found porting this (in rough order of pain)

- **vLLM's startup memory gate compares `util × total` against *current* free VRAM.**
  ~60 MiB of idle use (VM console framebuffer) makes `GPU_UTIL=0.98` unbootable
  forever. Run 0.95 and size KV with the pin instead — the pin overrides util anyway.
- **The radiance startup bandwidth sweep races that same gate** (it's backgrounded and
  holds ~0.7 GiB while vLLM profiles). `RADIANCE_RUN_BWTEST=0` in this launcher.
- **`--enable-per-request-metrics` / `--enable-prompt-tokens-details` cause ~0.5 s
  periodic stream stalls** on this stack (update-gap p99 523.8 ms → 34.7 ms without).
- **Trim the CUDA-graph ladder to `max_num_seqs × (spec_tokens + 1)`.** The default
  ladder captures ~1 GiB/GPU of graphs that can never replay; the launcher derives it.
- **`RADIANCE_DRAFT_RERANK=80`, not 64**, when the verify head is on with sampler
  `top_k 20` — the head needs 4× the sampler's top_k or it silently never engages.
- Benchmark prefix-cache **hits**, not just cold-cache runs — dflash corruption bugs
  historically hid on the cache-hit path, which nonce-based benchmarks never touch.
