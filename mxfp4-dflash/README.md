# MXFP4-W4A8 + DFlash2-FP8 profile (vLLM 0.28)

The fast profile: **+50% single-stream decode over the FP8 profile, +15% prefill, and
4.6× the KV capacity** (922k tokens — real multi-agent headroom at 200k+ contexts),
same hardware.
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

## Rebuilding this stack from scratch (disaster recovery)

Everything below is what the running server is actually made of. With these pins, a
bare VM (that already has the main README's P2P plumbing) rebuilds to a byte-equivalent
stack; nothing is fetched at container runtime except what's listed.

**Exact pins (verified working together, 2026-08-31):**

| component | pin |
|---|---|
| serving image | `magiccodingman/vllm-radiance:1.0.11` — digest `sha256:a3b1b26439260a3fc4fc23a30fd94e8b624123f8837aa328866f48e59f2d2f4b` (vLLM 0.28.0, torch 2.12, ROCm core-7.14) |
| patch/module source | `codeberg.org/ggz14/radiance-vllm-mxfp4` @ **`b9d7ecf`** |
| libr4d | `codeberg.org/StillDeadcode/libr4d` @ **`b9e42ab`** + the ggz14 repo's `r4d_radiance_extras.patch` (the "rx4" build — the patch is versioned inside the pinned ggz14 commit) |
| target checkpoint | `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (HF), converted once by the ggz14 repo's `fp8_mtp.py` → `Qwen3.8-27B-MXFP4-mtpfp8` (~19 GB) |
| drafter checkpoint | `tcclaviger/Qwen3.8-27B-DFlash2-FP8` (HF, ~2 GB, single safetensors) |
| RCCL | rebuilt 2.27.7 (`release/rocm-rel-7.1.1.1` + NDEBUG) + the two stubs from [`../rccl-stubs/`](../rccl-stubs/) — main README §4 |
| this repo | `run_mxfp4_dflash.sh` + `patch_transformers_docstring_lint.py` from this directory |

Pull images **by digest**, not tag — tags on both Docker Hub repos have moved before:
`docker pull magiccodingman/vllm-radiance@sha256:a3b1b264…`. (For reference, the
0.27.1-era alternative was `stilldeadcode/vllm-radiance:0.9.3` @
`sha256:4569420917…` — not needed for this profile.)

**Order of operations** (times for a 2× R9700 / EPYC box):

1. Main README first: VM PCIe-switch config, XanMod kernel, amdgpu grub params,
   RCCL rebuild (~40–85 min, once ever) → `~/rccl-build/`.
2. `git clone https://codeberg.org/ggz14/radiance-vllm-mxfp4 && git -C radiance-vllm-mxfp4 checkout b9d7ecf`,
   copy this directory's two files in.
3. Models: download the drafter; download AMD's MXFP4 release and run `fp8_mtp.py`
   (~15 min, once ever).
4. `MODELS=$HOME/models ./run_mxfp4_dflash.sh` — first run builds libr4d inside the
   image (~5 min, cached in `~/.cache/radiance-libr4d/b9e42ab-rx4/`) and cold-compiles
   Triton/inductor (~7–10 min, cached in the `CACHE` dir). Warm boots: ~3 min.
5. Re-derive box-specifics rather than copying them: the KV pin (recipe in the
   launcher; profiled boots work fine while you measure) and the render/video GIDs
   (the launcher auto-detects via `getent`).
6. Gate before trusting it (all of these bit us at least once): the log markers above,
   a prefix-cache-**hit** correctness probe, a long-context needle, one
   `docker restart` cycle.

**Bit-rot insurance:** the two codeberg repos and the HF drafter are one-person
projects — keep local mirrors. A running deployment already holds everything needed
(the repo clone, the built libr4d directory, the converted checkpoint, `~/rccl-build`)
— back those four paths up and step 1–3 become a restore instead of a rebuild.
Upstream moves fast and its patches are written against exact vLLM file contents:
before adopting a newer ggz14 commit, dry-run the patch battery against the image
(`docker run --rm --entrypoint bash -v $PWD:/patches <image> -lc 'cd /patches && python3 patch_<name>.py'`)
— an anchor mismatch means that patch needs re-porting, not forcing.

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
