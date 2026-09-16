# BetterBench results — second box (PLX-switch host, NVFP4, P2P one-shot AR)

Measured 2026-09-16 with [BetterBench](https://github.com/GGZ14/BetterBench) 0.6.0,
default profile (20 passes/category, cold prefix cache via unique nonces). Raw data:
[`plx-p2p-nvfp4-tp2-bb060.json`](plx-p2p-nvfp4-tp2-bb060.json) · charted report:
[`plx-p2p-nvfp4-tp2-bb060.html`](plx-p2p-nvfp4-tp2-bb060.html) (self-contained, open in a browser).

This is a **second, physically different machine** from the production box the other results
come from. Here the two R9700 sit behind **two separate PLX PEX 8747 switch chips** (not
separate root ports), which made both the 32GB BAR and P2P setup materially harder — see the
[setup notes](#setup-notes-plx-specifics) below.

## Setup (community posting format)

- **Host**: Gigabyte MZ22-G20, AMD EPYC 7H12 (64c), 440GB RAM, Proxmox VE 9.2 (VFIO passthrough)
- **GPUs**: 2× AMD Radeon AI PRO R9700 32GB, each behind its **own PLX PEX 8747 (Gen3 x16)** switch chip
- **P2P**: ON (emulated-switch topology + XanMod kernel + NDEBUG RCCL); **ReBAR**: ON (full 32GB BAR, forced per-chain)
- **Stack**: [GGZ14/vllm-mxfp4](https://github.com/GGZ14/vllm-mxfp4) (repo 0.12.0, commit `92eed82`) — pulls the pinned base image `stilldeadcode/vllm-radiance:0.9.3` (torch 2.11 / ROCm 7.14, RCCL 2.30) and applies its MXFP4/NVFP4 + R4D patches at container start; TP=2
- **Model**: `unsloth/Qwen3.8-27B-NVFP4` (NVFP4 → MXFP4 requant at load, `RADIANCE_NVFP4_MXFP4=1`), FP8 KV cache, DFlash2-FP8 drafter (k=7)
- **Context**: 262k max-model-len · `max-num-seqs 8` · chunk 8192 · GPU util 0.95 · 225W power cap/card
- **All-reduce**: R4D P2P one-shot (`RADIANCE_USE_R4D_AR=1`), byte-identical to RCCL
- **Server defaults**: thinking ON (Qwen3.8 reasoning model) — numbers reflect the endpoint as actually served
- **Headline**: prefill ~4,780–4,950 t/s (≤16k) / 4,470 t/s @64k · decode 195.9 t/s weighted (268 best-category) · TTFT p50 65 ms · aggregate 544 t/s @8

## Single-stream (batch = 1)

This server packs several tokens per stream update (speculative decoding), so the latency
figure is the wall-clock gap between updates, not per-token. **update p50/p99** in ms;
**tok/update** is tokens per update; decode = per-run tok/s.

| category | TTFT p50 (ms) | update p50 (ms) | update p99 (ms) | tok/update | decode t/s (med) |
|---|--:|--:|--:|--:|--:|
| chat | 70.4 | 23.4 | 24.0 | 3.33 | 139.2 |
| code | 64.9 | 23.6 | 24.3 | 4.64 | 218.1 |
| file_edit | 70.3 | 23.6 | 24.3 | 6.04 | 268.0 |
| json | 65.4 | 23.5 | 24.2 | 5.49 | 254.0 |
| math | 54.5 | 23.6 | 24.4 | 5.77 | 252.5 |
| prose | 55.0 | 23.5 | 24.3 | 2.74 | 118.1 |
| reasoning | 65.0 | 23.6 | 24.3 | 3.30 | 139.8 |
| summarization | 71.0 | 23.5 | 24.5 | 4.70 | 198.9 |

**Combined (weighted)**: decode **195.9 t/s** median · update p99 **24.3 ms** · TTFT p50 **65 ms**

Qwen3.8 is a thinking model: 81% of single-stream runs spent their token budget on the reasoning
trace before finishing an answer (BetterBench's answer-split section in the JSON/HTML has the
per-category detail). The token-rate figures above are unaffected.

## Concurrency sweep (48 requests per level)

| level | aggregate t/s | TTFT p50 (ms) | TTFT p99 (ms) | per-req decode (med) |
|--:|--:|--:|--:|--:|
| 1 | 175.5 | 65.8 | 85.7 | 213.2 |
| 2 | 294.7 | 93.8 | 135.1 | 203.0 |
| 4 | 426.8 | 105.8 | 207.3 | 151.3 |
| 8 | 544.1 | 135.1 | 505.4 | 99.5 |

## Prefill sweep (cold prefix cache, tiny decode)

| depth | prompt tokens (med) | TTFT p50 | PP t/s median |
|--:|--:|--:|--:|
| 2000 | 1514 | 317 ms | 4,776 |
| 8000 | 5892 | 1.20 s | 4,950 |
| 16000 | 11800 | 2.40 s | 4,906 |
| 32000 | 23548 | 4.97 s | 4,745 |
| 64000 | 47014 | 10.5 s | 4,470 |

## P2P impact (this box)

Quick before/after on the same stack, all-reduce path only (RCCL fallback vs R4D P2P):

| metric | RCCL (no P2P) | R4D P2P | gain |
|---|--:|--:|--:|
| single-stream decode | 108 | 130 t/s | +21% |
| aggregate @8 | 351 | 494 t/s | +41% |
| prefill ~500 tok | 966 | 2,607 t/s | +170% |
| prefill ~2000 tok | 2,757 | 4,406 t/s | +60% |
| prefill ~8000 tok | 3,024 | 4,855 t/s | +61% |

The gains land in prefill and multi-stream aggregate, which are all-reduce bound — exactly
where P2P replaces the host-bounced RCCL fallback.

## Setup notes (PLX specifics)

Two things differ from the separate-root-port production box and are worth recording for anyone
on a PLX-switch board:

1. **32GB BAR must be forced per PLX chain.** The firmware sizes each PLX bridge's prefetchable
   window for 256MB, so a card's ReBAR gets squeezed back down on any reallocation/FLR. The fix
   is a host-side pass that sets each card's ReBAR control to 32GB, programs the root-port +
   PLX-upstream 64-bit prefetch windows large enough, then removes the root port and rescans —
   run once per chain, before the VM starts (a Proxmox pre-start hookscript). One card in the
   original pair also turned out to be a genuine PSP-init failure under passthrough and was
   replaced; the replacement works.
2. **P2P needs all three of**: the emulated PCIe switch in the guest (so `pci_p2pdma` permits
   peer DMA), the XanMod kernel (`CONFIG_HSA_AMD_P2P=y`; stock Debian ships it off) plus
   `amdgpu.pcie_gen_cap`/`pcie_lane_cap` on the guest cmdline (the emulated switch drops PCIe
   atomics, so amdgpu SMU init fails without them), and the NDEBUG-rebuilt RCCL bind-mounted over
   the image's copy (vLLM's pynccl sanity all-reduce dies on the atomics-less switch, ROCm#6520).
