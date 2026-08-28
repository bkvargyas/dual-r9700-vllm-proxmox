# BetterBench results — config F (DFlash2 + P2P one-shot AR)

Measured 2026-08-28 with [BetterBench](https://github.com/GGZ14/BetterBench) 0.2.3,
default profile (20 passes/category, cold prefix cache via unique nonces) plus a 128k
prefill depth. Raw data: [`dflash-f.json`](dflash-f.json) · charted report:
[`dflash-f.html`](dflash-f.html) (self-contained, open in a browser).

## Setup (community posting format)

- **GPUs**: 2× AMD Radeon AI PRO R9700 32GB, each PCIe 5.0 x16 (Proxmox VFIO passthrough)
- **P2P**: ON (emulated-switch topology + XanMod kernel + NDEBUG RCCL — see the [README](../README.md)); **ReBAR**: ON (full 32GB BAR)
- **Image**: `magiccodingman/vllm-radiance:1.0.11` (vLLM 0.28, ROCm 7.14), TP=2
- **Model**: Qwen3.8-27B-FP8, FP8 KV cache, DFlash2 drafter (`z-lab/Qwen3.8-27B-DFlash2`, `TRITON_ATTN`, k=7)
- **Context**: 200k max-model-len · `max-num-seqs 8` · chunk 8192 + `RADIANCE_AR_MAX_KB=98304`
- **Server defaults**: thinking ON (`reasoning_effort xhigh`) — numbers reflect the endpoint as actually served
- **Headline**: prefill ~5,000 t/s (≤8k) / 3,430 t/s @128k · decode 100.2 t/s weighted (142 best-category) · TTFT p50 65 ms

## Single-stream (batch = 1)

| category | TTFT p50 (ms) | PP t/s | ITL 1% low | ITL median | decode t/s (med) |
|---|--:|--:|--:|--:|--:|
| chat | 143.1 | 1000.5 | 44.7 | 66.2 | 67.6 |
| code | 58.1 | 1913.3 | 16.0 | 71.1 | 89.6 |
| file_edit | 94.4 | 1986.1 | 87.7 | 120.2 | 129.6 |
| json | 57.8 | 2026.4 | 122.3 | 137.7 | 142.6 |
| math | 56.0 | 1583.5 | 102.7 | 130.5 | 141.4 |
| prose | 57.3 | 1910.8 | 58.9 | 66.3 | 70.4 |
| reasoning | 57.5 | 1825.8 | 58.5 | 70.8 | 75.4 |
| summarization | 94.5 | 2111.6 | 110.0 | 128.9 | 133.5 |

**Combined (weighted)**: decode **100.2 t/s** median · ITL 1%-low 63.5 t/s · TTFT p50 **65 ms**

(Thinking-heavy categories — chat, prose, reasoning, code — decode slower because the
reasoning trace dominates; json/math/file_edit show the drafter's ceiling.)

## Concurrency sweep (48 requests per level)

| level | aggregate t/s | TTFT p50 (ms) | TTFT p99 (ms) | per-req decode (med) |
|--:|--:|--:|--:|--:|
| 1 | 98.5 | 57.9 | 98.1 | 118.9 |
| 2 | 165.0 | 103.3 | 174.9 | 111.2 |
| 4 | 278.2 | 147.8 | 1531.2 | 100.8 |
| 8 | 390.4 | 158.5 | 1104.7 | 80.6 |
| 16 | 418.5 | 5874.8 | 8285.9 | 78.5 |

The knee is at 8: level 16 buys +7% aggregate for an 8.3 s TTFT p99. `--max-num-seqs=8`
is the right serving cap for this stack.

## Prefill sweep (cold prefix cache, tiny decode)

| depth | prompt tokens (med) | TTFT p50 | PP t/s median |
|--:|--:|--:|--:|
| 2000 | 1556 | 317 ms | 4,902 |
| 8000 | 5960 | 1.19 s | 4,997 |
| 16000 | 11836 | 2.43 s | 4,880 |
| 32000 | 23585 | 5.13 s | 4,600 |
| 64000 | 47098 | 11.4 s | 4,131 |
| 128000 | 94107 | 27.4 s | 3,430 |

With prefix caching on (server default), repeated prefixes return in well under a second
at any depth — the cold numbers above are the worst case.
