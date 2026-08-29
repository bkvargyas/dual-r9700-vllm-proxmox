# BetterBench results — MXFP4-W4A8 target + FP8 DFlash2 drafter

Measured 2026-08-29 with [BetterBench](https://github.com/GGZ14/BetterBench) **0.4.0**.

> **Not comparable with the 0.2.3 numbers** in [RESULTS.md](RESULTS.md): 0.4.0 measures
> real stream-update gaps instead of synthesizing per-token ITL, splits thinking from
> answering, and gates percentiles on sample size. The like-for-like baseline below is a
> fresh 0.4.0 run of the same FP8+DFlash2 config RESULTS.md describes.

All three runs: identical config (20 passes/category, cold prefix cache, concurrency
48 req × levels 1–16, prefill sweep 2k–128k), same LAN client, same VM, the endpoint's
own serving defaults (thinking ON, `reasoning_effort xhigh`).

## The three configs

| | baseline | mxfp4-dflash (0.27.1) | mxfp4-dflash (0.28 port) |
|---|---|---|---|
| image | `magiccodingman/vllm-radiance:1.0.11` | `stilldeadcode/vllm-radiance:0.9.3` | `magiccodingman/vllm-radiance:1.0.11` |
| vLLM / torch | 0.28.0 / 2.12 | 0.27.1 / 2.11 | 0.28.0 / 2.12 |
| target | Qwen3.8-27B-FP8 | Qwen3.8-27B-MXFP4 (W4A8) | Qwen3.8-27B-MXFP4 (W4A8) |
| drafter | DFlash2 bf16 (z-lab) | DFlash2-FP8 (tcclaviger) | DFlash2-FP8 (tcclaviger) |
| dflash impl | native | ggz14 backport | native |
| radiance extras | — | int2 fast-draft + verify-head, GDN merge, libr4d b9e42ab-rx2, small-M decode GEMM | same, overlaid at container start |
| KV capacity | ~202k tokens | 940k tokens | 922k tokens |
| raw data | [json](baseline-dflash-fp8-magic1011-bb040.json) · [html](baseline-dflash-fp8-magic1011-bb040.html) | [json](candidate-mxfp4-dflash-sd093-bb040.json) · [html](candidate-mxfp4-dflash-sd093-bb040.html) | [json](candidate-mxfp4-dflash-mcm1011-vllm028-bb040.json) · [html](candidate-mxfp4-dflash-mcm1011-vllm028-bb040.html) |

## Headline comparison

| metric | baseline (FP8+dflash) | mxfp4-dflash 0.27.1 | mxfp4-dflash 0.28 port |
|---|--:|--:|--:|
| combined decode t/s (weighted median) | 106.2 | **154.2** | 151.3 |
| stream update gap p50 | 39.1 ms | 25.5 ms | ~24.5 ms |
| stream update gap p99 (weighted) | 56.8 ms | **36.9 ms** | 523.8 ms¹ |
| TTFT p50 (weighted) | 155 ms | **165 ms** | 206 ms¹ |
| aggregate @ 8 streams | 435.4 | 464.6 | 465.2 |
| aggregate @ 16 streams | 404.3 (degrading) | 490.2 | 494.4 |
| prefill @ 32k (median) | 4,365 | 4,710 | 4,710 |
| prefill @ 128k (median) | 3,376 | **4,085** | 4,076 |
| GPU KV cache | ~202k tok | **940k tok** | 922k tok |

¹ measured with `--enable-per-request-metrics --enable-prompt-tokens-details` active —
see the note below. A 10-pass decode-only re-run **without** those flags:
combined decode **159.1 t/s**, update p99 **34.7 ms** — the stall is entirely the flags;
a ~35 ms TTFT tax vs the 0.27.1 stack remains (small prompts only, deep prefill at parity).

**Adopted: the 0.28 port** (without the metrics flags) — decode at parity-or-better,
best stream smoothness, native dflash instead of a pinned backport.

## Per-category single-stream decode t/s (median)

| category | baseline | mxfp4-dflash 0.27.1 | Δ |
|---|--:|--:|--:|
| code | 94.7 | 151.0 | +59% |
| json | 145.9 | 207.5 | +42% |
| math | 136.8 | 208.8 | +53% |
| file_edit | 136.6 | 188.8 | +38% |
| summarization | 134.5 | 186.3 | +38% |
| prose | 80.0 | 116.2 | +45% |
| reasoning | 84.2 | 114.1 | +36% |
| chat | 71.9 | 111.4 | +55% |

## Output-quality gates (run on every adopted config)

- 24 sequential factual probes: all correct, 0 repeated-8-gram loops, no mojibake
- tool-call parsing (`qwen3_coder`), thinking on/off paths (incl. the thinkoff patch)
- prefix-cache-HIT correctness under dflash: 6 sequential hits against a 4k system
  prompt, all correct, TTFT 1.6s → 0.65s (this box's historical dflash failure mode was
  corruption on cache hits — always test hits, cold-cache benchmarks can't see it)
- 55k-token needle retrieval
- restart-in-place (`docker restart` → healthy, patch battery idempotent)

## Notes

- The per-request metrics flags cost real streaming latency on the 0.28 port: a
  periodic ~0.5 s update-gap stall (p99 523.8 ms with them, 34.7 ms without; A/B'd
  directly). Leave `--enable-per-request-metrics` / `--enable-prompt-tokens-details`
  off unless you consume them.
- Independent of those flags, the 0.28 port pays ~35 ms more TTFT than the 0.27.1
  stack on short prompts (p50 ~198 vs 165 ms); deep prefill and decode are at parity.
  Root cause not chased — it's a fixed per-request cost, invisible at depth.
- xhigh thinking frequently spends the whole token budget before the answer begins
  (~70% of bounded-budget runs across all three configs). That is the serving default,
  not a stack property — budget-capped clients may prefer `reasoning_effort medium`.
