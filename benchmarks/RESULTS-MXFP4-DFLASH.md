# BetterBench results — MXFP4-W4A8 target + FP8 DFlash2 drafter

Measured 2026-08-29/30 with [BetterBench](https://github.com/GGZ14/BetterBench) **0.4.0**.

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
| radiance extras | — | int2 fast-draft + verify-head, GDN merge, libr4d b9e42ab-rx2, small-M decode GEMM | same @ upstream `2d72e78` (adds dynamic verify width, tunable compressed-AR geometry, libr4d rx3), overlaid at container start |
| KV capacity | ~202k tokens | 940k tokens | 922k tokens |
| raw data | [json](baseline-dflash-fp8-magic1011-bb040.json) · [html](baseline-dflash-fp8-magic1011-bb040.html) | [json](candidate-mxfp4-dflash-sd093-bb040.json) · [html](candidate-mxfp4-dflash-sd093-bb040.html) | [json](final2-mxfp4-dflash-vllm028-dynwidth-bb040.json) · [html](final2-mxfp4-dflash-vllm028-dynwidth-bb040.html) (pre-dynwidth run: [json](final-mxfp4-dflash-vllm028-bb040.json) · [html](final-mxfp4-dflash-vllm028-bb040.html)) |

## Headline comparison

| metric | baseline (FP8+dflash) | mxfp4-dflash 0.27.1 | mxfp4-dflash 0.28 (adopted) |
|---|--:|--:|--:|
| combined decode t/s (weighted median) | 106.2 | 154.2 | **164.2** |
| stream update gap p50 | 39.1 ms | 25.5 ms | **25.1 ms** |
| stream update gap p99 (weighted) | 56.8 ms | **36.9 ms** | 42.3 ms |
| TTFT p50 (weighted) | 155 ms | **165 ms** | 216 ms¹ |
| aggregate @ 8 streams | 435.4 | 464.6 | **467.6** |
| aggregate @ 16 streams | 404.3 (degrading) | 490.2 | **502.9** |
| prefill @ 16k / 32k (median) | 4,490 / 4,365 | 4,734 / 4,710 | 4,682 / 4,722 |
| prefill @ 128k (median) | 3,376 | **4,085** | 4,074 |
| GPU KV cache | ~202k tok | **940k tok** | 922k tok |

The adopted column is upstream `2d72e78` (2026-08-30, with dynamic verify width). The
same port the day before (pre-dynwidth) measured combined 161.5 / @8 474.7 / @16 470.7
— dynwidth's win is concentrated where its author measured it: batching (@16 +7%) and
the low-acceptance categories (code +8%, reasoning +9% single-stream).

¹ the 0.28 port pays ~35–50 ms more TTFT than the 0.27.1 stack on short prompts; deep
prefill and decode are at parity. Root cause unchased — a fixed per-request cost,
invisible at depth.

**Adopted: the 0.28 port** — best decode and @8 aggregate, clean streaming, native
dflash instead of a pinned backport. An earlier 20-pass run of the same port WITH
`--enable-per-request-metrics --enable-prompt-tokens-details`
([json](candidate-mxfp4-dflash-mcm1011-vllm028-bb040.json) ·
[html](candidate-mxfp4-dflash-mcm1011-vllm028-bb040.html)) measured combined decode
151.3 with a ~0.5 s periodic stream stall (update p99 **523.8 ms**) — kept as the
demonstration of what those flags cost on this profile.

## Per-category single-stream decode t/s (median)

| category | baseline | mxfp4-dflash 0.27.1 | mxfp4-dflash 0.28 @2d72e78 (adopted) | Δ vs baseline |
|---|--:|--:|--:|--:|
| json | 145.9 | 207.5 | **216.5** | +48% |
| math | 136.8 | 208.8 | **208.8** | +53% |
| file_edit | 136.6 | 188.8 | **201.6** | +48% |
| summarization | 134.5 | 186.3 | **184.8** | +37% |
| code | 94.7 | 151.0 | **169.9** | +79% |
| reasoning | 84.2 | 114.1 | **125.4** | +49% |
| prose | 80.0 | 116.2 | **113.4** | +42% |
| chat | 71.9 | 111.4 | **109.3** | +52% |

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
