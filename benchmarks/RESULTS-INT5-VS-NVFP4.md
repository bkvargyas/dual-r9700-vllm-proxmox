# NVFP4 vs int5 ParoQuant — accuracy and speed (second box, P2P TP=2)

Both quantizations of the same Qwen3.8-27B, served on the [GGZ14/vllm-mxfp4](https://github.com/GGZ14/vllm-mxfp4)
stack (pinned `stilldeadcode/vllm-radiance:0.9.3`), TP=2 with the R4D P2P all-reduce, same DFlash2-FP8
drafter, same corpus, same chat template, on the PLX-switch box (2× R9700, 225W/card).

- **NVFP4**: `unsloth/Qwen3.8-27B-NVFP4`, NVFP4→MXFP4 requant at load (4.25 bits/weight). Report:
  [`plx-p2p-nvfp4-tp2-bb060.html`](plx-p2p-nvfp4-tp2-bb060.html)
- **int5 ParoQuant**: `Launch80/Qwen3.8-27B-PARO-int5` (W5A8, 5.25 bits/weight). Report:
  [`int5-paro-w5a8-tp2-bb060.html`](int5-paro-w5a8-tp2-bb060.html)

## Accuracy

| eval | NVFP4 | int5 ParoQuant |
|---|--:|--:|
| **HumanEval** pass@1 (164, greedy, thinking on) | **98.2%** (161/164) | 96.3% (158/164) |
| **GSM8K** (200, greedy, thinking on) | 97.0% (194/200) | 96.5% (193/200) |

On HumanEval, NVFP4 solved a **strict superset** of int5's problems (int5's failures 10/32/38/39/50/145;
NVFP4's 38/50/145 all shared). int5's lower divergence-from-bf16 (KL / top-1 agreement, per the checkpoint)
did **not** translate into better code correctness here. GSM8K is a tie within noise.

## Speed (BetterBench 0.6.0, 20 passes/category, cold prefix)

| metric | NVFP4 | int5 ParoQuant | int5 vs NVFP4 |
|---|--:|--:|--:|
| combined decode (t/s) | 195.9 | 148.8 | −24% |
| update step p99 (ms) | 24.3 | 28.0 | +15% (slower) |
| TTFT p50 (ms) | 65 | 84 | +29% |
| aggregate @8 (t/s) | 544.1 | 422.6 | −22% |
| prefill 2k (t/s) | 4,776 | 3,631 | −24% |
| prefill 8k (t/s) | 4,950 | 3,624 | −27% |
| prefill 16k (t/s) | 4,906 | 3,580 | −27% |
| prefill 64k (t/s) | 4,470 | 3,381 | −24% |
| weights per card (GiB) | 10.4 | 12.2 | +17% |
| KV cache (tokens) | larger | 483,850 | less headroom |

## Verdict

For throughput-oriented serving and for coding specifically, **NVFP4 is the better choice on this box**:
~24% faster on decode and prefill, more KV headroom for long context, and at least as accurate on code
(HumanEval) with GSM8K a tie. int5 W5A8 keeps its niche where divergence-from-bf16 matters directly —
logprobs, draft-acceptance studies — but it did not win any task-accuracy measurement here, and it costs a
full extra bit per weight in bandwidth and memory. NVFP4 is the standing config.

*Method notes: pass@1 is a single greedy sample per problem; a 3-problem HumanEval gap at n=164 is inside
the noise band, so read it as "NVFP4 at least as accurate", not "decisively better". HumanEval is
single-function completion, not a multi-step agent loop. Qwen3.8 is a thinking model, so most decode runs
spend tokens on the reasoning trace; token-rate figures are unaffected.*
