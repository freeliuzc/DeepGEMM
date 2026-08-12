# Add an opt-in EVICT_FIRST hint for small-batch MegaMoE

## Summary

This change adds an opt-in SM100 FP8xFP4 MegaMoE cache-policy tuning switch.
When `DG_MEGA_MOE_WEIGHT_EVICT_FIRST=1`, routed expert-weight demand TMA loads
use the GPU L2 `EVICT_FIRST` replacement hint. The normal path remains the
default (`0`), and shared-expert weights and scale-factor loads are unchanged.

The hint is deliberately independent of speculative previous-active-expert
prefetch. The attribution measurements show that the cache hint is the main
small-token contributor, while prefetch is routing-sensitive and is not ready
to be enabled by default in an upstream patch.

## Performance

Three alternating repetitions with the same v3 two-CTA configuration and
random routing gave the following mean latency improvements over normal cache
policy:

| Tokens | `EVICT_FIRST` only |
| ---: | ---: |
| 8 | +5.17% |
| 16 | +4.07% |
| 32 | +4.28% |

A same-session 512-token recheck measured +1.64% for the hint alone. A single-
run extension screen crossed over at larger batches (+/-0.84% at 1024 and
-2.31% at 2048), so callers should gate the switch to their workload rather
than assuming it is universally beneficial.

Configuration: DeepSeek-V4-Flash-FP8, 8 GPU ranks, SM100, FP8xFP4 MegaMoE,
v3 two-CTA configuration, hidden size 4096, intermediate size 2048, 256 total
experts, 32 local experts per rank, top-k 6. Latency is the eight-rank mean
from alternating A/B runs; the 8/16/32-token attribution used three
repetitions per variant.

| Tokens per rank | `EVICT_NORMAL` | `EVICT_FIRST` | Improvement |
| ---: | ---: | ---: | ---: |
| 8 | 101.937 us | 96.669 us | +5.17% |
| 16 | 110.169 us | 105.688 us | +4.07% |
| 32 | 113.270 us | 108.419 us | +4.28% |
| 512 | 140.901 us | 138.586 us | +1.64% |

The raw attribution CSVs and the full experiment notes are retained in the
separate experiment archive; they are intentionally not part of this focused
source PR.

## Repro

Build the branch in a CUDA/SM100 environment, then run the existing MegaMoE
test with a fresh JIT cache. Run the two variants in alternating order and
repeat each token count at least three times:

```bash
export DG_JIT_CACHE_DIR=/tmp/deepgemm-evict-first-cache
for hint in 0 1; do
  DG_MEGA_MOE_WEIGHT_EVICT_FIRST=$hint \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python tests/test_mega_moe.py \
    --num-processes 8 \
    --num-max-tokens-per-rank 512 \
    --num-tokens 8 \
    --hidden 4096 \
    --intermediate-hidden 2048 \
    --num-shared-experts 0 \
    --num-experts 256 \
    --num-topk 6 \
    --mma-type fp8xfp4 \
    --num-correctness-tests 1
done
```

For a performance comparison, disable the legacy baseline and collect the
fused-kernel latency from the existing benchmark output. Use a clean GPU node,
alternate the variant order, and report rank means rather than a single run.

## Scope

- Add a compile-time cache-hint parameter to the existing TMA copy helper.
- Thread one boolean through MegaMoE runtime code generation.
- Apply the hint only to routed expert-weight demand copies.
- Document the opt-in environment variable.

No routing, scheduler, tiling, cluster, synchronization, numerical, or public
Python API behavior changes are included.

## Testing

- `git diff --check`
- `python -m py_compile tests/test_mega_moe.py`
- Focused SM100 A/B benchmark and correctness run described above
- Existing default path remains `EVICT_NORMAL` when the variable is unset or
  set to `0`

The source and Python checks pass in the development environment. The final
GPU benchmark must be rerun on an idle SM100 node when CI or reviewer hardware
is available.

## Known Limitations

- This is an experimental, opt-in cache-policy hint and is not enabled by
  default.
- The measured benefit is workload-dependent and decreases with larger token
  counts; 1024 and 2048-token screens showed small regressions.
- Speculative previous-active-expert prefetch is deliberately excluded from
  this PR and remains preserved in the experiment archive for a separate,
  routing-aware proposal.
