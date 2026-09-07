# Compiler-stage reasoning

The kernel computes one row sum per program. `tl.arange` creates a power-of-two
logical block; `col < N_COLS` guards padded lanes; loaded values convert to f32;
`tl.sum(axis=0)` reduces to a scalar; one store writes that row's result.

For the fixed TTIR regression, the operand of `tt.reduce` has rank 1. An axis
must be in `[0, rank)`. The invalid fixtures use 1 or -1; no tensor allocation,
JIT driver discovery, backend execution or undefined runtime memory access is
needed to reproduce the verifier diagnostic. The axis-0 control differs only in
the relevant attribute and compiles to actual target code. A separate `BLOCK=96`
case fails earlier, at the Python AST/Triton semantic construction step.

The offline suite supplies `GPUTarget('cuda',89,32)` explicitly and checks that
`triton.runtime.driver._active` remains unset before and after compilation.
`CUDA_VISIBLE_DEVICES=''` also hides devices. Every configuration retains its
TTIR, TTGIR, LLVM IR, PTX, cubin and resource metadata. The installed wheel's own
LLVM backend and bundled ptxas are used; local LLVM 21 or A's LLVM main build is
not mixed into this pipeline.

The release's `python/build_helpers.py` reads `cmake/llvm-info.json` for its
prebuilt dependency. At Triton release commit
`c01b6774b1865984607d89d89d3a10833de92037`, that file selects LLVM
`5f07f818b51b786b0a87b4e514882600ecba112f`, build 1. The distinct
`llvm-build-info.json` contains a different build metadata revision and is not
the JSON consumed by that download helper. Both observations are recorded to
avoid silently conflating them. The wheel's native-library SHA-256 identifies
the binary actually executed; this project does not attest a reproducible build
of the wheel from source.

## Layout choice and actual code

Changing `num_warps=4` to `8` changes the blocked layout's `warpsPerCTA`. On the
4096-column f32 correctness kernel, the measured JIT resource records and
`cuobjdump` output show:

| Resource/code observation | 4 warps | 8 warps |
|---|---:|---:|
| Registers per thread | 37 | 29 |
| Shared bytes | 16 | 32 |
| Spills | 0 | 0 |
| SASS instruction lines | 216 | 136 |
| BAR.SYNC count | 1 | 1 |
| SHFL count | 7 | 8 |

`evidence/codegen-comparison.json` identifies the exact correctness-test cubins.
Those kernels use canary-separated outputs. Timing uses contiguous output arrays;
its exact cubins are separately retained in `gpu-final-20260906/benchmark-code`
and linked by hashes in the timing records. We do not conflate these signatures.

The resource tradeoff is lower per-thread register usage versus more threads and
shared memory per CTA. Static instruction counts alone cannot determine latency
or occupancy. The paired timing cohort does not show a material benefit for
eight warps. This no-benefit outcome is retained instead of treating fewer SASS
instructions as proof of optimization.
