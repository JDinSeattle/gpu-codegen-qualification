# GPU code generation and execution report

Target: NVIDIA RTX 4090 / sm_89; Triton 3.8.0; PyTorch 2.14.0+cu130; driver 595.84.

The interpreter and GPU layers each passed 140 cases × 3 repetitions. Offline compilation passed 24 configurations without driver initialization. Memcheck, racecheck and synccheck report zero errors/hazards.

## Device timing

CUDA events time a captured graph of 32 launches, divided by 32. Each configuration has 31 randomized paired samples. Inputs/outputs are preallocated; compilation, host transfers, Python eager dispatch and graph capture are excluded. The GPU is shared and clocks are not locked.

| Rows | Cols | Warps 4 µs | Warps 8 µs | torch.sum µs | Torch / warps 4 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.896 | 0.918 | 1.376 | 1.536× |
| 7 | 31 | 0.960 | 0.960 | 1.504 | 1.567× |
| 7 | 32 | 0.960 | 0.864 | 1.429 | 1.489× |
| 7 | 33 | 0.928 | 0.928 | 1.408 | 1.517× |
| 33 | 127 | 0.960 | 0.982 | 1.664 | 1.733× |
| 64 | 1024 | 1.118 | 1.056 | 2.836 | 2.537× |
| 128 | 4096 | 1.600 | 1.599 | 6.080 | 3.800× |

Geometric mean Torch / warps-4 ratio: 1.904×.

This cohort does not establish that eight warps improve application performance. Resource/code changes must be assessed separately from timing; small differences are not treated as significant. Exact timed cubins are saved under gpu-final-20260906/benchmark-code and linked by SHA-256 in each workload record.

## Compiler-stage regression

kernels/reduce-axis-invalid.ttir selects axis 1 for a rank-1 operand. The actual Triton parser/verifier rejects it with `axis out of bounds for operand rank 1`; axis -1 is rejected too. The otherwise identical axis-0 input compiles to cubin without initializing a GPU driver. A BLOCK=96 kernel is independently rejected during AST-to-TTIR construction because the range must have a power-of-two size. These are expected-invalid regression fixtures, not claims of newly discovered compiler bugs.

## Evidence layers

- offline-20260906-complete/: compiler IR, PTX, cubins, metadata and exact invalid-input diagnostics.
- interpreter-20260906/: CPU interpreter numerical evidence, explicitly bypassing compilation.
- gpu-final-20260906/: real sm_89 numerical tests, runtime-generated code and paired samples.
- sanitizers/: three independent real GPU runs with NVIDIA diagnostic logs.
- codegen-comparison.json: SASS/resources for four versus eight warps in the canary-output correctness kernel.
- toolchain.json: release commit, release-selected LLVM pin, native binary identity and tool versions.
- gpu-20260906/: earlier timing cohort retained, before exact timed-code capture was added.

Numerical scope: fp16/fp32 inputs, fp32 accumulation/output, fixed seeded bounded inputs, float64 reference, atol=rtol=2e-4, matching NaN and ±Inf masks. Input row padding and output canaries are checked independently. A passing sanitizer run covers these launches; it is not a proof for all inputs or GPU architectures.
