# GPU Codegen Qualification

A scoped Triton row-reduction qualification project that keeps **program semantics,
compiler validation, and real GPU execution** separate. Target: NVIDIA RTX 4090 / sm_89.

| Validation layer | Recorded result | What it establishes |
|---|---|---|
| CPU interpreter | 140 cases × 3 passed | Supported program semantics; compilation is bypassed |
| Explicit-target compiler | 24 configurations, no driver initialization | TTIR → TTGIR → LLVM IR → PTX → cubin |
| RTX 4090 execution | 140 cases × 3 passed | Numerical outputs, input immutability, output canaries |
| NVIDIA Compute Sanitizer | memcheck, racecheck, synccheck: zero errors/hazards | Memory/synchronization checks for these real launches |
| Unit/IR regressions | 14 tests passed | Oracle sensitivity and real fixed-IR verifier behavior |

The [measured report](evidence/REPORT.md) includes a `torch.sum` baseline, randomized paired
CUDA-event samples and exact hashes of timed cubins. Four versus eight warps changes the
generated code and resource use; **the measurements do not establish a performance benefit
from eight warps**. All samples and both configurations are retained.

## Reproduce

Python 3.12 on Linux x86-64. The full pinned PyTorch CUDA environment needs several GB.
The GPU step requires an sm_89 device; the offline compiler step does not.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q

# Layer 1: interpreter — no compilation, no GPU claim.
PYTHONPATH=src TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m gpu_qualification.execute \
  --mode interpreter --output .work/interpreter

# Layer 2: actual explicit-target compilation; asserts driver remains uninitialized.
PYTHONPATH=src TRITON_INTERPRET=0 CUDA_VISIBLE_DEVICES='' TRITON_CACHE_DIR=.work/offline-cache \
  .venv/bin/python -m gpu_qualification.offline --output .work/offline

# Layer 3: real RTX 4090 execution and paired graph timing.
PYTHONPATH=src TRITON_INTERPRET=0 TRITON_CACHE_DIR=.work/gpu-cache \
  .venv/bin/python -m gpu_qualification.execute \
  --mode gpu --output .work/gpu --benchmark

# Separate sanitizer invocations; require NVIDIA CUDA Toolkit Compute Sanitizer.
bash tools/sanitizers.sh .work/sanitizers

# Verify the retained hardware evidence without a GPU or dependencies.
python3 tools/verify_evidence.py
```

Output directories must be empty to preserve previous results. For CPU-only CI, install
`requirements-cpu.lock` and `torch==2.14.0` from PyTorch's CPU wheel index. The workflow runs
the first two layers and verifies retained hardware evidence; it **does not claim to rerun GPU tests**.

## Scope and compiler regression

Input dtypes: float16 and float32; accumulation/output: float32. Shapes cover columns
1, 31, 32, 33, 127, 1024, 4096 and 1–128 rows. Row padding is deliberately nonzero;
masked loads must exclude it. Output canaries surround and separate row results.
Normal/zero/cancellation/all-NaN/mixed-infinity inputs use independent float64 references,
absolute and relative error limits, and exact special-value classification checks.

[The minimal fixed IR](kernels/reduce-axis-invalid.ttir) selects axis 1 for a rank-1
`tt.reduce`. It fails in the real verifier with an axis-bounds diagnostic. The
[legal axis-0 sibling](kernels/reduce-valid.ttir) compiles all the way to cubin;
axis -1 and a non-power-of-two AST block are separate negative regressions.
These are qualification fixtures for expected invalid input, not an upstream bug-fix claim.

See [compiler analysis](docs/compiler-analysis.md), [measurement limits](docs/methodology.md)
and [interview notes](docs/interview.md). The installed Triton wheel is fixed at 3.8.0;
its release commit, release-selected LLVM revision and native library checksum are recorded
in [toolchain evidence](evidence/toolchain.json). System LLVM is not substituted into Triton.

This is one reduction family and one GPU architecture. No H100-specific instructions,
descriptor lowering, arbitrary strided-column layout, or cross-architecture qualification
is claimed. No upstream message or PR has been sent.
