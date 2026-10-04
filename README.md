# GPU Codegen Qualification

A Triton masked row-reduction qualification with separate interpreter semantics, explicit sm_89 compilation, and physical-GPU evidence. Fresh profiles preserve historical artifacts while exposing boundary lengths and rank-2 invalid-axis diagnostics.

## Confirmed qualification results

These are the user-confirmed results from a separate cloud test run, recorded in the experience bank. The device, workload, timing round, and counting boundaries below remain part of each result. They are distinct from the CPU checks performed in this checkout; cloud-hosted testing does not imply production deployment.

- Built a three-layer qualification for one Triton masked row-reduction family (fp16/fp32 inputs, fp32 accumulation, one program per row): 140 interpreter identities — 7 lengths 1/31/32/33/255/1024/4097 × 2 dtypes × 5 data/layout modes, each repeated 3 times for 420 executions, not 420 independent cases — plus 24 explicit sm_89 compile configurations and retained historical RTX 4090 evidence.

- Checked interpreter semantics against an independent CPU FP64 row-wise sum with masked padding filled as NaN and output canaries unchanged before and after; all 24 compile configurations produced a complete TTIR/TTGIR/LLVM IR/PTX/cubin chain with no CUDA device discovery before process exit, and the current CPU-only maintenance runs 14 pytest cases.

- Localized invalid-axis failures to standalone TTIR: a rank-2 tensor with axis=2 is rejected by the verifier, the legal sibling axis=1 compiles all the way to cubin, and negative-axis and invalid-block-shape cases produce explicit diagnostics.

- Kept the historical GPU performance result mixed and scoped: under the same historical-style pairing protocol, 1024 columns measured 4 warp at 6.2 μs versus 8 warp at 6.8 μs, while 4097 columns measured 4 warp at 11.9 μs versus 8 warp at 11.4 μs, so warp count trades off per workload; timings include 100 replays per graph across 30 pairs and exclude compilation, H2D transfer and Python launch.

- Tied timing artifacts to the exact cubin that was loaded, recording compile parameters, input shape and graph capture order, and retained resource and disassembly records only as constraints on comparison identity, not as proof that a particular instruction arrangement caused a speedup.

## Implementation and reproduction

| Contract | Implementation |
|---|---|
| Cloud and historical input profiles | [src/gpu_qualification/cases.py](src/gpu_qualification/cases.py) |
| Interpreter and hardware runner | [src/gpu_qualification/execute.py](src/gpu_qualification/execute.py) |
| Rank-2 invalid axis and legal sibling | [kernels](kernels) |
| Optimization-safe historical evidence replay | [tools/verify_evidence.py](tools/verify_evidence.py) |

Run each experiment into a fresh output directory to preserve earlier evidence.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q
PYTHONPATH=src TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m gpu_qualification.execute --mode interpreter \
  --profile cloud --output .work/cloud-interpreter
PYTHONPATH=src TRITON_INTERPRET=0 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m gpu_qualification.offline --output .work/offline
python3 -O tools/verify_evidence.py
```

Regression entry points: [tests/test_oracle.py](tests/test_oracle.py), [tests/test_ir.py](tests/test_ir.py), [tests/test_evidence_rejection.py](tests/test_evidence_rejection.py).

## Scope and evidence

The default cloud profile includes 255 and 4097 columns with NaN padding; --profile historical keeps the old 127/4096 cohort available. Retained GPU artifacts are not replaced by interpreter results.

- The current update reruns only the CPU interpreter and explicit sm_89 compile layers; RTX 4090 numerical, graph-timing and sanitizer evidence is historical and is not a GPU rerun in this round.

- A CPU interpreter pass does not verify thread synchronization, hardware memory access, code generation or resource limits, and sanitizer diagnostics and real-device numerics are independent layers that cannot substitute for each other.

- The 4/8-warp outcomes are mixed — 4 warp faster at 1024 columns and 8 warp slightly faster at 4097 — so no general performance win is established; the pairing numbers are an analysis example under the historical-style protocol.

- No H100/Blackwell/Rubin or arbitrary-layout guarantee; TMA and WGMMA cannot be claimed inside this sm_89 qualification scope, and Hopper features are not mixed into the retained code chain.

- The standalone checksum verifier still uses assertions and runs only without -O; the Bounded Pass hardening is not ported here and must not be borrowed to claim this project was fixed.

- A cubin digest alone is insufficient: it must come from the exact binary used for timing, together with compile parameters, input shape and graph capture order; resource and disassembly records constrain comparison identity but cannot prove a specific instruction arrangement caused the speedup.

- Graph timings exclude compilation, H2D transfer and Python launch; the 140 identities are repeated 3 times (420 executions) and must not be described as 420 independent cases.

The [previous README](README.historical.md) preserves earlier setup details, design discussion, and historical measurements. Its older counts, splits, versions, and timing cohorts must not be mixed with the confirmed round above. [Result provenance](docs/experience-bank-results.json) retains the confirmed bullet text; [checkout validation](docs/checkout-validation.md) records what was actually rerun here.
