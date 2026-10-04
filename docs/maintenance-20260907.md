# Compiler layer refresh — 2026-09-07

Current source and pinned packages were inspected. This update reran 14 pytest regressions,
140 CPU-interpreter cases with three repetitions each, and 24 actual explicit-target
Triton compilations to sm_89 cubin without GPU driver initialization. All commands passed;
see [validation receipt](../evidence/maintenance-20260907/validation.json),
[interpreter results](../evidence/maintenance-20260907/interpreter/summary.json) and
[offline compiler results](../evidence/maintenance-20260907/offline/summary.json).

The retained RTX 4090 numerical tests, paired CUDA-graph timings, exact cubins/SASS and
memcheck/racecheck/synccheck logs passed the existing integrity checker. They were originally
executed on 2026-09-06 America/Los_Angeles and were **not rerun on hardware in this update**.
The interpreter bypasses compilation; offline code generation establishes compilation,
not execution or performance on hardware. No simulated GPU is involved in the refreshed
CPU layers. Keep the original mixed 4/8-warp conclusion and numerical scope.

[Triton 3.8.0](https://github.com/triton-lang/triton/releases/tag/v3.8.0), released 2026-08-28,
is already pinned with the release-selected LLVM revision and PyTorch 2.14.0 wheel. Its
newer architecture/tooling features do not justify claiming sm_90/Blackwell/Rubin coverage
from a reduction qualified on sm_89. No upgrade or extra framework was required.

The portfolio [market review](../../D-bounded-pass-selection/docs/market-review-20260907.md)
prioritized data/model lineage reliability in D. C retains its compiler-stage boundaries
and independent numerical reference. C implementation is unchanged; this update adds local
validation/documentation and evidence-bank entries. No fresh hosted CI, cloud/production
execution or upstream acceptance is claimed.

The older standalone C checksum verifier still relies on normal Python assertions; the
optimization-safe semantic validator added to D has not been ported to C. This refresh uses
C's verifier without `-O`. Future work should separately qualify any expanded runtime or
GPU architecture and reconsider standalone-verifier hardening.
