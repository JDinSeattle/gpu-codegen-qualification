# Measurement and validation limits

The CPU interpreter simulates supported operations through NumPy. It bypasses
the compiler and is explicitly labeled as program-semantic evidence. Actual
compiler tests use Triton's native IR verifier and target pipeline. Real hardware
tests execute JIT kernels on the RTX 4090 and compare outputs independently.

All numerical cases use input values rounded to their declared fp16/fp32 type,
then float64 accumulation for the reference. Finite tolerance is
`2e-4 + 2e-4*abs(reference)` for this bounded input domain. NaN and positive/negative
infinity masks must match. NaN payloads, exception flags, subnormal stress and
arbitrary magnitudes are outside the contract. Cancellation data are exactly
representable bounded values; they are not a comprehensive rounding oracle.

Each case checks nonzero row padding is excluded, output canaries are unchanged,
input storage is unchanged, and three invocations produce correct outputs.
Memcheck, racecheck and synccheck each run the complete hardware numerical set in
a separate process. Their zero-error results apply to these launches, shapes,
compiler versions and architecture, not every possible program or scheduling.

The benchmark uses f32 inputs only. It compares four-warps, eight-warps and
PyTorch `torch.sum` with f32 output, using the same padded input layout and
preallocated contiguous outputs. After eager warmups, a CUDA graph captures 32
launches of each variant. Thirty-one paired trials randomize variant order;
CUDA events measure graph device time divided by 32. Compilation, Python eager
launch overhead, graph capture and transfers are excluded. Each captured result
is numerically checked outside the timer. Small differences near event resolution
or normal GPU noise are not claimed significant.

GPU clocks are not locked and other user processes are not stopped. Per-workload
GPU temperature/utilization/clocks/power snapshots are retained. Shared-machine
measurements can vary; performance claims apply to this cohort. The baseline is
the installed PyTorch release, not a vendor guarantee or an exhaustive best kernel.

The final GPU cohort adds hashes and saved compiler stages for the exact timed
signatures. The earlier cohort and sanitizer runs are retained; the intervening
source edit only adds benchmark artifact recording, not kernel logic or the
non-benchmark correctness path. Full source snapshots accompany the evidence.
