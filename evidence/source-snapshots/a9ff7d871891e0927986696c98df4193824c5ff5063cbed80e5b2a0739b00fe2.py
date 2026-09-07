"""One scoped reduction, with output canaries and explicit row strides."""
import triton
import triton.language as tl


@triton.jit
def row_sum(X, Y, ROW_STRIDE, N_COLS, OUT_STRIDE, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    values = tl.load(X + row * ROW_STRIDE + col, mask=col < N_COLS, other=0).to(tl.float32)
    total = tl.sum(values, axis=0)
    # Output pointer is offset to leave canaries before, after and between rows.
    tl.store(Y + row * OUT_STRIDE, total)
