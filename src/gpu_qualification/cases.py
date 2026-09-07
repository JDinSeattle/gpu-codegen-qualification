"""Shared seeded operands and oracle for CPU semantics and actual GPU execution."""
import numpy as np

SHAPES = [(1, 1), (7, 31), (7, 32), (7, 33), (33, 127), (64, 1024), (128, 4096)]
KINDS = ("normal", "zero", "cancellation", "all_nan", "mixed_inf")


def make_input(rows, cols, dtype, kind, seed=20260906):
    rng = np.random.default_rng(seed + rows * 10000 + cols)
    # Row padding must never contribute to a valid row's masked reduction.
    storage = np.full((rows, cols + 7), 12345, dtype=dtype)
    x = storage[:, :cols]
    x[:] = rng.normal(size=(rows, cols)).astype(dtype)
    if kind == "zero":
        x[:] = 0
    elif kind == "cancellation":
        x[:, ::2] = 16
        x[:, 1::2] = -16
        x[:, -1] = 0.5
    elif kind == "all_nan":
        x[:] = np.nan
    elif kind == "mixed_inf":
        x[:, 0] = np.inf
        if cols > 1:
            x[:, 1] = -np.inf
    elif kind != "normal":
        raise ValueError(kind)
    return storage, x


def compare(output, source):
    with np.errstate(invalid="ignore"):
        expected = source.astype(np.float64).sum(axis=1)
    if output.dtype != np.float32 or output.shape != expected.shape:
        raise AssertionError("wrong output shape/dtype")
    for mask in (np.isnan, np.isposinf, np.isneginf):
        np.testing.assert_array_equal(mask(output), mask(expected))
    finite = np.isfinite(expected)
    error = np.abs(output[finite].astype(np.float64) - expected[finite])
    # Absolute + relative criterion for the bounded input domain above.
    bound = 2e-4 + 2e-4 * np.abs(expected[finite])
    if np.any(error > bound):
        raise AssertionError(f"reduction error exceeds contract: max {error.max()}")
    return {"max_abs_error": float(error.max(initial=0)),
            "max_error_over_bound": float((error / bound).max(initial=0)),
            "nan_count": int(np.isnan(expected).sum())}
