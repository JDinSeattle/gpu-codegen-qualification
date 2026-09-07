import numpy as np
import pytest
from gpu_qualification.cases import compare, make_input


@pytest.mark.parametrize("rows,cols", [(1, 1), (7, 31), (7, 33), (128, 4096)])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_input_padding_and_independent_reference(rows, cols, dtype):
    storage, source = make_input(rows, cols, dtype, "zero")
    assert source.strides[0] // source.itemsize == cols + 7
    assert np.all(storage[:, cols:] != 0)
    compare(np.zeros(rows, np.float32), source)
    # A masked-load bug would include nonzero sentinel values.
    with pytest.raises(AssertionError):
        compare(storage.astype(np.float64).sum(axis=1).astype(np.float32), source)


@pytest.mark.parametrize("kind", ["all_nan", "mixed_inf"])
def test_wrong_special_values_are_rejected(kind):
    _, source = make_input(7, 33, np.float32, kind)
    with pytest.raises(AssertionError):
        compare(np.zeros(7, np.float32), source)


def test_wrong_finite_result_and_output_type_rejected():
    _, source = make_input(1, 32, np.float32, "zero")
    for actual in (np.ones(1, np.float32), np.zeros(2, np.float32), np.zeros(1, np.float64)):
        with pytest.raises(AssertionError):
            compare(actual, source)
