"""Real offline compiler regressions, not text-only checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name,code,diagnostic", [
    ("reduce-valid.ttir", 0, None),
    ("reduce-axis-invalid.ttir", 1, "axis out of bounds for operand rank 1"),
    ("reduce-axis-negative.ttir", 1, "axis"),
])
def test_fixed_ir_verifier(name, code, diagnostic, tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TRITON_CACHE_DIR=str(tmp_path))
    proc = subprocess.run([sys.executable, str(ROOT / "tools/compile_ir.py"), str(ROOT / "kernels" / name)],
                          env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == code
    result = json.loads(proc.stdout)
    assert not result["driver_initialized"]
    if diagnostic:
        assert diagnostic in proc.stderr
    else:
        assert "cubin" in result["stages"]
