"""Explicit-target compilation without querying a GPU driver."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.runtime import driver
from .kernel import row_sum

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("output must be empty")
    assert driver._active is None
    records = []
    for dtype in ("fp16", "fp32"):
        for block in (1, 32, 64, 128, 1024, 4096):
            for warps in (4, 8):
                source = ASTSource(row_sum, {"X": f"*{dtype}", "Y": "*fp32", "ROW_STRIDE": "i32",
                                            "N_COLS": "i32", "OUT_STRIDE": "i32"}, constexprs={"BLOCK": block})
                start = time.perf_counter_ns()
                kernel = triton.compile(source, target=GPUTarget("cuda", 89, 32), options={"num_warps": warps})
                elapsed = time.perf_counter_ns() - start
                assert driver._active is None, "offline compiler initialized a driver"
                location = output / f"{dtype}-b{block}-w{warps}"
                location.mkdir()
                hashes = {}
                for ext, value in kernel.asm.items():
                    path = location / f"row_sum.{ext}"
                    data = value if isinstance(value, bytes) else value.encode()
                    path.write_bytes(data)
                    hashes[path.name] = hashlib.sha256(data).hexdigest()
                metadata = kernel.metadata._asdict()
                metadata["target"] = vars(kernel.metadata.target)
                (location / "metadata.json").write_text(json.dumps(metadata, indent=2))
                records.append({"dtype": dtype, "block": block, "warps": warps,
                                "compile_wall_ns": elapsed, "hashes": hashes,
                                "shared_bytes": kernel.metadata.shared})
    negatives = []
    for name, marker in (("reduce-axis-invalid.ttir", "axis out of bounds for operand rank 1"),
                         ("reduce-axis-negative.ttir", "axis"), ("reduce-valid.ttir", None)):
        command = [sys.executable, str(ROOT / "tools/compile_ir.py"), str(ROOT / "kernels" / name)]
        proc = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (output / f"{name}.stdout").write_text(proc.stdout)
        (output / f"{name}.stderr").write_text(proc.stderr)
        if marker:
            assert proc.returncode == 1 and marker in proc.stderr, proc.stderr
        else:
            assert proc.returncode == 0, proc.stderr
        negatives.append({"input": name, "command": command, "returncode": proc.returncode,
                          "expected_diagnostic": marker, "result": json.loads(proc.stdout)})
    # Non-power-of-two blocks fail at AST-to-TTIR semantic construction, a different stage.
    source = ASTSource(row_sum, {"X": "*fp32", "Y": "*fp32", "ROW_STRIDE": "i32",
                                "N_COLS": "i32", "OUT_STRIDE": "i32"}, constexprs={"BLOCK": 96})
    try:
        triton.compile(source, target=GPUTarget("cuda", 89, 32), options={"num_warps": 4})
        raise AssertionError("non-power-of-two block accepted")
    except triton.CompilationError as exc:
        message = str(exc)
        assert "power of 2" in message or "power of two" in message
        (output / "invalid-block.txt").write_text(message)
    native = Path(triton.__file__).parent / "_C/libtriton.so"
    report = {"layer": "offline compiler", "run_utc": datetime.now(timezone.utc).isoformat(),
              "triton": triton.__version__, "target": "cuda:89, warp_size=32", "driver_initialized": driver._active is not None,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "native_library_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
              "configurations": records, "ir_regressions": negatives, "status": "passed"}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS {len(records)} real offline compilations; driver untouched; 3 IR verifier cases + invalid block")


if __name__ == "__main__":
    main()
