#!/usr/bin/env python3
"""Isolated real Triton compiler invocation for a fixed sm_89 target."""
import argparse
import json
import triton
from triton.backends.compiler import GPUTarget
from triton.runtime import driver

parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()
try:
    kernel = triton.compile(args.path, target=GPUTarget("cuda", 89, 32), options={"num_warps": 4})
except Exception as exc:
    print(json.dumps({"passed": False, "exception": type(exc).__name__, "message": str(exc),
                      "driver_initialized": driver._active is not None}))
    raise SystemExit(1)
assert driver._active is None
print(json.dumps({"passed": True, "stages": list(kernel.asm), "driver_initialized": False}))
