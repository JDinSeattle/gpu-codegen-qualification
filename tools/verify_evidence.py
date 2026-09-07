#!/usr/bin/env python3
"""Assert coverage in each distinct evidence layer and checksum retained files."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
args = parser.parse_args()
root = args.root
evidence = root / "evidence"
for relative, sha in json.loads((evidence / "checksums.json").read_text()).items():
    assert hashlib.sha256((evidence / relative).read_bytes()).hexdigest() == sha, relative
offline = json.loads((evidence / "offline-20260906-complete/summary.json").read_text())
assert offline["status"] == "passed" and not offline["driver_initialized"]
assert len(offline["configurations"]) == 24
assert {(c["dtype"], c["block"], c["warps"]) for c in offline["configurations"]} == set(
    itertools.product(["fp16", "fp32"], [1, 32, 64, 128, 1024, 4096], [4, 8]))
for c in offline["configurations"]:
    path = evidence / "offline-20260906-complete" / f"{c['dtype']}-b{c['block']}-w{c['warps']}"
    for name, sha in c["hashes"].items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == sha
expected = {(r, c, d, k, w) for (r, c), d, k, w in itertools.product(
    [(1,1),(7,31),(7,32),(7,33),(33,127),(64,1024),(128,4096)],
    ["float16", "float32"], ["normal", "zero", "cancellation", "all_nan", "mixed_inf"], [4, 8])}
for layer, folder in (("interpreter", "interpreter-20260906"), ("gpu", "gpu-final-20260906")):
    report = json.loads((evidence / folder / "summary.json").read_text())
    assert report["status"] == "passed" and report["layer"] == layer
    assert report["compilation_bypassed"] == (layer == "interpreter")
    assert {(c["rows"],c["cols"],c["dtype"],c["kind"],c["warps"]) for c in report["cases"]} == expected
    assert len(report["cases"]) == 140
    assert all(c["repetitions"] == 3 and c["max_error_over_bound"] <= 1 and
               c["input_unchanged"] and c["output_canaries_unchanged"] for c in report["cases"])
    if layer == "gpu":
        assert report["capability"] == [8, 9]
        assert len(report["benchmark"]["workloads"]) == 7
for name in ("memcheck", "racecheck", "synccheck"):
    log = (evidence / "sanitizers" / f"{name}.log").read_text()
    if name == "racecheck":
        assert "0 hazards" in log and "0 errors" in log
    else:
        assert "ERROR SUMMARY: 0 errors" in log
    report = json.loads((evidence / "sanitizers" / name / "summary.json").read_text())
    assert report["status"] == "passed" and len(report["cases"]) == 140
print("Verified: 24 offline configurations; interpreter and sm_89 hardware 140 cases ×3 each; three sanitizer runs")
