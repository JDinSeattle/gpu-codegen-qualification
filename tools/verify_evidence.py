#!/usr/bin/env python3
"""Verify coverage in each distinct evidence layer and checksum retained files."""
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
    if not (hashlib.sha256((evidence / relative).read_bytes()).hexdigest() == sha):
        raise ValueError(relative)
offline = json.loads((evidence / "offline-20260906-complete/summary.json").read_text())
if not (offline['status'] == 'passed' and (not offline['driver_initialized'])):
    raise ValueError("evidence contract failed: offline['status'] == 'passed' and (not offline['driver_initialized'])")
if not (len(offline['configurations']) == 24):
    raise ValueError("evidence contract failed: len(offline['configurations']) == 24")
if not ({(c['dtype'], c['block'], c['warps']) for c in offline['configurations']} == set(itertools.product(['fp16', 'fp32'], [1, 32, 64, 128, 1024, 4096], [4, 8]))):
    raise ValueError("evidence contract failed: {(c['dtype'], c['block'], c['warps']) for c in offline['configurations']} == set(itertools.product(['fp16', 'fp32'], [1, 32, 64, 128, 1024, 4096], [4, 8]))")
for c in offline["configurations"]:
    path = evidence / "offline-20260906-complete" / f"{c['dtype']}-b{c['block']}-w{c['warps']}"
    for name, sha in c["hashes"].items():
        if not (hashlib.sha256((path / name).read_bytes()).hexdigest() == sha):
            raise ValueError('evidence contract failed: hashlib.sha256((path / name).read_bytes()).hexdigest() == sha')
expected = {(r, c, d, k, w) for (r, c), d, k, w in itertools.product(
    [(1,1),(7,31),(7,32),(7,33),(33,127),(64,1024),(128,4096)],
    ["float16", "float32"], ["normal", "zero", "cancellation", "all_nan", "mixed_inf"], [4, 8])}
for layer, folder in (("interpreter", "interpreter-20260906"), ("gpu", "gpu-final-20260906")):
    report = json.loads((evidence / folder / "summary.json").read_text())
    if not (report['status'] == 'passed' and report['layer'] == layer):
        raise ValueError("evidence contract failed: report['status'] == 'passed' and report['layer'] == layer")
    if not (report['compilation_bypassed'] == (layer == 'interpreter')):
        raise ValueError("evidence contract failed: report['compilation_bypassed'] == (layer == 'interpreter')")
    if not ({(c['rows'], c['cols'], c['dtype'], c['kind'], c['warps']) for c in report['cases']} == expected):
        raise ValueError("evidence contract failed: {(c['rows'], c['cols'], c['dtype'], c['kind'], c['warps']) for c in report['cases']} == expected")
    if not (len(report['cases']) == 140):
        raise ValueError("evidence contract failed: len(report['cases']) == 140")
    if not (all((c['repetitions'] == 3 and c['max_error_over_bound'] <= 1 and c['input_unchanged'] and c['output_canaries_unchanged'] for c in report['cases']))):
        raise ValueError("evidence contract failed: all((c['repetitions'] == 3 and c['max_error_over_bound'] <= 1 and c['input_unchanged'] and c['output_canaries_unchanged'] for c in report['cases']))")
    if layer == "gpu":
        if not (report['capability'] == [8, 9]):
            raise ValueError("evidence contract failed: report['capability'] == [8, 9]")
        if not (len(report['benchmark']['workloads']) == 7):
            raise ValueError("evidence contract failed: len(report['benchmark']['workloads']) == 7")
for name in ("memcheck", "racecheck", "synccheck"):
    log = (evidence / "sanitizers" / f"{name}.log").read_text()
    if name == "racecheck":
        if not ('0 hazards' in log and '0 errors' in log):
            raise ValueError("evidence contract failed: '0 hazards' in log and '0 errors' in log")
    else:
        if not ('ERROR SUMMARY: 0 errors' in log):
            raise ValueError("evidence contract failed: 'ERROR SUMMARY: 0 errors' in log")
    report = json.loads((evidence / "sanitizers" / name / "summary.json").read_text())
    if not (report['status'] == 'passed' and len(report['cases']) == 140):
        raise ValueError("evidence contract failed: report['status'] == 'passed' and len(report['cases']) == 140")
print("Verified: 24 offline configurations; interpreter and sm_89 hardware 140 cases ×3 each; three sanitizer runs")
