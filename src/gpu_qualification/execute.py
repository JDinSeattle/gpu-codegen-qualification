"""Interpreter and real GPU validation are distinct modes and distinct reports."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
import numpy as np
import torch
import triton
from .cases import SHAPES, HISTORICAL_SHAPES, KINDS, make_input, compare
from .kernel import row_sum


def gpu_snapshot():
    proc = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,pstate,temperature.gpu,utilization.gpu,clocks.sm,clocks.mem,power.draw",
                           "--format=csv"], capture_output=True, text=True)
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def canaries(storage, rows):
    host = storage.detach().cpu().numpy()
    np.testing.assert_array_equal(host[::2], np.full(rows + 1, -9876.5, np.float32))


def run_correctness(device, output, shapes=SHAPES):
    records, compiled_keys = [], set()
    for rows, cols in shapes:
        for dtype in (np.float16, np.float32):
            for kind in KINDS:
                storage, reference_input = make_input(rows, cols, dtype, kind)
                x_storage = torch.from_numpy(storage).to(device)
                x = x_storage[:, :cols]
                for warps in (4, 8):
                    y_storage = torch.full((rows * 2 + 1,), -9876.5, dtype=torch.float32, device=device)
                    for repetition in range(3):
                        compiled = row_sum[(rows,)](x, y_storage[1:], x.stride(0), cols, 2,
                                                   BLOCK=triton.next_power_of_2(cols), num_warps=warps)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        result = y_storage[1::2].detach().cpu().numpy()
                        numeric = compare(result, reference_input)
                        canaries(y_storage, rows)
                    np.testing.assert_array_equal(x_storage.detach().cpu().numpy(), storage)
                    key = f"{np.dtype(dtype).name}-n{cols}-w{warps}"
                    if device == "cuda" and key not in compiled_keys:
                        location = output / "executed-code" / key
                        location.mkdir(parents=True)
                        for ext, value in compiled.asm.items():
                            path = location / f"row_sum.{ext}"
                            path.write_bytes(value) if isinstance(value, bytes) else path.write_text(value)
                        (location / "launch.json").write_text(json.dumps({
                            "name": compiled.metadata.name, "hash": compiled.hash,
                            "num_warps": warps, "shared_bytes": compiled.metadata.shared,
                            "registers_per_thread": compiled.n_regs, "spills": compiled.n_spills,
                            "arch": compiled.metadata.arch,
                        }, indent=2))
                        compiled_keys.add(key)
                    records.append({"rows": rows, "cols": cols, "dtype": np.dtype(dtype).name,
                                    "kind": kind, "warps": warps, "repetitions": 3,
                                    "output_canaries_unchanged": True, "input_unchanged": True, **numeric})
    return records


def benchmark(samples, launches, output, shapes=SHAPES):
    measurements = []
    rng = random.Random(20260906)
    for rows, cols in shapes:
        storage, source = make_input(rows, cols, np.float32, "normal")
        x_storage = torch.from_numpy(storage).cuda()
        x = x_storage[:, :cols]
        outputs = {name: torch.empty(rows, device="cuda", dtype=torch.float32)
                   for name in ("warps4", "warps8", "torch")}
        def call(name):
            if name == "torch":
                torch.sum(x, dim=1, dtype=torch.float32, out=outputs[name])
            else:
                return row_sum[(rows,)](x, outputs[name], x.stride(0), cols, 1,
                                        BLOCK=triton.next_power_of_2(cols), num_warps=int(name[-1]))
        graphs = {}
        code_hashes = {}
        for name in outputs:
            for _ in range(10):
                compiled = call(name)
            if name != "torch":
                location = output / "benchmark-code" / f"r{rows}-n{cols}-{name}"
                location.mkdir(parents=True)
                for ext, value in compiled.asm.items():
                    path = location / f"row_sum.{ext}"
                    path.write_bytes(value) if isinstance(value, bytes) else path.write_text(value)
                code_hashes[name] = {"kernel_hash": compiled.hash,
                                     "cubin_sha256": hashlib.sha256(compiled.asm["cubin"]).hexdigest(),
                                     "registers_per_thread": compiled.n_regs, "shared_bytes": compiled.metadata.shared}
            torch.cuda.synchronize()
            compare(outputs[name].cpu().numpy(), source)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(launches):
                    call(name)
            graphs[name] = graph
        # Pre-initialize graph uploads and event creation outside recorded samples.
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                  for _ in range(samples * len(graphs))]
        times, orders = {name: [] for name in outputs}, []
        idx = 0
        before = gpu_snapshot()
        for _ in range(samples):
            order = list(outputs)
            rng.shuffle(order)
            orders.append(order)
            for name in order:
                start, stop = events[idx]
                idx += 1
                start.record()
                graphs[name].replay()
                stop.record()
                stop.synchronize()
                times[name].append(start.elapsed_time(stop) * 1000 / launches)
        for name in outputs:
            compare(outputs[name].cpu().numpy(), source)
        measurements.append({"rows": rows, "cols": cols, "dtype": "float32",
                             "latency_us": times, "orders": orders,
                             "median_us": {name: statistics.median(v) for name, v in times.items()},
                             "measured_code": code_hashes,
                             "gpu_before": before, "gpu_after": gpu_snapshot()})
    return measurements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["interpreter", "gpu"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["cloud", "historical"], default="cloud")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--launches", type=int, default=32)
    args = parser.parse_args()
    shapes = SHAPES if args.profile == "cloud" else HISTORICAL_SHAPES
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("output must be empty")
    if args.mode == "interpreter":
        if os.environ.get("TRITON_INTERPRET") != "1":
            parser.error("interpreter requires TRITON_INTERPRET=1 before importing kernels")
        if args.benchmark:
            parser.error("interpreter results are not GPU performance evidence")
        device = "cpu"
    else:
        if os.environ.get("TRITON_INTERPRET", "0") != "0":
            parser.error("GPU mode requires compilation (TRITON_INTERPRET=0)")
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
            parser.error("hardware qualification requires an sm_89 CUDA device")
        device = "cuda"
    report = {"profile": args.profile, "layer": args.mode, "run_utc": datetime.now(timezone.utc).isoformat(),
              "torch": torch.__version__, "triton": triton.__version__, "torch_cuda": torch.version.cuda,
              "compilation_bypassed": args.mode == "interpreter", "seed": 20260906,
              "tolerances": {"atol": 2e-4, "rtol": 2e-4, "reference": "float64 sum of input values"},
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path(__file__).parent.glob("*.py")}}
    if device == "cuda":
        report["gpu"] = gpu_snapshot()
        report["device"] = torch.cuda.get_device_name()
        report["capability"] = list(torch.cuda.get_device_capability())
    report["cases"] = run_correctness(device, output, shapes)
    (output / "correctness.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.benchmark:
        report["benchmark"] = {"metric": "CUDA-event graph time per launch; fixed operands and outputs; excludes compilation and transfers",
                               "samples": args.samples, "launches_per_graph": args.launches,
                               "clock_controlled": False, "shared_gpu": True,
                               "workloads": benchmark(args.samples, args.launches, output, shapes)}
    report["status"] = "passed"
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS layer={args.mode}; {len(report['cases'])} cases × 3; benchmark={args.benchmark}")


if __name__ == "__main__":
    main()
