"""Evidence admission must still reject a rehashed false summary under -O."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("optimized", [False, True])
def test_rehashed_failure_summary_is_rejected(tmp_path, optimized):
    evidence = tmp_path / "evidence"
    shutil.copytree(ROOT / "evidence", evidence, ignore=shutil.ignore_patterns("maintenance-*"))
    summary = evidence / "offline-20260906-complete/summary.json"
    argument = ["--root", str(tmp_path)]
    report = json.loads(summary.read_text())
    report["status"] = "failed"
    summary.write_text(json.dumps(report))
    checksums_path = evidence / "checksums.json"
    checksums = json.loads(checksums_path.read_text())
    checksums[str(summary.relative_to(evidence))] = hashlib.sha256(summary.read_bytes()).hexdigest()
    checksums_path.write_text(json.dumps(checksums))
    command = [sys.executable] + (["-O"] if optimized else [])
    process = subprocess.run(command + [str(ROOT / "tools/verify_evidence.py"), *argument],
                             capture_output=True, text=True, timeout=30)
    assert process.returncode != 0
    assert "evidence contract failed" in process.stderr
