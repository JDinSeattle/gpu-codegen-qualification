# Checkout validation

This local validation is separate from the user-confirmed cloud results in the README.

18 tests passed; new cloud interpreter profile passed 140 identities x 3 executions; 24 sm_89 offline compilations and 5 IR verifier cases plus invalid-block rejection passed with no driver initialization. python -O historical evidence verifier passed.

## Commands

```text
/home/postedism/Desktop/Find work/github-alignment/repos/gpu-codegen-qualification/.venv/bin/python -m pytest -q -ra
```

```text
PYTHONPATH=src TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m gpu_qualification.execute --mode interpreter --profile cloud --output .work/cloud-interpreter
```

```text
PYTHONPATH=src TRITON_INTERPRET=0 CUDA_VISIBLE_DEVICES='' TRITON_CACHE_DIR=.work/compile-cache .venv/bin/python -m gpu_qualification.offline --output .work/cloud-offline
```

Retained evidence check: `python3 -O tools/verify_evidence.py` passed.

No new physical-GPU performance, cloud rerun, or production availability claim is made. Local build and runtime artifacts remain outside Git; the committed source and profiles reproduce the checks with the pinned dependencies. Historical source changes already present in the user’s working tree are preserved in this update.
