## LlamaFactory (Vendored)

This directory vendors a minimal subset of the upstream LLaMA-Factory project
needed by this repository's CL/DATA training scripts.

Why this exists:
- The latest RoboBrain checkpoints require newer `transformers` versions.
- We run LLaMA-Factory from source via `PYTHONPATH=LlamaFactory/src` to avoid
  environment-level pinning issues.

Entry points:
- `config/train.sh` invokes `python -m llamafactory.cli ...` with
  `PYTHONPATH=LlamaFactory/src`.

Upstream:
- https://github.com/hiyouga/LLaMA-Factory

