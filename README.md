# our method — Minimal Reproduction & Tests (Appendix-focused)

This repository contains a compact, runnable skeleton and tests to help you reproduce
the appendix experiments and validate core algorithmic pieces (MinHash, affinity channels,
RMT sparsification, fairness-aware clustering, prototype contrastive loss, etc.).

## Quickstart

1. Create a python environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# For GPU, install a CUDA-compatible torch build as appropriate.
```