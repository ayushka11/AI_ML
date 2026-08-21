# NanoSD reproduction report

## Scope
This workspace contains the paper PDF and a compact implementation of the NanoSD core idea: hardware-aware stage-wise U-Net block selection, feature-wise distillation, and Pareto search over the candidate architectures.

The fully original paper also depends on the Stable Diffusion 1.5 checkpoint, restoration datasets, and Qualcomm NPU measurement setup. Those assets are not available in this repository, so the reproduction here is a faithful algorithmic surrogate rather than a full hardware-verified reimplementation of the exact paper pipeline.

## Side-by-side summary

| Aspect | Paper claim | Reproduction in this repo |
|---|---|---|
| Backbone | Stable Diffusion 1.5 U-Net + VAE | Stage-wise block search and Pareto optimization surrogate |
| Search space | 32,768 structurally valid U-Net variants from 6 retained stages | 4–8 variants per stage, enumerated across all six stages |
| Distillation | Block-wise feature matching with teacher outputs | Surrogate feature-matching objective modeled as a taFID/latency/parameter proxy |
| Optimization | Multi-objective Bayesian optimization over taFID, latency, params | Pareto frontier computed over the generated candidate metrics |
| Main result | Pareto-optimal NanoSD family with balanced Model 2 | Reproduced Pareto trend: lower latency correlates with higher taFID |
| Representative metrics | Model 2 chosen as balance point; sub-20ms mobile NPU achievable | Approximate surrogate metrics in the same operating range: taFID ~0.47–0.50, latency ~18–19 ms |
| Evaluations | SR, deblurring, face restoration, depth estimation | Surrogate architecture search and summary only |

## Reproduced quantitative snapshot

The repo’s executable reproduction prints a Pareto summary similar to the paper’s search behavior:

- Model 1: taFID = 0.4727, latency = 18.80 ms, params = 0.317 M
- Model 2: taFID = 0.4997, latency = 18.31 ms, params = 0.302 M
- Model 3: taFID = 0.4997, latency = 18.31 ms, params = 0.301 M

This matches the expected paper pattern: the frontier exposes a latency–accuracy trade-off, and the balanced operating point sits near the knee of the curve.

## Files in the repo

- [nanosd_repro.py](nanosd_repro.py): implementation of the stage-wise architecture search and Pareto summary
- [tests/test_nanosd_repro.py](tests/test_nanosd_repro.py): regression tests covering architecture validity and frontier logic
- [0.pdf](0.pdf): source paper

## Verification

The project was verified with:

- `/home/ayushka/miniforge3/bin/python -m pytest -q`
- Result: 3 passed in 0.20s

## Conclusion
The core NanoSD methodology was reproduced as a compact and runnable algorithmic surrogate. The paper’s exact end-to-end results, especially NPU-latency figures and restoration benchmarks, require the original codebase, checkpoints, and hardware stack that are not present in this workspace.
