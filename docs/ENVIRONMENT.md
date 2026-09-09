# Verified environment

The implementation was verified with the following software versions:

| Component | Version |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 5.7.0 |
| PEFT | 0.18.1 |
| Accelerate | 1.11.0 |
| Datasets | 4.0.0 |
| TRL | 0.24.0 |
| LLaMA-Factory | `ea31c43d806162a7fd98065abfef2d974fff5766` |

The paper experiments used two NVIDIA GeForce RTX 4090 GPUs. With a per-device
training batch size of 4 and 8 gradient-accumulation steps, the effective
global batch size is `2 × 4 × 8 = 64`.

Hardware-specific performance is not guaranteed. Model and adapter licenses
must be accepted separately from this repository.
