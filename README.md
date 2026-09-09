# AsyCore

Implementation of **AsyCore: Dynamic and Parameter-Efficient LoRA Expert
Fusion via Asymmetric Core Factorization**.

AsyCore projects LoRA experts into a unified core space defined by fixed
reference bases. Guided by the observed asymmetry between the two low-rank
factors, it retains expert-specific left core factors, shares the right core
factor, and learns input-dependent expert weights at the module level. Fusion
training updates the routing parameters and low-dimensional core factors while
keeping the high-dimensional reference bases fixed.

## Implementation

The source tree provides AsyCore, its factor-sharing and routing variants, and
the fusion baselines used in the study: Average Merge, CAT, LoRA-Flow,
LoRA-All, and CoMoL. CAT corresponds to layer-wise weighting of independently
trained LoRA experts. The integration patches add multi-adapter setup,
method-specific forward functions, optimizer groups, and fusion-state
serialization to LLaMA-Factory and PEFT.

| Family | Method | Core structure or routing unit |
|---|---|---|
| Static fusion | Average Merge | Equal-weight sum of the two expert updates |
| Static fusion | CAT | One trainable expert-weight vector per Transformer layer |
| Dynamic routing | LoRA-Flow | Token-dependent weights shared by the modules in one layer |
| Dynamic routing | LoRA-All | Independent token-dependent weights for every LoRA module |
| Expert reconstruction | CoMoL | Trainable high-dimensional shared bases and expert cores |
| Asymmetric core | AsyCore | Fixed reference bases, expert-specific left cores, one shared right core, and module-wise routing |

The exact configuration switches for the main method and all structural
variants reported in the paper are listed in [docs/METHODS.md](docs/METHODS.md).

## Upstream baseline

The patches target LLaMA-Factory commit
`ea31c43d806162a7fd98065abfef2d974fff5766`.

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout ea31c43d806162a7fd98065abfef2d974fff5766
```

See [docs/INSTALL.md](docs/INSTALL.md) for installation, patching, and the
training-to-evaluation workflow. The Chinese and Russian fusion-training sets
used in the paper are provided under `data/`; dataset and expert provenance is
described in [docs/DATA.md](docs/DATA.md).

## Citation

Please cite the AsyCore paper when using this implementation. Bibliographic
metadata is provided in [CITATION.cff](CITATION.cff).

## License

The code is released under the Apache License 2.0. The integration patches are
based on Apache-2.0-licensed LLaMA-Factory files; see [NOTICE](NOTICE).
