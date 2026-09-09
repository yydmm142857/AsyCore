# Fusion-training datasets

This directory contains the Chinese and Russian fusion-training sets used in
the AsyCore experiments. Each file has 1,000 instruction-tuning records with
the fields `instruction`, `input`, and `output`.

| Dataset key | File | Records |
|---|---|---:|
| `zh_math_train_1k` | `zh_math_train_1k.json` | 1,000 |
| `ru_math_train_1k` | `ru_math_train_1k.json` | 1,000 |

The records were sampled from the GSM8K training split and translated into
the target language for fusion training. Dataset provenance and links to the
evaluation sets and expert resources are documented in
[`docs/DATA.md`](../docs/DATA.md). SHA-256 digests are provided in
`SHA256SUMS`.

To register the files in LLaMA-Factory, copy both JSON files into its `data/`
directory and merge the entries from `dataset_info.json` into the project's
existing `data/dataset_info.json` object.
