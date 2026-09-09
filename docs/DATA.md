# Experimental resources

The experiments combine one target-language LoRA expert with one English
mathematics LoRA expert on a Llama-2-7B backbone. The experts use the unified
LoRA configuration released with LoRA-Flow: rank 64, scaling factor 16,
dropout 0.1, and adapters attached to seven linear modules in each decoder
layer.

Chinese and Russian mathematical reasoning are evaluated on the MGSM-Zh and
MGSM-Ru test sets, each containing 250 examples. The 200 training examples
released with LoRA-Flow are used as the validation set. For each task, the
fusion-training set contains 1,000 examples sampled from the GSM8K training
split and translated into the target language. The two fusion-training files
used in the reported experiments are distributed under `data/` without
post-experiment alteration.

| Resource | Location |
|---|---|
| Fusion training data | `data/zh_math_train_1k.json`, `data/ru_math_train_1k.json` |
| GSM8K source corpus | [OpenAI grade-school-math](https://github.com/openai/grade-school-math) |
| MGSM evaluation data | [Google Research URL-NLP: MGSM](https://github.com/google-research/url-nlp/tree/main/mgsm) |
| LoRA experts and released fusion data | [LoRA-Flow](https://github.com/Bowen232/LoRA-Flow) and [LoRA-Flow checkpoints](https://huggingface.co/Bowen232/LoRA-Flow) |
| Llama-2 backbone | [Meta Llama 2](https://huggingface.co/meta-llama/Llama-2-7b-hf) |

Both fusion-training files contain 1,000 records. Their SHA-256 hashes are
recorded in `data/SHA256SUMS`.

The registration entries for the released training files are provided in
`data/dataset_info.json`. Register the MGSM and LoRA-Flow validation resources
under the evaluation keys used by the configuration templates.
