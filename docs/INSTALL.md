# Installation

## 1. Clone the pinned LLaMA-Factory baseline

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout ea31c43d806162a7fd98065abfef2d974fff5766
```

## 2. Install dependencies

Create an isolated Python environment, install the pinned LLaMA-Factory
checkout, and then install the dependencies listed in `requirements.txt`.
The currently verified software versions are documented in
`docs/ENVIRONMENT.md`.

## 3. Add the fusion packages

Copy these directories from this repository to the LLaMA-Factory project root:

```text
src/lora_mix
src/lora_mix_forward
src/lora_mix_save
```

They should become top-level Python packages alongside the LLaMA-Factory
`src/` directory when `PYTHONPATH` includes both the project root and its
`src` directory.

## 4. Apply the integration patches

From the LLaMA-Factory project root, apply every patch under `patches/` with
`patch -p1`. Review each patch if using a different upstream revision; the
patches are only guaranteed to target the pinned commit.

## 5. Configure the experimental resources

Prepare the Llama-2 backbone and compatible LoRA-Flow experts described in
`docs/DATA.md`. Copy this repository's `data/*.json` files into the
LLaMA-Factory `data/` directory and merge the entries from
`data/dataset_info.json` into LLaMA-Factory's `data/dataset_info.json`. Update
the model, adapter, dataset, and output paths in the YAML files under
`configs/`.

## 6. Memory-related environment flags

The verified setup used these compatibility flags when required:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ASYCORE_FORCE_CHECKPOINT_ALL=1
export ASYCORE_FORCE_BASE_BF16=1
```

Their implementation is contained in the integration patches. Whether they
are required depends on hardware and the installed framework versions.

## 7. Reproduce the main AsyCore run

The training YAML follows the paper's two-GPU setup: batch size 4 per GPU,
8 gradient-accumulation steps, and effective global batch size 64. From the
patched LLaMA-Factory root, launch distributed training with:

```bash
export PYTHONPATH="$PWD:$PWD/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ASYCORE_FORCE_CHECKPOINT_ALL=1
export ASYCORE_FORCE_BASE_BF16=1
CUDA_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 llamafactory-cli train /path/to/AsyCore/configs/asycore_train.yaml
```

After training, set `asymcore_state_path` in `asycore_predict.yaml` to the
saved `asymcore_state.pt`, select `zh_math_test_250` or `ru_math_test_250` as
`eval_dataset`, and run generation:

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train /path/to/AsyCore/configs/asycore_predict.yaml
```

LLaMA-Factory writes `generated_predictions.jsonl` in the prediction output
directory. Compute mathematical exact-answer accuracy with:

```bash
python /path/to/AsyCore/scripts/evaluate_math.py \
  --pred_file /path/to/prediction_output/generated_predictions.jsonl
```

The evaluator extracts the final numerical value from each generated response
and reference answer, normalizes decimal and fractional forms, and writes
`new_acc.json`, `predict_label.json`, and the summary fields in
`all_results.json` beside the prediction file.
