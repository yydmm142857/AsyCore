import os
import json
import torch

def save_cat_layer_weights(model, trainer):

    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    os.makedirs(trainer.args.output_dir, exist_ok=True)
    real_model = model.module if hasattr(model, "module") else model
    layer_params = [
        (name, p) for name, p in real_model.named_parameters()
        if name.startswith("adapter_weights_")
    ]
    if not layer_params:
        print("[CAT_LAYER] No layer-shared adapter weights found.")
        return

    first_name, first_param = layer_params[0]
    last_name, last_param = layer_params[-1]

    def _save_single(name, param, suffix):
        w = param.clone().detach().cpu()
        record = {
            "step": int(trainer.state.global_step),
            "param_name": name,
            "adapter_weights": w.tolist(),
            "normalized_weights": torch.softmax(w, dim=0).tolist(),
        }

        path = os.path.join(trainer.args.output_dir, f"catlayer_adapter_weights_{suffix}.json")
        data = []
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
        data.append(record)

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"[CAT_LAYER] Saved {suffix} adapter weights ({name}) -> step {record['step']} to {path}")

    _save_single(first_name, first_param, "first")
    _save_single(last_name, last_param, "last")

    current_step = trainer.state.global_step
    total_steps = getattr(trainer.state, "max_steps", None)
    if total_steps is not None and current_step >= total_steps:
        print(f"[CAT_LAYER] Final step {current_step} reached -- saving full adapter weights snapshot...")

        all_weights = {}
        for name, param in layer_params:
            w = torch.softmax(param.detach().cpu(), dim=0)
            all_weights[name] = w.tolist()

        final_path = os.path.join(trainer.args.output_dir, "catlayer_final_adapter_weights.json")
        record = {
            "step": current_step,
            "total_layers": len(all_weights),
            "weights": all_weights
        }
        with open(final_path, "w") as f:
            json.dump(record, f, indent=4)

        print(f"[CAT_LAYER] Saved final adapter weights ({len(all_weights)} layers) -> {final_path}")
