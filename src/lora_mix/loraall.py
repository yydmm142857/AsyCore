import os
import json
import torch
import re
from .base import register_mix_mode


def _infer_module_input_dim(module) -> int:

    candidates = []

    base_layer = getattr(module, "base_layer", None)
    if base_layer is not None:
        in_features = getattr(base_layer, "in_features", None)
        if isinstance(in_features, int):
            candidates.append(in_features)

        weight = getattr(base_layer, "weight", None)
        if weight is not None and hasattr(weight, "shape") and len(weight.shape) == 2:
            candidates.append(int(weight.shape[1]))

    in_features = getattr(module, "in_features", None)
    if isinstance(in_features, int):
        candidates.append(in_features)

    weight = getattr(module, "weight", None)
    if weight is not None and hasattr(weight, "shape") and len(weight.shape) == 2:
        candidates.append(int(weight.shape[1]))

    for d in candidates:
        if isinstance(d, int) and d > 0:
            return d

    raise ValueError(
        f"[LORA-ALL] cannot infer module input dimension; module={module.__class__.__name__}"
    )


@register_mix_mode("loraall")
def init_loraall_mix(model, adapters_to_load, is_trainable, finetuning_args, training_args):
    device = next(model.parameters()).device
    n_adapter = len(adapters_to_load)
    gate_map = {}
    total_registered = 0

    print(f"\n[LORA-ALL INIT] initializing module-wise LoRA-Flow")
    print(f"  > Device: {device}")
    print(f"  > Merging {n_adapter} adapters: {adapters_to_load}")

    do_train = getattr(training_args, "do_train", True)
    print(f"training mode: {do_train}")

    if not do_train:
        gate_path = os.path.join(training_args.output_dir, "loraall_final_gate_params.json")
        if os.path.exists(gate_path):
            print(f"[LORA-ALL] inference mode; loading gate parameters from {gate_path} ...")
            with open(gate_path, "r") as f:
                gate_data = json.load(f)
            gate_params = gate_data.get("gate_params", gate_data)
        else:
            print(f"[LORA-ALL] gate file not found at {gate_path}; initializing gate parameters.")
            gate_params = None
    else:
        gate_params = None

    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = False

    module_map = {
        "q_proj": 0,
        "k_proj": 1,
        "v_proj": 2,
        "o_proj": 3,
        "gate_proj": 4,
        "up_proj": 5,
        "down_proj": 6,
    }

    for module_name, module in model.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue

        match = re.search(r"layers\.(\d+)", module_name)
        if match:
            layer_id = match.group(1)
            layer_key = f"layer_{layer_id}"
        else:
            layer_key = "shared"

        module_id = None
        for k, v in module_map.items():
            if k in module_name:
                module_id = v
                break

        if module_id is None:
            continue

        full_key = f"{layer_key}_m{module_id}"
        gate_in_dim = _infer_module_input_dim(module)

        module.is_generating = (do_train is False)
        module.loraall_cache = []
        module._module_id = module_id
        module._layer_id = int(layer_id) if match else -1
        module._gate_in_dim = gate_in_dim
        module._adapter_order = list(module.lora_A.keys())
        module._adapter_source_names = list(adapters_to_load)
        module._enable_loraall_diagnostics = False

        if full_key not in gate_map:
            if gate_params and full_key in gate_params:
                print(f"[LORA-ALL] loading parameters for {full_key}")
                W_np = torch.tensor(gate_params[full_key]["W"], dtype=torch.float32, device=device)
                b_np = None
                if gate_params[full_key].get("b") is not None:
                    b_np = torch.tensor(gate_params[full_key]["b"], dtype=torch.float32, device=device)

                if W_np.shape != (n_adapter, gate_in_dim):
                    raise ValueError(
                        f"[LORA-ALL] shape mismatch for {full_key} loaded W shape={tuple(W_np.shape)}, "
                        f"expected=({n_adapter}, {gate_in_dim})"
                    )

                W_gate = torch.nn.Parameter(W_np, requires_grad=False)
                b_gate = torch.nn.Parameter(b_np, requires_grad=False) if b_np is not None else None
            else:
                print(f"[LORA-ALL] initializing parameters for {full_key}, in_dim={gate_in_dim}")
                W_gate = torch.nn.Parameter(
                    torch.randn(n_adapter, gate_in_dim, device=device) * 1e-3
                )
                b_gate = torch.nn.Parameter(torch.zeros(n_adapter, device=device))

            model.register_parameter(f"gate_W_{full_key}", W_gate)
            model.register_parameter(f"gate_b_{full_key}", b_gate)
            gate_map[full_key] = (W_gate, b_gate)
            total_registered += 1

        W_ref, b_ref = gate_map[full_key]
        module._get_gate_params = (lambda W=W_ref, b=b_ref: (W, b))

    if not do_train and gate_params:
        print(f"[LORA-ALL] loaded {len(gate_params)} gate groups")
    elif not do_train:
        print("[LORA-ALL] no valid gate parameters were loaded for inference")

    print(f"[LORA-ALL] registered gate groups: {total_registered}")
    return model