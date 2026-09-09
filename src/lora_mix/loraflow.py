# lora_mix/loraflow.py
import os
import json
import torch
import re
from .base import register_mix_mode

@register_mix_mode("loraflow")
def init_loraflow_mix(model, adapters_to_load, is_trainable, finetuning_args, training_args):
    device = next(model.parameters()).device
    n_adapter = len(adapters_to_load)
    layer_gate = {}
    total_registered = 0

    print(f"\n[LORA-FLOW INIT] initializing dynamic LoRA-Flow routing ...")
    print(f"  > Device: {device}")
    print(f"  > Merging {n_adapter} adapters: {adapters_to_load}")

    
    do_train = getattr(training_args, "do_train", True)
    print(f"training mode: {do_train}")
    if not do_train:
        gate_path = os.path.join(training_args.output_dir, "loraflow_final_gate_params.json")
        if os.path.exists(gate_path):
            print(f"[LORA-FLOW] inference mode; loading gate parameters from {gate_path} ...")
            with open(gate_path, "r") as f:
                gate_data = json.load(f)
            gate_params = gate_data.get("gate_params", gate_data)
        else:
            print(f"[LORA-FLOW] gate file not found at {gate_path}; initializing gate parameters.")
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
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            match = re.search(r"layers\.(\d+)", module_name)
            if match:
                layer_id = match.group(1)
                layer_key = f"layer_{layer_id}"
            else:
                layer_key = "shared"


            if layer_key not in layer_gate:

                module.is_generating = (do_train is False)
                module.loraflow_cache = []
                hidden_dim = getattr(model.config, "hidden_size", 4096)
                if gate_params and layer_key in gate_params:
                    print(f"[LORA-flow] loading parameters for layer {layer_key}.")
                    W_np = torch.tensor(gate_params[layer_key]["W"], dtype=torch.float32, device=device)
                    b_np = None
                    if gate_params[layer_key].get("b") is not None:
                        b_np = torch.tensor(gate_params[layer_key]["b"], dtype=torch.float32, device=device)
                    W_gate = torch.nn.Parameter(W_np, requires_grad=False)
                    b_gate = torch.nn.Parameter(b_np, requires_grad=False) if b_np is not None else None
                    
                else:
                    print(f"[LORA-flow] initializing parameters for layer {layer_key}.")
                    W_gate = torch.nn.Parameter(torch.randn(n_adapter, hidden_dim, device=device) * 1e-3)
                    b_gate = torch.nn.Parameter(torch.zeros(n_adapter, device=device))
                model.register_parameter(f"gate_W_{layer_key}", W_gate)
                model.register_parameter(f"gate_b_{layer_key}", b_gate)
                layer_gate[layer_key] = (W_gate, b_gate)
                total_registered += 1

            W_ref, b_ref = layer_gate[layer_key]
            module_id = None
            for k, v in module_map.items():
                if k in module_name:
                    module_id = v
                    break

            module._layer_id = int(layer_id) if match else -1
            module._module_id = module_id
            module._get_gate_params = (lambda W=W_ref, b=b_ref: (W, b))


    for name, module in model.named_modules():
        if re.search(r"layers\.(\d+)$", name):
            def _capture_layer_input(mod, inputs, kwargs, layer_name=name):
                x_in = None
                if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
                    x_in = inputs[0]
                elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
                    x_in = kwargs["hidden_states"]
                else:
                    return

                x_in = x_in.detach()

                for _, sub_module in mod.named_modules():
                    if hasattr(sub_module, "lora_A") and hasattr(sub_module, "lora_B"):
                        setattr(sub_module, "_layer_input_for_gate", x_in)
            module.register_forward_pre_hook(_capture_layer_input, with_kwargs=True)

    if not do_train and gate_params:
        print(f"[LORA-FLOW] loaded {len(gate_params)} layer gates and bound them to LoRA modules.")
    elif not do_train:
        print("[LORA-FLOW] no valid gate parameters were loaded for inference.")

    return model
