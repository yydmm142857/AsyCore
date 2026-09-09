# lora_mix/cat_layer.py
import torch
import re
from .base import register_mix_mode

@register_mix_mode("cat_layer")
def init_cat_layer_mix(model, adapters_to_load, is_trainable, finetuning_args, training_args):
    device = next(model.parameters()).device
    n_adapter = len(adapters_to_load)
    layer_shared = {}
    total_registered = 0

    print(f"\n[CAT_LAYER INIT] initializing layer-wise CAT...")
    print(f"  > Device: {device}")
    print(f"  > Merging {n_adapter} adapters: {adapters_to_load}")

    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = False

    for module_name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            match = re.search(r"layers\.(\d+)", module_name)
            if match:
                layer_id = match.group(1)
                layer_key = f"layer_{layer_id}"
            else:
                layer_key = "shared"

            if layer_key not in layer_shared:
                shared_logits = torch.zeros(n_adapter, device=device)
                shared_param = torch.nn.Parameter(shared_logits.clone().detach(), requires_grad=True)
                param_name = f"adapter_weights_{layer_key}"
                model.register_parameter(param_name, shared_param)
                layer_shared[layer_key] = shared_param
                total_registered += 1

            param_ref = layer_shared[layer_key]
            module._get_adapter_weights = (lambda p=param_ref: p)


    print(f"\n[CAT_LAYER INIT] registered {total_registered} layer-shared adapter-weight tensors.")
    print(f"  (Each has {n_adapter} elements -> total scalars = {total_registered * n_adapter})")
    return model
