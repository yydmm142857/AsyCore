# lora_mix_forward/cat_layer_forward.py
import torch
import torch.nn.functional as F

def forward_cat_layer(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    result = self.base_layer(x, *args, **kwargs)
    torch_result_dtype = result.dtype

    x = x.to(next(iter(self.lora_A.values())).weight.dtype)

    if hasattr(self, "_get_adapter_weights"):
        weights = self._get_adapter_weights()
    else:
        raise RuntimeError("LoRA module has no _get_adapter_weights() binding in cat_layer mode.")

    weights = F.softmax(weights, dim=0)

    for i, active_adapter in enumerate(self.active_adapters):
        if active_adapter not in self.lora_A:
            continue

        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]

        delta = lora_B(lora_A(dropout(x))) * scaling
        result = result + weights[i] * delta

    return result.to(torch_result_dtype)
