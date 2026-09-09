# lora_mix/default_forward.py
import torch

def forward_default(self, x, *args, **kwargs):
    result = self.base_layer(x, *args, **kwargs)
    torch_result_dtype = result.dtype

    for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A.keys():
            continue
        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]
        x_cast = x.to(lora_A.weight.dtype)

        if not self.use_dora[active_adapter]:
            result = result + lora_B(lora_A(dropout(x_cast))) * scaling
        else:
            x_cast = dropout(x_cast)
            result = result + self.lora_magnitude_vector[active_adapter](
                x_cast, lora_A=lora_A, lora_B=lora_B, scaling=scaling, base_layer=self.get_base_layer()
            )

    return result.to(torch_result_dtype)
