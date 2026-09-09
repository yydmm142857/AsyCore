import torch


def forward_average_merge(
    self,
    x: torch.Tensor,
    *args,
    **kwargs,
) -> torch.Tensor:
    result = self.base_layer(x, *args, **kwargs)
    result_dtype = result.dtype

    if not hasattr(self, "_get_average_merge_weights"):
        raise RuntimeError(
            "[Average Merge] Missing fixed-weight binding."
        )

    weights = self._get_average_merge_weights()
    x_lora = x.to(next(iter(self.lora_A.values())).weight.dtype)

    for index, active_adapter in enumerate(self.active_adapters):
        if active_adapter not in self.lora_A:
            continue

        delta = (
            self.lora_B[active_adapter](
                self.lora_A[active_adapter](
                    self.lora_dropout[active_adapter](x_lora)
                )
            )
            * self.scaling[active_adapter]
        )
        result = result + weights[index].to(delta.dtype) * delta

    return result.to(result_dtype)
