import torch
from .base import register_mix_mode


@register_mix_mode("average_merge")
def init_average_merge(
    model,
    adapters_to_load,
    is_trainable,
    finetuning_args,
    training_args,
):
    n_adapter = len(adapters_to_load)
    if n_adapter < 2:
        raise ValueError("[Average Merge] At least two adapters are required.")

    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad_(False)

    device = next(model.parameters()).device
    weights = torch.full(
        (n_adapter,),
        1.0 / n_adapter,
        dtype=torch.float32,
        device=device,
    )
    model.register_buffer("average_merge_weights", weights)

    for module in model.modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            module._get_average_merge_weights = (
                lambda w=model.average_merge_weights: w
            )

    print(
        f"[Average Merge] Fixed uniform weights registered: "
        f"{weights.detach().cpu().tolist()}"
    )
    return model
