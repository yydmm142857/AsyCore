from pathlib import Path
import torch


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _is_asymcore_trainable_name(name: str) -> bool:
    keys = [
        "gate_W_",
        "gate_b_",
        "gate_residual_A_",
        "gate_residual_B_",
        "asymcore_C_",
        "asymcore_R_",
        "asymcore_Cshared_",
        "asymcore_Rshared_",
    ]
    return any(k in name for k in keys)


def save_asymcore_weight(model, trainer):
    """
    Save only trainable AsymCore parameters.

    Do NOT save asymcore_Uref/asymcore_Vref buffers, because they are reconstructed
    from the two source LoRA adapters during setup. Saving those buffers makes
    asymcore_state.pt about 1GB+, while true trainable state is only tens of MB.
    """
    raw_model = _unwrap_model(model)
    output_dir = Path(trainer.args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = {}
    total_numel = 0
    total_bytes = 0

    for name, param in raw_model.named_parameters():
        if _is_asymcore_trainable_name(name):
            tensor = param.detach().cpu()
            state[name] = tensor
            total_numel += tensor.numel()
            total_bytes += tensor.numel() * tensor.element_size()

    save_path = output_dir / "asymcore_state.pt"
    torch.save(
        {
            "state_dict": state,
            "format": "asymcore_trainable_only",
            "num_tensors": len(state),
            "total_numel": total_numel,
            "total_bytes": total_bytes,
        },
        save_path,
    )

    print(
        f"[AsymCore] Saved trainable-only state to {save_path} "
        f"({len(state)} tensors, {total_bytes / 1024**2:.2f} MB)"
    )


# compatibility aliases
save_asymcore = save_asymcore_weight
save_asymcore_state = save_asymcore_weight
