# lora_mix/patch_peft.py
# Runtime patch for PEFT LoRA Linear.forward.
# It keeps the new PEFT source untouched and only redirects LoRA-Mix modes.

import torch
from typing import Any

from lora_mix_forward import __all__ as forward_modes


_MODE_MAP = {
    0: "default",
    5: "cat_layer",
    6: "loraflow",
    10: "loraall",
    12: "comol",
    14: "asymcore",
    16: "average_merge",
}


def _patched_linear_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    # Keep default PEFT behavior when LoRA-Mix is not initialized.
    if not hasattr(self, "_get_mix_mode"):
        return self._lora_mix_original_forward(x, *args, **kwargs)

    try:
        mode_id = int(self._get_mix_mode().item())
    except Exception:
        return self._lora_mix_original_forward(x, *args, **kwargs)

    mode = _MODE_MAP.get(mode_id, "default")
    if mode == "default":
        return self._lora_mix_original_forward(x, *args, **kwargs)

    # Respect PEFT's normal bypass cases.
    adapter_names = kwargs.get("adapter_names", None)
    if getattr(self, "disable_adapters", False) or getattr(self, "merged", False) or adapter_names is not None:
        return self._lora_mix_original_forward(x, *args, **kwargs)

    forward_func = forward_modes.get(mode)
    if forward_func is None:
        available = ", ".join(sorted(forward_modes.keys()))
        raise KeyError(
            f"[LoRA-Mix] No forward function found for mode '{mode}'. "
            f"Available modes: [{available}]"
        )

    return forward_func(self, x, *args, **kwargs)


def apply_lora_mix_patch():
    from peft.tuners.lora.layer import Linear

    if getattr(Linear, "_lora_mix_patched", False):
        return

    Linear._lora_mix_original_forward = Linear.forward
    Linear.forward = _patched_linear_forward
    Linear._lora_mix_patched = True

    print("[LoRA-Mix] Patched peft.tuners.lora.layer.Linear.forward")
