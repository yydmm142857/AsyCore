import os
import torch


def _rank0_only():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def _to_cpu_tensor(x):
    if x is None:
        return None
    return x.detach().cpu().to(torch.float32).contiguous()


def _full_key_from_module(module):
    layer_id = getattr(module, "_layer_id", -1)
    module_id = getattr(module, "_module_id", None)

    if module_id is None:
        raise ValueError("[CoMoL] current module has no _module_id; cannot construct full_key.")

    if layer_id is None or int(layer_id) < 0:
        layer_key = "shared"
    else:
        layer_key = f"layer_{int(layer_id)}"

    return f"{layer_key}_m{int(module_id)}"


def _collect_single_module_state(module):
    if not hasattr(module, "_get_comol_params"):
        raise RuntimeError("[CoMoL] current module has no _get_comol_params(); cannot save state.")

    if not hasattr(module, "_get_gate_params"):
        raise RuntimeError("[CoMoL] current module has no _get_gate_params(); cannot save state.")

    refs = module._get_comol_params()
    W_gate, b_gate = module._get_gate_params()

    state = {
        "layer_id": int(getattr(module, "_layer_id", -1)),
        "module_id": int(getattr(module, "_module_id", -1)),
        "gate_in_dim": int(getattr(module, "_gate_in_dim", -1)),
        "gate_out_dim": int(getattr(module, "_gate_out_dim", -1)),
        "gate_mode": str(getattr(module, "_comol_gate_mode", "vproj")),
        "shared_rank": int(refs["shared_rank"]),
        "adapter_order": list(getattr(module, "_adapter_order", [])),
        "B_shared": _to_cpu_tensor(refs["B_shared"]),
        "A_shared": _to_cpu_tensor(refs["A_shared"]),
        "per_adapter": {},
        "gate": {
            "W": _to_cpu_tensor(W_gate),
            "b": _to_cpu_tensor(b_gate) if b_gate is not None else None,
        },
    }

    for adapter_name, st in refs["per_adapter"].items():
        state["per_adapter"][adapter_name] = {
            "M": _to_cpu_tensor(st["M"]),
        }

    return state


def save_comol_params(model, trainer):
    if not _rank0_only():
        return

    os.makedirs(trainer.args.output_dir, exist_ok=True)

    real_model = model.module if hasattr(model, "module") else model

    modules_state = {}
    seen_full_keys = set()

    for _, module in real_model.named_modules():
        if getattr(module, "_mix_method_name", None) != "comol":
            continue

        full_key = _full_key_from_module(module)
        if full_key in seen_full_keys:
            continue

        seen_full_keys.add(full_key)
        modules_state[full_key] = _collect_single_module_state(module)

    if len(modules_state) == 0:
        print("[CoMoL] No CoMoL modules found, skip saving.")
        return

    save_obj = {
        "version": 1,
        "mix_mode": "comol",
        "global_step": int(getattr(trainer.state, "global_step", 0)),
        "modules": modules_state,
    }

    save_path = os.path.join(trainer.args.output_dir, "comol_state.pt")
    torch.save(save_obj, save_path)

    print(
        f"[CoMoL] Saved full state "
        f"({len(modules_state)} modules) -> {save_path}"
    )
