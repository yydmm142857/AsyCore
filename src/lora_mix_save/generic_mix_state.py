import json
from pathlib import Path

import torch


SUPPORTED_GENERIC_MODES = {"cat_layer", "loraflow", "loraall"}


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _selected_parameters(model, mix_mode):
    real_model = _unwrap(model)

    if mix_mode == "cat_layer":
        prefixes = ("adapter_weights_",)
    elif mix_mode in {"loraflow", "loraall"}:
        prefixes = ("gate_W_", "gate_b_")
    else:
        raise ValueError(
            f"Unsupported generic state mode: {mix_mode}"
        )

    return {
        name: param
        for name, param in real_model.named_parameters()
        if name.startswith(prefixes)
    }


def save_generic_mix_state(model, trainer, mix_mode):
    if mix_mode not in SUPPORTED_GENERIC_MODES:
        raise ValueError(mix_mode)

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() != 0
    ):
        return None

    params = _selected_parameters(model, mix_mode)
    if not params:
        raise RuntimeError(
            f"[{mix_mode}] No fusion parameters found for saving."
        )

    state = {
        name: param.detach().cpu().contiguous()
        for name, param in params.items()
    }

    output_dir = Path(trainer.args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"{mix_mode}_state.pt"

    payload = {
        "format": "lora_mix_trainable_only",
        "mix_mode": mix_mode,
        "global_step": int(trainer.state.global_step),
        "num_tensors": len(state),
        "num_parameters": sum(x.numel() for x in state.values()),
        "state": state,
    }
    torch.save(payload, state_path)

    print(
        f"[{mix_mode}] Saved trainable-only state: "
        f"tensors={len(state)}, path={state_path}"
    )
    return state_path


def load_generic_mix_state(
    model,
    mix_mode,
    state_path,
    output_dir,
):
    if mix_mode not in SUPPORTED_GENERIC_MODES:
        raise ValueError(mix_mode)

    state_path = Path(state_path).expanduser().resolve()
    if not state_path.is_file():
        raise FileNotFoundError(
            f"[{mix_mode}] State file not found: {state_path}"
        )

    payload = torch.load(
        state_path,
        map_location="cpu",
        weights_only=True,
    )

    if payload.get("format") != "lora_mix_trainable_only":
        raise RuntimeError(
            f"[{mix_mode}] Invalid state format: "
            f"{payload.get('format')}"
        )

    if payload.get("mix_mode") != mix_mode:
        raise RuntimeError(
            f"[{mix_mode}] State mode mismatch: "
            f"{payload.get('mix_mode')}"
        )

    saved = payload.get("state", {})
    current = _selected_parameters(model, mix_mode)

    saved_keys = set(saved)
    current_keys = set(current)

    missing = sorted(current_keys - saved_keys)
    unexpected = sorted(saved_keys - current_keys)

    if missing or unexpected:
        raise RuntimeError(
            f"[{mix_mode}] State key mismatch: "
            f"missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )

    mismatched = []
    loaded = 0
    max_abs_diff = 0.0

    with torch.no_grad():
        for name, param in current.items():
            value = saved[name]

            if tuple(value.shape) != tuple(param.shape):
                mismatched.append({
                    "name": name,
                    "saved": list(value.shape),
                    "current": list(param.shape),
                })
                continue

            param.copy_(value.to(
                device=param.device,
                dtype=param.dtype,
            ))

            diff = (
                param.detach().float().cpu()
                - value.detach().float().cpu()
            ).abs().max().item()

            max_abs_diff = max(max_abs_diff, diff)
            loaded += 1

    if mismatched:
        raise RuntimeError(
            f"[{mix_mode}] Shape mismatches: {mismatched[:10]}"
        )

    expected = len(current)
    if loaded != expected:
        raise RuntimeError(
            f"[{mix_mode}] Partial load: {loaded}/{expected}"
        )

    report = {
        "status": "FULL_STATE_LOAD_OK",
        "mix_mode": mix_mode,
        "state_path": str(state_path),
        "expected_tensors": expected,
        "loaded_tensors": loaded,
        "verified_exact": loaded,
        "missing": 0,
        "unexpected": 0,
        "mismatched": 0,
        "max_abs_diff_after_copy": max_abs_diff,
    }

    report_path = Path(output_dir) / f"{mix_mode}_load_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"[{mix_mode}] FULL_STATE_LOAD_OK: "
        f"{loaded}/{expected}, report={report_path}"
    )
    return report
