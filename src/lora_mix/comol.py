import os
import re
import torch
from .base import register_mix_mode


def is_global_rank0():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0

    rank = os.environ.get("RANK")
    if rank is not None:
        return int(rank) == 0

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank) == 0

    return True


def print0(*args, **kwargs):
    if is_global_rank0():
        print(*args, **kwargs)


def _infer_module_input_dim(module) -> int:
    candidates = []

    base_layer = getattr(module, "base_layer", None)
    if base_layer is not None:
        in_features = getattr(base_layer, "in_features", None)
        if isinstance(in_features, int):
            candidates.append(in_features)

        weight = getattr(base_layer, "weight", None)
        if weight is not None and hasattr(weight, "shape") and len(weight.shape) == 2:
            candidates.append(int(weight.shape[1]))

    in_features = getattr(module, "in_features", None)
    if isinstance(in_features, int):
        candidates.append(in_features)

    weight = getattr(module, "weight", None)
    if weight is not None and hasattr(weight, "shape") and len(weight.shape) == 2:
        candidates.append(int(weight.shape[1]))

    for d in candidates:
        if isinstance(d, int) and d > 0:
            return d

    raise ValueError(
        f"[CoMoL] cannot infer module input dimension; module={module.__class__.__name__}"
    )


def _infer_module_output_dim(module) -> int:
    candidates = []

    base_layer = getattr(module, "base_layer", None)
    if base_layer is not None:
        out_features = getattr(base_layer, "out_features", None)
        if isinstance(out_features, int):
            candidates.append(out_features)

        weight = getattr(base_layer, "weight", None)
        if weight is not None and hasattr(weight, "shape") and len(weight.shape) == 2:
            candidates.append(int(weight.shape[0]))

    out_features = getattr(module, "out_features", None)
    if isinstance(out_features, int):
        candidates.append(out_features)

    weight = getattr(module, "weight", None)
    if weight is not None and hasattr(weight, "shape") and len(weight.shape) == 2:
        candidates.append(int(weight.shape[0]))

    for d in candidates:
        if isinstance(d, int) and d > 0:
            return d

    raise ValueError(
        f"[CoMoL] cannot infer module output dimension; module={module.__class__.__name__}"
    )


def _module_key_and_id(module_name: str):
    module_map = {
        "q_proj": 0,
        "k_proj": 1,
        "v_proj": 2,
        "o_proj": 3,
        "gate_proj": 4,
        "up_proj": 5,
        "down_proj": 6,
    }

    match = re.search(r"layers\.(\d+)", module_name)
    if match:
        layer_id = int(match.group(1))
        layer_key = f"layer_{layer_id}"
    else:
        layer_id = -1
        layer_key = "shared"

    module_id = None
    for k, v in module_map.items():
        if k in module_name:
            module_id = v
            break

    return layer_id, layer_key, module_id


def _to_float_tensor(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.detach().to(device=device, dtype=torch.float32).contiguous()


def _safe_fullname(prefix: str, full_key: str, adapter_name: str = None) -> str:
    if adapter_name is None:
        return f"{prefix}_{full_key}"
    return f"{prefix}_{full_key}_{adapter_name}"


def _register_parameter_once(
    model,
    name: str,
    tensor: torch.Tensor,
    requires_grad: bool,
) -> torch.nn.Parameter:
    if hasattr(model, name):
        p = getattr(model, name)
        p.requires_grad = requires_grad
        return p
    p = torch.nn.Parameter(tensor, requires_grad=requires_grad)
    model.register_parameter(name, p)
    return getattr(model, name)


def _extract_effective_lora_AB(module, adapter_name: str, device: torch.device):
    if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
        raise KeyError(f"[CoMoL] adapter {adapter_name} not found in module")

    A = _to_float_tensor(module.lora_A[adapter_name].weight, device)  # [r, n]
    B = _to_float_tensor(module.lora_B[adapter_name].weight, device)  # [m, r]
    scaling = float(module.scaling[adapter_name])

    B_eff = B * scaling
    return A, B_eff


def _diag_from_vec(s: torch.Tensor) -> torch.Tensor:
    return torch.diag(s).contiguous()


def _svd_init_from_single_expert(A: torch.Tensor, B_eff: torch.Tensor):
    U_B, S_B, Vh_B = torch.linalg.svd(B_eff, full_matrices=False)  # [m,r], [r], [r,r]
    U_A, S_A, Vh_A = torch.linalg.svd(A, full_matrices=False)      # [r,r], [r], [r,n]

    Sigma_B = _diag_from_vec(S_B)                                  # [r,r]
    Sigma_A = _diag_from_vec(S_A)                                  # [r,r]

    B_shared_init = U_B.contiguous()                               # [m,r]
    A_shared_init = Vh_A.contiguous()                              # [r,n]
    M_init = (Sigma_B @ Vh_B @ U_A @ Sigma_A).contiguous()         # [r,r]

    return B_shared_init, A_shared_init, M_init


def _project_core_from_shared(
    A: torch.Tensor,
    B_eff: torch.Tensor,
    B_shared: torch.Tensor,
    A_shared: torch.Tensor,
):
    delta_w = torch.matmul(B_eff, A)                               # [m,n]
    M = torch.matmul(B_shared.transpose(0, 1), delta_w)            # [r,n]
    M = torch.matmul(M, A_shared.transpose(0, 1)).contiguous()    # [r,r]
    return M


def _build_comol_init_from_module(
    module,
    adapter_order,
    device,
    ref_adapter_name: str,
):
    if ref_adapter_name not in adapter_order:
        raise ValueError(f"[CoMoL] ref_adapter_name={ref_adapter_name} is absent from adapter_order")

    expert_cache = {}
    rank_set = set()

    for adapter_name in adapter_order:
        A, B_eff = _extract_effective_lora_AB(module, adapter_name, device)
        expert_cache[adapter_name] = (A, B_eff)
        rank_set.add(A.shape[0])

    if len(rank_set) != 1:
        raise ValueError(
            f"[CoMoL] inconsistent expert ranks within a module: {sorted(rank_set)}"
        )

    shared_rank = list(rank_set)[0]

    ref_A, ref_B_eff = expert_cache[ref_adapter_name]
    B_shared_init, A_shared_init, M_ref = _svd_init_from_single_expert(ref_A, ref_B_eff)

    result = {
        "B_shared": B_shared_init,
        "A_shared": A_shared_init,
        "shared_rank": shared_rank,
        "per_adapter": {},
    }

    for adapter_name in adapter_order:
        A, B_eff = expert_cache[adapter_name]

        if adapter_name == ref_adapter_name:
            M0 = M_ref
        else:
            M0 = _project_core_from_shared(
                A=A,
                B_eff=B_eff,
                B_shared=B_shared_init,
                A_shared=A_shared_init,
            )

        result["per_adapter"][adapter_name] = {
            "M": M0,
        }

    return result


def _load_comol_state_if_exists(training_args, finetuning_args, device):
    explicit_path = getattr(finetuning_args, "comol_state_path", None)
    state_path = (
        str(explicit_path)
        if explicit_path
        else os.path.join(training_args.output_dir, "comol_state.pt")
    )

    if not os.path.isfile(state_path):
        raise FileNotFoundError(
            f"[CoMoL] Prediction requires a valid comol_state.pt: {state_path}"
        )

    print0(f"[CoMoL] found structural state file: {state_path}")
    state = torch.load(state_path, map_location=device)

    if not isinstance(state, dict):
        raise ValueError(f"[CoMoL] Invalid state type: {type(state)}")

    if state.get("mix_mode") != "comol":
        raise ValueError(
            f"[CoMoL] Invalid mix_mode in state: {state.get('mix_mode')}"
        )

    if not isinstance(state.get("modules"), dict) or not state["modules"]:
        raise ValueError("[CoMoL] State contains no module structures.")

    state["_loaded_from"] = state_path
    return state


@register_mix_mode("comol")
def init_comol_mix(model, adapters_to_load, is_trainable, finetuning_args, training_args):
    freeze_shared_ab = bool(getattr(finetuning_args, "comol_freeze_shared_ab", True))
    gate_mode = str(getattr(finetuning_args, "comol_gate_mode", "vproj")).lower()
    if gate_mode not in {"vproj", "raw"}:
        raise ValueError("[CoMoL] comol_gate_mode must be vproj or raw")
    device = next(model.parameters()).device
    do_train = getattr(training_args, "do_train", True)

    adapter_order = [f"adapter_{i}" for i in range(len(adapters_to_load))]
    n_adapter = len(adapter_order)

    ref_idx = int(getattr(finetuning_args, "comol_ref_adapter_idx", 0))
    ref_idx = max(0, min(ref_idx, n_adapter - 1))

    ref_adapter_name = adapter_order[ref_idx]


    print0(f"\n[CoMoL INIT] initializing Core Space Mixture of LoRA")
    print0(f"  > Device: {device}")
    print0(f"  > Adapters: {adapter_order}")
    print0(f"  > Reference expert: {ref_adapter_name}")
    print0(f"  > Gate mode: {gate_mode}")
    print0(f"  > Training mode: {do_train}")

    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = False

    saved_state = _load_comol_state_if_exists(training_args, finetuning_args, device) if not do_train else None

    structure_cache = {}
    gate_cache = {}

    for module_name, module in model.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue

        layer_id, layer_key, module_id = _module_key_and_id(module_name)
        if module_id is None:
            continue

        full_key = f"{layer_key}_m{module_id}"

        module_in_dim = _infer_module_input_dim(module)
        module_out_dim = _infer_module_output_dim(module)

        module._layer_id = layer_id
        module._module_id = module_id
        module._adapter_order = list(adapter_order)
        module._mix_method_name = "comol"

        if full_key not in structure_cache:
            if saved_state is not None and full_key not in saved_state["modules"]:
                raise KeyError(
                    f"[CoMoL] State is missing required module: {full_key}"
                )

            if saved_state is not None and full_key in saved_state.get("modules", {}):
                print0(f"[CoMoL] loading structure from saved state: {full_key}")
                state_mod = saved_state["modules"][full_key]

                saved_gate_mode = str(state_mod.get("gate_mode", "vproj")).lower()
                if saved_gate_mode != gate_mode:
                    raise ValueError(
                        f"[CoMoL] gate mode differs from saved state: "
                        f"saved={saved_gate_mode}, current={gate_mode}"
                    )

                saved_adapter_order = list(state_mod.get("adapter_order", []))
                if saved_adapter_order != adapter_order:
                    raise ValueError(
                        f"[CoMoL] cannot load state because adapter_order differs: "
                        f"saved={saved_adapter_order}, current={adapter_order}"
                    )

                B_shared_param = _register_parameter_once(
                    model,
                    _safe_fullname("comol_B_shared", full_key),
                    state_mod["B_shared"].to(device=device, dtype=torch.float32),
                    requires_grad=bool(do_train and is_trainable and (not freeze_shared_ab)),
                )
                A_shared_param = _register_parameter_once(
                    model,
                    _safe_fullname("comol_A_shared", full_key),
                    state_mod["A_shared"].to(device=device, dtype=torch.float32),
                    requires_grad=bool(do_train and is_trainable and (not freeze_shared_ab)),
                )

                shared_rank = int(state_mod["shared_rank"])

                per_adapter_refs = {}
                for adapter_name in adapter_order:
                    st = state_mod["per_adapter"][adapter_name]

                    M_param = _register_parameter_once(
                        model,
                        _safe_fullname("comol_M", full_key, adapter_name),
                        st["M"].to(device=device, dtype=torch.float32),
                        requires_grad=bool(do_train and is_trainable),
                    )
                    per_adapter_refs[adapter_name] = {
                        "M": M_param,
                    }

            else:
                available = [a for a in adapter_order if a in module.lora_A and a in module.lora_B]
                if len(available) != len(adapter_order):
                    raise ValueError(
                        f"[CoMoL] module {module_name} is missing adapters; "
                        f"expected={adapter_order}, available={list(module.lora_A.keys())}"
                    )

                tmp_A, _ = _extract_effective_lora_AB(module, adapter_order[0], device)
                base_rank = int(tmp_A.shape[0])

                print0(
                    f"[CoMoL] initializing structure {full_key} | "
                    f"in={module_in_dim}, out={module_out_dim}, rank={base_rank}"
                )

                init_pack = _build_comol_init_from_module(
                    module=module,
                    adapter_order=adapter_order,
                    device=device,
                    ref_adapter_name=ref_adapter_name,
                )

                shared_rank = int(init_pack["shared_rank"])

                B_shared_param = _register_parameter_once(
                    model,
                    _safe_fullname("comol_B_shared", full_key),
                    init_pack["B_shared"],
                    requires_grad=bool(do_train and is_trainable and (not freeze_shared_ab)),
                )
                A_shared_param = _register_parameter_once(
                    model,
                    _safe_fullname("comol_A_shared", full_key),
                    init_pack["A_shared"],
                    requires_grad=bool(do_train and is_trainable and (not freeze_shared_ab)),
                )

                per_adapter_refs = {}
                for adapter_name in adapter_order:
                    st = init_pack["per_adapter"][adapter_name]

                    M_param = _register_parameter_once(
                        model,
                        _safe_fullname("comol_M", full_key, adapter_name),
                        st["M"],
                        requires_grad=bool(do_train and is_trainable),
                    )
                    per_adapter_refs[adapter_name] = {
                        "M": M_param,
                    }

            if (
                saved_state is not None
                and full_key in saved_state.get("modules", {})
                and "gate" in saved_state["modules"][full_key]
            ):
                gate_state = saved_state["modules"][full_key]["gate"]

                W_gate = _register_parameter_once(
                    model,
                    f"gate_W_{full_key}",
                    gate_state["W"].to(device=device, dtype=torch.float32),
                    requires_grad=bool(do_train and is_trainable),
                )

                if gate_state["b"] is None:
                    b_gate = None
                else:
                    b_gate = _register_parameter_once(
                        model,
                        f"gate_b_{full_key}",
                        gate_state["b"].to(device=device, dtype=torch.float32),
                        requires_grad=bool(do_train and is_trainable),
                    )
            else:
                gate_in_dim = module_in_dim if gate_mode == "raw" else shared_rank
                W_gate = _register_parameter_once(
                    model,
                    f"gate_W_{full_key}",
                    torch.randn(n_adapter, gate_in_dim, device=device, dtype=torch.float32) * 1e-3,
                    requires_grad=bool(do_train and is_trainable),
                )
                b_gate = _register_parameter_once(
                    model,
                    f"gate_b_{full_key}",
                    torch.zeros(n_adapter, device=device, dtype=torch.float32),
                    requires_grad=bool(do_train and is_trainable),
                )

            structure_cache[full_key] = {
                "B_shared": B_shared_param,   # [m, r]
                "A_shared": A_shared_param,   # [r, n]
                "shared_rank": shared_rank,
                "per_adapter": per_adapter_refs,
            }
            gate_cache[full_key] = (W_gate, b_gate)

        refs = structure_cache[full_key]
        gate_refs = gate_cache[full_key]

        module._gate_in_dim = int(module_in_dim if gate_mode == "raw" else refs["shared_rank"])
        module._gate_out_dim = int(module_out_dim)
        module._comol_gate_mode = gate_mode
        module._comol_shared_rank = int(refs["shared_rank"])

        module._get_gate_params = (lambda W=gate_refs[0], b=gate_refs[1]: (W, b))
        module._get_comol_params = (lambda refs=refs: refs)

    if saved_state is not None:
        expected_modules = len(saved_state["modules"])
        loaded_modules = len(structure_cache)

        if loaded_modules != expected_modules:
            raise RuntimeError(
                f"[CoMoL] Module load incomplete: "
                f"loaded={loaded_modules}, expected={expected_modules}"
            )

        expected_tensors = 0
        verified_tensors = 0

        for full_key, state_mod in saved_state["modules"].items():
            refs = structure_cache[full_key]
            gate_W, gate_b = gate_cache[full_key]

            comparisons = [
                (refs["B_shared"], state_mod["B_shared"]),
                (refs["A_shared"], state_mod["A_shared"]),
                (gate_W, state_mod["gate"]["W"]),
            ]

            if state_mod["gate"].get("b") is not None:
                comparisons.append((gate_b, state_mod["gate"]["b"]))

            for adapter_name in state_mod["adapter_order"]:
                comparisons.append((
                    refs["per_adapter"][adapter_name]["M"],
                    state_mod["per_adapter"][adapter_name]["M"],
                ))

            for current, saved in comparisons:
                expected_tensors += 1
                saved_cast = saved.to(
                    device=current.device,
                    dtype=current.dtype,
                )
                if torch.equal(current.detach(), saved_cast):
                    verified_tensors += 1

        if verified_tensors != expected_tensors:
            raise RuntimeError(
                f"[CoMoL] Tensor verification failed: "
                f"verified={verified_tensors}, expected={expected_tensors}"
            )

        import json

        report = {
            "state_path": saved_state["_loaded_from"],
            "expected_modules": expected_modules,
            "loaded_modules": loaded_modules,
            "expected_tensors": expected_tensors,
            "verified_exact": verified_tensors,
            "skipped_modules": 0,
            "mismatched_tensors": 0,
        }

        os.makedirs(training_args.output_dir, exist_ok=True)
        report_path = os.path.join(
            training_args.output_dir,
            "comol_load_report.json",
        )

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print0(
            f"[CoMoL] COMOL_FULL_STATE_LOAD_OK: "
            f"modules={loaded_modules}/{expected_modules}, "
            f"verified={verified_tensors}/{expected_tensors}, "
            f"report={report_path}"
        )

    print0(
        f"[CoMoL] initialization complete; registered "
        f"{len(structure_cache)} module structures"
    )
    return model
