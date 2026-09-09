# lora_mix/asymcore.py
# Drop-in enhanced AsymCore initializer.
# Added:
#   1) asymcore_route_granularity: module / layer
#   2) asymcore_factor_variant: right_shared / left_shared / both_diff
#   3) strict layer-level gate implemented like LoRA-Flow: one gate per Transformer layer,
#      using layer input hidden states. For layer-level routing, gate_mode is forced to raw.

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
    raise ValueError(f"[AsymCore] cannot infer module input dimension; module={module.__class__.__name__}")


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
    raise ValueError(f"[AsymCore] cannot infer module output dimension; module={module.__class__.__name__}")


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


def _safe_fullname(prefix: str, full_key: str, suffix: str = None) -> str:
    if suffix is None:
        return f"{prefix}_{full_key}"
    return f"{prefix}_{full_key}_{suffix}"


def _to_float_tensor(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.detach().to(device=device, dtype=torch.float32).contiguous()


def _register_buffer_once(model, name: str, tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(model, name):
        return getattr(model, name)
    model.register_buffer(name, tensor)
    return getattr(model, name)


def _register_parameter_once(model, name: str, tensor: torch.Tensor, requires_grad: bool) -> torch.nn.Parameter:
    if hasattr(model, name):
        p = getattr(model, name)
        p.requires_grad = requires_grad
        return p
    p = torch.nn.Parameter(tensor, requires_grad=requires_grad)
    model.register_parameter(name, p)
    return getattr(model, name)


def _extract_effective_lora_AB(module, adapter_name: str, device: torch.device):
    if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
        raise KeyError(f"[AsymCore] adapter {adapter_name} not found in module")
    A = _to_float_tensor(module.lora_A[adapter_name].weight, device)  # [r, n]
    B = _to_float_tensor(module.lora_B[adapter_name].weight, device)  # [m, r]
    scaling = float(module.scaling[adapter_name])
    B_eff = B * scaling
    return A, B_eff


def _build_reference_bases_exact(A_list, B_list):
    """
    Exact Core-style reference bases:
      U_ref spans union column space of all B_i.
      V_ref spans union row space of all A_i, equivalently column space of A_i^T.
    """
    A_stack = torch.cat(A_list, dim=0)  # [K*r, n]
    B_stack = torch.cat(B_list, dim=1)  # [m, K*r]
    U_b, _, _ = torch.linalg.svd(B_stack, full_matrices=False)
    _, _, Vh_a = torch.linalg.svd(A_stack, full_matrices=False)
    U_ref = U_b.contiguous()
    V_ref = Vh_a.transpose(0, 1).contiguous()
    return U_ref, V_ref


def _build_core_factors(A_list, B_list, U_ref: torch.Tensor, V_ref: torch.Tensor):
    C_list, R_list = [], []
    U_ref_t = U_ref.transpose(0, 1)
    for A, B_eff in zip(A_list, B_list):
        C = (U_ref_t @ B_eff).contiguous()  # [d_left, r]
        R = (A @ V_ref).contiguous()        # [r, d_right]
        C_list.append(C)
        R_list.append(R)
    return C_list, R_list


def _solve_shared_right_ls(C_list, R_list):
    """
    R_s = argmin_R sum_i || C_i R_i - C_i R ||_F^2
    Closed form: (sum_i C_i^T C_i) R = sum_i C_i^T C_i R_i
    """
    rank = int(C_list[0].shape[1])
    d_right = int(R_list[0].shape[1])
    device = C_list[0].device
    dtype = C_list[0].dtype
    gram = torch.zeros(rank, rank, device=device, dtype=dtype)
    rhs = torch.zeros(rank, d_right, device=device, dtype=dtype)
    for C, R in zip(C_list, R_list):
        CtC = C.transpose(0, 1) @ C
        gram += CtC
        rhs += CtC @ R
    return (torch.linalg.pinv(gram) @ rhs).contiguous()


def _solve_shared_left_ls(C_list, R_list):
    """
    C_s = argmin_C sum_i || C_i R_i - C R_i ||_F^2
    Closed form: C (sum_i R_i R_i^T) = sum_i C_i R_i R_i^T
    """
    d_left = int(C_list[0].shape[0])
    rank = int(C_list[0].shape[1])
    device = C_list[0].device
    dtype = C_list[0].dtype
    gram = torch.zeros(rank, rank, device=device, dtype=dtype)
    rhs = torch.zeros(d_left, rank, device=device, dtype=dtype)
    for C, R in zip(C_list, R_list):
        RRt = R @ R.transpose(0, 1)
        gram += RRt
        rhs += C @ RRt
    return (rhs @ torch.linalg.pinv(gram)).contiguous()



def _pad_lora_AB_to_rank(A: torch.Tensor, B: torch.Tensor, target_rank: int, adapter_name: str):
    """Pad heterogeneous LoRA ranks to a shared max-rank space.

    A: [r_i, in_features]
    B: [out_features, r_i]

    Padding with zeros preserves the original LoRA matrix:
        B_pad @ A_pad == B @ A
    """
    cur_rank = int(A.shape[0])
    if int(B.shape[1]) != cur_rank:
        raise ValueError(
            f"[AsymCore] {adapter_name} A/B rank mismatch: "
            f"A.shape={tuple(A.shape)}, B.shape={tuple(B.shape)}"
        )

    if cur_rank == target_rank:
        return A.contiguous(), B.contiguous()

    if cur_rank > target_rank:
        raise ValueError(
            f"[AsymCore] {adapter_name} rank {cur_rank} > target_rank {target_rank}"
        )

    A_pad = A.new_zeros((target_rank, A.shape[1]))
    B_pad = B.new_zeros((B.shape[0], target_rank))

    A_pad[:cur_rank, :] = A
    B_pad[:, :cur_rank] = B

    return A_pad.contiguous(), B_pad.contiguous()


def _build_asymcore_init_from_module(module, adapter_order, device):
    raw_A_list, raw_B_list = [], []
    rank_set = set()
    rank_by_adapter = {}

    for adapter_name in adapter_order:
        A, B_eff = _extract_effective_lora_AB(module, adapter_name, device)
        cur_rank = int(A.shape[0])
        rank_set.add(cur_rank)
        rank_by_adapter[adapter_name] = cur_rank
        raw_A_list.append(A)
        raw_B_list.append(B_eff)

    if not rank_set:
        raise ValueError("[AsymCore] empty rank_set when building init pack.")

    # Embed heterogeneous-rank experts into a shared maximum-rank latent space.
    base_rank = int(max(rank_set))

    if len(rank_set) != 1:
        print0(
            f"[AsymCore] heterogeneous ranks detected: {sorted(rank_set)}; "
            f"zero-pad all experts to max_rank={base_rank}; "
            f"rank_by_adapter={rank_by_adapter}"
        )

    A_list, B_list = [], []
    for adapter_name, A, B_eff in zip(adapter_order, raw_A_list, raw_B_list):
        A_pad, B_pad = _pad_lora_AB_to_rank(
            A=A,
            B=B_eff,
            target_rank=base_rank,
            adapter_name=adapter_name,
        )
        A_list.append(A_pad)
        B_list.append(B_pad)

    U_ref, V_ref = _build_reference_bases_exact(A_list=A_list, B_list=B_list)
    C_list, R_list = _build_core_factors(A_list=A_list, B_list=B_list, U_ref=U_ref, V_ref=V_ref)
    R_shared = _solve_shared_right_ls(C_list=C_list, R_list=R_list)
    C_shared = _solve_shared_left_ls(C_list=C_list, R_list=R_list)

    result = {
        "U_ref": U_ref,
        "V_ref": V_ref,
        "R_shared": R_shared,
        "C_shared": C_shared,
        "base_rank": base_rank,
        "left_core_dim": int(U_ref.shape[1]),
        "right_core_dim": int(V_ref.shape[1]),
        "rank_by_adapter": rank_by_adapter,
        "rank_padding_mode": "zero_pad_to_max_rank",
        "per_adapter": {},
    }
    for adapter_name, C0, R0 in zip(adapter_order, C_list, R_list):
        result["per_adapter"][adapter_name] = {"C": C0, "R": R0}
    return result


def _load_asymcore_state_if_exists(training_args, device):
    state_path = os.path.join(training_args.output_dir, "asymcore_state.pt")
    if not os.path.exists(state_path):
        return None
    print0(f"[AsymCore] found structural state file: {state_path}")
    state = torch.load(state_path, map_location=device)
    if not isinstance(state, dict) or state.get("mix_mode") != "asymcore":
        print0("[AsymCore] invalid state or mismatched mix_mode; rebuilding the structure.")
        return None
    return state


def _resolve_module_gate_in_dim(gate_mode: str, raw_input_dim: int, right_core_dim: int, base_rank: int) -> int:
    gate_mode = str(gate_mode).lower()
    if gate_mode == "raw":
        return int(raw_input_dim)
    if gate_mode == "vproj":
        return int(right_core_dim)
    if gate_mode == "rproj":
        return int(base_rank)
    raise ValueError(f"[AsymCore] unsupported asymcore_gate_mode={gate_mode}; expected raw, vproj, or rproj")


def _get_layer_hidden_size(model) -> int:
    cfg = getattr(model, "config", None)
    h = getattr(cfg, "hidden_size", None)
    if isinstance(h, int) and h > 0:
        return h
    raise ValueError("[AsymCore] cannot infer a layer-gate dimension from model.config.hidden_size.")


def _register_layer_input_hooks(model):
    """Like LoRA-Flow: capture each decoder layer input once and share it for gates in that layer."""
    for name, module in model.named_modules():
        if re.search(r"layers\.(\d+)$", name):
            def _capture_layer_input(mod, inputs, kwargs, layer_name=name):
                x_in = None
                if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
                    x_in = inputs[0]
                elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
                    x_in = kwargs["hidden_states"]
                else:
                    return
                # Do not detach: gate must receive gradients.
                for _, sub_module in mod.named_modules():
                    if hasattr(sub_module, "lora_A") and hasattr(sub_module, "lora_B"):
                        setattr(sub_module, "_layer_input_for_gate", x_in)
            module.register_forward_pre_hook(_capture_layer_input, with_kwargs=True)


def _validate_options(route_granularity: str, gate_mode: str, factor_variant: str):
    route_granularity = str(route_granularity).lower()
    gate_mode = str(gate_mode).lower()
    factor_variant = str(factor_variant).lower()

    if route_granularity not in {"module", "layer"}:
        raise ValueError("[AsymCore] asymcore_route_granularity must be module or layer")
    if gate_mode not in {"raw", "vproj", "rproj"}:
        raise ValueError("[AsymCore] asymcore_gate_mode must be raw, vproj, or rproj")
    if factor_variant not in {"right_shared", "left_shared", "both_diff"}:
        raise ValueError("[AsymCore] asymcore_factor_variant must be right_shared, left_shared, or both_diff")
    if route_granularity == "layer" and gate_mode != "raw":
        # Strict LoRA-Flow-like layer gate uses layer input hidden states.
        # vproj/rproj are module-specific because each module has its own V_ref/R.
        raise ValueError(
            "[AsymCore] layer routing requires asymcore_gate_mode=raw; "
            "vproj and rproj use module-specific reference-space inputs."
        )


@register_mix_mode("asymcore")
def init_asymcore_mix(model, adapters_to_load, is_trainable, finetuning_args, training_args):
    device = next(model.parameters()).device
    do_train = bool(getattr(training_args, "do_train", True))
    train_shared = bool(getattr(finetuning_args, "asymcore_train_shared", False))
    gate_mode = str(getattr(finetuning_args, "asymcore_gate_mode", "raw")).lower()
    route_granularity = str(getattr(finetuning_args, "asymcore_route_granularity", "module")).lower()
    factor_variant = str(getattr(finetuning_args, "asymcore_factor_variant", "right_shared")).lower()
    layer_gate_residual_rank = int(getattr(finetuning_args, "asymcore_layer_gate_residual_rank", 0))

    if layer_gate_residual_rank < 0:
        raise ValueError("[AsymCore] asymcore_layer_gate_residual_rank must be non-negative.")
    if route_granularity != "layer" and layer_gate_residual_rank != 0:
        raise ValueError("[AsymCore] the residual gate branch is only defined for layer routing.")

    _validate_options(route_granularity, gate_mode, factor_variant)

    adapter_order = [f"adapter_{i}" for i in range(len(adapters_to_load))]
    n_adapter = len(adapter_order)

    print0("\n[AsymCore INIT] initializing AsyCore")
    print0(f"  > Device: {device}")
    print0(f"  > Adapters: {adapter_order}")
    print0(f"  > Training mode: {do_train}")
    print0(f"  > Route granularity: {route_granularity}")
    print0(f"  > Gate mode: {gate_mode}")
    print0(f"  > Factor variant: {factor_variant}")
    print0(f"  > Train shared side: {train_shared}")
    print0(f"  > Layer gate residual rank: {layer_gate_residual_rank}")

    # Freeze original LoRA experts.
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = False

    if route_granularity == "layer":
        _register_layer_input_hooks(model)

    saved_state = _load_asymcore_state_if_exists(training_args, device) if not do_train else None

    structure_cache = {}
    gate_cache = {}

    for module_name, module in model.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue

        layer_id, layer_key, module_id = _module_key_and_id(module_name)
        if module_id is None:
            continue

        # Structure remains module-specific because U/V and C/R dimensions differ by module.
        struct_key = f"{layer_key}_m{module_id}"
        # Gate can be module-specific or layer-shared.
        gate_key = struct_key if route_granularity == "module" else layer_key

        raw_input_dim = _infer_module_input_dim(module)
        output_dim = _infer_module_output_dim(module)

        module._layer_id = layer_id
        module._module_id = module_id
        module._gate_out_dim = output_dim
        module._adapter_order = list(adapter_order)
        module._mix_method_name = "asymcore"
        module._asymcore_route_granularity = route_granularity
        module._asymcore_factor_variant = factor_variant

        if struct_key not in structure_cache:
            if saved_state is not None and struct_key in saved_state.get("modules", {}):
                print0(f"[AsymCore] loading structure from saved state: {struct_key}")
                state_mod = saved_state["modules"][struct_key]

                saved_gate_mode = str(state_mod.get("gate_mode", gate_mode)).lower()
                saved_route = str(state_mod.get("route_granularity", route_granularity)).lower()
                saved_variant = str(state_mod.get("factor_variant", factor_variant)).lower()
                if saved_gate_mode != gate_mode or saved_route != route_granularity or saved_variant != factor_variant:
                    raise ValueError(
                        f"[AsymCore] inference configuration mismatch: "
                        f"saved gate/route/variant=({saved_gate_mode},{saved_route},{saved_variant}), "
                        f"current=({gate_mode},{route_granularity},{factor_variant})"
                    )

                U_ref_init = state_mod["U_ref"].to(device=device, dtype=torch.float32)
                V_ref_init = state_mod["V_ref"].to(device=device, dtype=torch.float32)
                base_rank = int(state_mod["base_rank"])
                left_core_dim = int(state_mod["left_core_dim"])
                right_core_dim = int(state_mod["right_core_dim"])

                U_ref_buf = _register_buffer_once(model, _safe_fullname("asymcore_Uref", struct_key), U_ref_init)
                V_ref_buf = _register_buffer_once(model, _safe_fullname("asymcore_Vref", struct_key), V_ref_init)

                R_shared_param = None
                C_shared_param = None
                if factor_variant == "right_shared":
                    R_shared_param = _register_parameter_once(
                        model,
                        _safe_fullname("asymcore_Rshared", struct_key),
                        state_mod["R_shared"].to(device=device, dtype=torch.float32),
                        requires_grad=bool(do_train and is_trainable and train_shared),
                    )
                elif factor_variant == "left_shared":
                    C_shared_param = _register_parameter_once(
                        model,
                        _safe_fullname("asymcore_Cshared", struct_key),
                        state_mod["C_shared"].to(device=device, dtype=torch.float32),
                        requires_grad=bool(do_train and is_trainable and train_shared),
                    )

                per_adapter_refs = {}
                for adapter_name in adapter_order:
                    st = state_mod["per_adapter"][adapter_name]
                    pa = {}
                    if factor_variant in {"right_shared", "both_diff"}:
                        pa["C"] = _register_parameter_once(
                            model,
                            _safe_fullname("asymcore_C", struct_key, adapter_name),
                            st["C"].to(device=device, dtype=torch.float32),
                            requires_grad=bool(do_train and is_trainable),
                        )
                    if factor_variant in {"left_shared", "both_diff"}:
                        pa["R"] = _register_parameter_once(
                            model,
                            _safe_fullname("asymcore_R", struct_key, adapter_name),
                            st["R"].to(device=device, dtype=torch.float32),
                            requires_grad=bool(do_train and is_trainable),
                        )
                    per_adapter_refs[adapter_name] = pa

            else:
                available = [a for a in adapter_order if a in module.lora_A and a in module.lora_B]
                if len(available) != len(adapter_order):
                    raise ValueError(
                        f"[AsymCore] module {module_name} is missing adapters; "
                        f"expected={adapter_order}, available={list(module.lora_A.keys())}"
                    )
                print0(f"[AsymCore] initializing structure {struct_key} | in={raw_input_dim}, out={output_dim}")
                init_pack = _build_asymcore_init_from_module(module=module, adapter_order=adapter_order, device=device)

                base_rank = int(init_pack["base_rank"])
                left_core_dim = int(init_pack["left_core_dim"])
                right_core_dim = int(init_pack["right_core_dim"])

                U_ref_buf = _register_buffer_once(model, _safe_fullname("asymcore_Uref", struct_key), init_pack["U_ref"])
                V_ref_buf = _register_buffer_once(model, _safe_fullname("asymcore_Vref", struct_key), init_pack["V_ref"])

                R_shared_param = None
                C_shared_param = None
                if factor_variant == "right_shared":
                    R_shared_param = _register_parameter_once(
                        model,
                        _safe_fullname("asymcore_Rshared", struct_key),
                        init_pack["R_shared"],
                        requires_grad=bool(do_train and is_trainable and train_shared),
                    )
                elif factor_variant == "left_shared":
                    C_shared_param = _register_parameter_once(
                        model,
                        _safe_fullname("asymcore_Cshared", struct_key),
                        init_pack["C_shared"],
                        requires_grad=bool(do_train and is_trainable and train_shared),
                    )

                per_adapter_refs = {}
                for adapter_name in adapter_order:
                    st = init_pack["per_adapter"][adapter_name]
                    pa = {}
                    if factor_variant in {"right_shared", "both_diff"}:
                        pa["C"] = _register_parameter_once(
                            model,
                            _safe_fullname("asymcore_C", struct_key, adapter_name),
                            st["C"],
                            requires_grad=bool(do_train and is_trainable),
                        )
                    if factor_variant in {"left_shared", "both_diff"}:
                        pa["R"] = _register_parameter_once(
                            model,
                            _safe_fullname("asymcore_R", struct_key, adapter_name),
                            st["R"],
                            requires_grad=bool(do_train and is_trainable),
                        )
                    per_adapter_refs[adapter_name] = pa

            structure_cache[struct_key] = {
                "U_ref": U_ref_buf,
                "V_ref": V_ref_buf,
                "R_shared": R_shared_param,
                "C_shared": C_shared_param,
                "base_rank": base_rank,
                "left_core_dim": left_core_dim,
                "right_core_dim": right_core_dim,
                "train_shared": train_shared,
                "gate_mode": gate_mode,
                "route_granularity": route_granularity,
                "factor_variant": factor_variant,
                "per_adapter": per_adapter_refs,
            }

        refs = structure_cache[struct_key]

        # Gate registration is separated from structure registration.
        if gate_key not in gate_cache:
            if route_granularity == "layer":
                gate_in_dim = _get_layer_hidden_size(model)
            else:
                gate_in_dim = _resolve_module_gate_in_dim(
                    gate_mode=gate_mode,
                    raw_input_dim=raw_input_dim,
                    right_core_dim=refs["right_core_dim"],
                    base_rank=refs["base_rank"],
                )

            if saved_state is not None:
                # For module route, gate is saved in the module state.
                # For layer route, gate is saved once under top-level gate_params.
                gate_state = None
                if route_granularity == "module" and struct_key in saved_state.get("modules", {}):
                    gate_state = saved_state["modules"][struct_key].get("gate")
                elif route_granularity == "layer":
                    gate_state = saved_state.get("gate_params", {}).get(gate_key)
            else:
                gate_state = None

            if gate_state is not None:
                W_gate = _register_parameter_once(
                    model,
                    f"gate_W_{gate_key}",
                    gate_state["W"].to(device=device, dtype=torch.float32),
                    requires_grad=bool(do_train and is_trainable),
                )
                if int(W_gate.shape[1]) != int(gate_in_dim):
                    raise ValueError(
                        f"[AsymCore] gate dimension mismatch: W={tuple(W_gate.shape)}, expected_in={gate_in_dim}, key={gate_key}"
                    )
                b_gate = None if gate_state.get("b") is None else _register_parameter_once(
                    model,
                    f"gate_b_{gate_key}",
                    gate_state["b"].to(device=device, dtype=torch.float32),
                    requires_grad=bool(do_train and is_trainable),
                )
            else:
                W_gate = _register_parameter_once(
                    model,
                    f"gate_W_{gate_key}",
                    torch.randn(n_adapter, gate_in_dim, device=device, dtype=torch.float32) * 1e-3,
                    requires_grad=bool(do_train and is_trainable),
                )
                b_gate = _register_parameter_once(
                    model,
                    f"gate_b_{gate_key}",
                    torch.zeros(n_adapter, device=device, dtype=torch.float32),
                    requires_grad=bool(do_train and is_trainable),
                )
            residual_A = None
            residual_B = None
            if route_granularity == "layer" and layer_gate_residual_rank > 0:
                residual_A = _register_parameter_once(
                    model,
                    f"gate_residual_A_{gate_key}",
                    torch.randn(
                        layer_gate_residual_rank,
                        gate_in_dim,
                        device=device,
                        dtype=torch.float32,
                    ) * 1e-3,
                    requires_grad=bool(do_train and is_trainable),
                )
                residual_B = _register_parameter_once(
                    model,
                    f"gate_residual_B_{gate_key}",
                    torch.zeros(
                        n_adapter,
                        layer_gate_residual_rank,
                        device=device,
                        dtype=torch.float32,
                    ),
                    requires_grad=bool(do_train and is_trainable),
                )
            gate_cache[gate_key] = (
                W_gate,
                b_gate,
                gate_in_dim,
                residual_A,
                residual_B,
            )

        gate_refs = gate_cache[gate_key]
        module._gate_in_dim = int(gate_refs[2])
        module._gate_key = gate_key
        module._struct_key = struct_key
        module._asymcore_gate_mode = str(refs["gate_mode"])
        module._asymcore_base_rank = refs["base_rank"]
        module._asymcore_left_dim = refs["left_core_dim"]
        module._asymcore_right_dim = refs["right_core_dim"]
        module._get_gate_params = (lambda W=gate_refs[0], b=gate_refs[1]: (W, b))
        module._get_layer_gate_residual_params = (
            lambda A=gate_refs[3], B=gate_refs[4]: (A, B)
        )
        module._get_asymcore_params = (lambda refs=refs: refs)

    print0(
        f"[AsymCore] initialization complete; structures={len(structure_cache)}; gates={len(gate_cache)}"
    )
    return model
