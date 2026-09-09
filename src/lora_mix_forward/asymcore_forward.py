# lora_mix/asymcore_forward.py
# Enhanced AsymCore forward.
# Supports:
#   route_granularity: module / layer
#   factor_variant: right_shared / left_shared / both_diff
#   gate_mode: raw / vproj / rproj

import torch

import os

def _asycore_stage_mem(tag, self=None, extra=""):
    if os.environ.get("ASYCORE_STAGE_DIAG", "0") != "1":
        return
    try:
        import torch
        if not torch.cuda.is_available():
            return
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        dev = torch.cuda.current_device()
        torch.cuda.synchronize(dev)
        alloc = torch.cuda.memory_allocated(dev) / 1024**3
        reserv = torch.cuda.memory_reserved(dev) / 1024**3
        max_alloc = torch.cuda.max_memory_allocated(dev) / 1024**3
        layer = getattr(self, "_layer_id", None) if self is not None else None
        module = getattr(self, "_module_id", None) if self is not None else None
        print(
            f"[ASY-STAGE][rank={rank}] {tag} "
            f"layer={layer} module={module} "
            f"alloc={alloc:.3f}GB reserved={reserv:.3f}GB max={max_alloc:.3f}GB {extra}"
        )
    except Exception as e:
        print(f"[ASY-STAGE][ERR] {tag}: {e}")

import torch.nn.functional as F


def _check_asymcore_ready(module):
    if not hasattr(module, "_get_gate_params"):
        raise RuntimeError("[AsymCore] current module has no _get_gate_params(); initialize it before forward.")
    if not hasattr(module, "_get_asymcore_params"):
        raise RuntimeError("[AsymCore] current module has no _get_asymcore_params(); initialize it before forward.")
    if not hasattr(module, "_adapter_order"):
        raise RuntimeError("[AsymCore] current module has no _adapter_order; initialize it before forward.")


def _get_asymcore_runtime_cache(self, refs, target_device, target_dtype):
    cache_key = (str(target_device), str(target_dtype))
    if hasattr(self, "_asymcore_runtime_cache"):
        old_key = self._asymcore_runtime_cache.get("key", None)
        if old_key == cache_key:
            return self._asymcore_runtime_cache["data"]

    U_ref = refs["U_ref"].to(device=target_device, dtype=target_dtype)  # [m, d_left]
    V_ref = refs["V_ref"].to(device=target_device, dtype=target_dtype)  # [n, d_right]
    data = {
        "U_ref_t": U_ref.transpose(0, 1).contiguous(),
        "V_ref": V_ref.contiguous(),
    }
    self._asymcore_runtime_cache = {"key": cache_key, "data": data}
    return data


def _delta_from_C_and_z(z: torch.Tensor, C: torch.Tensor, U_ref_t: torch.Tensor) -> torch.Tensor:
    # z: [..., r], C: [d_left, r], U_ref_t: [d_left, m]
    return (z @ C.transpose(0, 1)) @ U_ref_t


def _mean_R_for_gate(refs, adapter_order, device, dtype):
    Rs = []
    for adapter_name in adapter_order:
        R = refs["per_adapter"][adapter_name].get("R", None)
        if R is None:
            continue
        Rs.append(R.to(device=device, dtype=dtype))
    if len(Rs) == 0:
        return None
    return torch.stack(Rs, dim=0).mean(dim=0).contiguous()  # [r, d_right]


def _build_module_gate_feature(
    x: torch.Tensor,
    xV: torch.Tensor,
    z_gate: torch.Tensor,
    gate_mode: str,
    W_gate: torch.Tensor,
) -> torch.Tensor:
    gate_mode = str(gate_mode).lower()
    if gate_mode == "raw":
        return x.to(device=W_gate.device, dtype=W_gate.dtype)
    if gate_mode == "vproj":
        return xV.to(device=W_gate.device, dtype=W_gate.dtype)
    if gate_mode == "rproj":
        return z_gate.to(device=W_gate.device, dtype=W_gate.dtype)
    raise ValueError(f"[AsymCore] unsupported gate_mode={gate_mode}; expected raw, vproj, or rproj")


def _build_gate_weights(self, x, xV, z_gate, refs, W_gate, b_gate):
    route_granularity = str(refs.get("route_granularity", getattr(self, "_asymcore_route_granularity", "module"))).lower()
    gate_mode = str(refs.get("gate_mode", getattr(self, "_asymcore_gate_mode", "raw"))).lower()

    if route_granularity == "layer":
        if gate_mode != "raw":
            raise RuntimeError("[AsymCore] layer-level route only supports raw gate mode.")
        if not hasattr(self, "_layer_input_for_gate"):
            raise RuntimeError(
                f"[AsymCore] Missing _layer_input_for_gate for layer-level gate: "
                f"layer_id={getattr(self, '_layer_id', None)}, module_id={getattr(self, '_module_id', None)}"
            )
        # Treat the shared layer input as a fixed routing feature so that each
        # layer gate does not retain an additional hidden-state autograd graph.
        gate_feat = self._layer_input_for_gate.detach().to(device=W_gate.device, dtype=W_gate.dtype)
    else:
        gate_feat = _build_module_gate_feature(
            x=x,
            xV=xV,
            z_gate=z_gate,
            gate_mode=gate_mode,
            W_gate=W_gate,
        )

    x_flat = gate_feat.reshape(-1, gate_feat.shape[-1])

    import os
    if os.environ.get("ASYCORE_NEW_DIAG_GATE", "0") == "1":
        import torch
        cnt = getattr(self, "_asycore_gate_diag_count", 0)
        if cnt < 20:
            self._asycore_gate_diag_count = cnt + 1
            rank = int(os.environ.get("LOCAL_RANK", "0"))
            dev = torch.cuda.current_device() if torch.cuda.is_available() else -1
            if torch.cuda.is_available():
                torch.cuda.synchronize(dev)
                alloc = torch.cuda.memory_allocated(dev) / 1024**3
                reserv = torch.cuda.memory_reserved(dev) / 1024**3
                max_alloc = torch.cuda.max_memory_allocated(dev) / 1024**3
            else:
                alloc = reserv = max_alloc = -1
            print(
                f"[NEW-DIAG][GATE][rank={rank}] "
                f"layer={getattr(self, '_layer_id', None)} module={getattr(self, '_module_id', None)} "
                f"route={route_granularity} gate_mode={gate_mode} "
                f"gate_feat={tuple(gate_feat.shape)} {gate_feat.dtype} device={gate_feat.device} "
                f"requires_grad={gate_feat.requires_grad} contiguous={gate_feat.is_contiguous()} "
                f"x_flat={tuple(x_flat.shape)} {x_flat.dtype} contiguous={x_flat.is_contiguous()} "
                f"x_flat_base_is_none={x_flat._base is None} "
                f"W_gate={tuple(W_gate.shape)} {W_gate.dtype} requires_grad={W_gate.requires_grad} "
                f"b_gate={None if b_gate is None else (tuple(b_gate.shape), str(b_gate.dtype))} "
                f"mem_alloc={alloc:.3f}GB mem_reserved={reserv:.3f}GB max_alloc={max_alloc:.3f}GB"
            )

    if x_flat.shape[-1] != W_gate.shape[-1]:
        raise RuntimeError(
            f"[AsymCore] gate dim mismatch: "
            f"layer_id={getattr(self, '_layer_id', None)}, module_id={getattr(self, '_module_id', None)}, "
            f"route={route_granularity}, gate_mode={gate_mode}, "
            f"gate_feat.shape={tuple(gate_feat.shape)}, W_gate.shape={tuple(W_gate.shape)}"
        )

    logits = x_flat @ W_gate.transpose(0, 1)
    if b_gate is not None:
        logits = logits + b_gate
    if route_granularity == "layer" and hasattr(self, "_get_layer_gate_residual_params"):
        residual_A, residual_B = self._get_layer_gate_residual_params()
        if residual_A is not None or residual_B is not None:
            if residual_A is None or residual_B is None:
                raise RuntimeError("[AsymCore] layer-gate residual A and B must either both exist or both be absent.")
            residual_A = residual_A.to(device=x_flat.device, dtype=x_flat.dtype)
            residual_B = residual_B.to(device=x_flat.device, dtype=x_flat.dtype)
            logits = logits + (x_flat @ residual_A.transpose(0, 1)) @ residual_B.transpose(0, 1)
    weights = F.softmax(logits, dim=-1).view(*gate_feat.shape[:-1], -1)
    return weights

def _compute_deltas(xV, refs, adapter_order, U_ref_t, struct_device, struct_dtype):
    factor_variant = str(refs.get("factor_variant", "right_shared")).lower()
    deltas = []

    if factor_variant == "right_shared":
        R_shared = refs["R_shared"].to(device=struct_device, dtype=struct_dtype)
        z_shared = xV @ R_shared.transpose(0, 1).contiguous()  # [..., r]
        for adapter_name in adapter_order:
            C = refs["per_adapter"][adapter_name]["C"].to(device=struct_device, dtype=struct_dtype)
            deltas.append(_delta_from_C_and_z(z_shared, C, U_ref_t))
        z_gate = z_shared

    elif factor_variant == "left_shared":
        C_shared = refs["C_shared"].to(device=struct_device, dtype=struct_dtype)
        R_gate_list = []
        for adapter_name in adapter_order:
            R = refs["per_adapter"][adapter_name]["R"].to(device=struct_device, dtype=struct_dtype)
            z_i = xV @ R.transpose(0, 1).contiguous()
            deltas.append(_delta_from_C_and_z(z_i, C_shared, U_ref_t))
            R_gate_list.append(R)
        R_gate = torch.stack(R_gate_list, dim=0).mean(dim=0).contiguous()
        z_gate = xV @ R_gate.transpose(0, 1).contiguous()

    elif factor_variant == "both_diff":
        R_gate_list = []
        for adapter_name in adapter_order:
            pa = refs["per_adapter"][adapter_name]
            C = pa["C"].to(device=struct_device, dtype=struct_dtype)
            R = pa["R"].to(device=struct_device, dtype=struct_dtype)
            z_i = xV @ R.transpose(0, 1).contiguous()
            deltas.append(_delta_from_C_and_z(z_i, C, U_ref_t))
            R_gate_list.append(R)
        R_gate = torch.stack(R_gate_list, dim=0).mean(dim=0).contiguous()
        z_gate = xV @ R_gate.transpose(0, 1).contiguous()

    else:
        raise ValueError(f"[AsymCore] unsupported factor_variant={factor_variant}")

    return deltas, z_gate


def forward_asymcore(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    Generalized AsymCore forward.

    right_shared:
        expert_i(x) = x V_ref R_s^T C_i^T U_ref^T

    left_shared:
        expert_i(x) = x V_ref R_i^T C_s^T U_ref^T

    both_diff:
        expert_i(x) = x V_ref R_i^T C_i^T U_ref^T

    module-level gate:
        raw   : x
        vproj : x V_ref
        rproj : x V_ref R_gate^T

    layer-level gate:
        raw layer input, shared by all LoRA modules in the same Transformer layer.
    """
    _check_asymcore_ready(self)

    result = self.base_layer(x, *args, **kwargs)
    torch_result_dtype = result.dtype

    refs = self._get_asymcore_params()
    adapter_order = getattr(self, "_adapter_order", [])
    if len(adapter_order) == 0:
        raise RuntimeError("[AsymCore] adapter_order is empty; cannot run forward.")

    K = len(adapter_order)
    struct_device = refs["U_ref"].device
    # Use activation dtype for runtime computation to match bf16 training memory behavior.
    # Params may be stored in fp32, but forward activations should not be forced to fp32.
    struct_dtype = x.dtype

    runtime = _get_asymcore_runtime_cache(
        self=self,
        refs=refs,
        target_device=struct_device,
        target_dtype=struct_dtype,
    )

    x_struct = x.to(device=struct_device, dtype=struct_dtype)
    xV = x_struct @ runtime["V_ref"]  # [..., d_right]

    U_ref_t = runtime["U_ref_t"]
    deltas, z_gate = _compute_deltas(
        xV=xV,
        refs=refs,
        adapter_order=adapter_order,
        U_ref_t=U_ref_t,
        struct_device=struct_device,
        struct_dtype=struct_dtype,
    )

    W_gate, b_gate = self._get_gate_params()
    weights = _build_gate_weights(self, x, xV, z_gate, refs, W_gate, b_gate)

    if weights.shape[-1] != K:
        raise RuntimeError(
            f"[AsymCore] gate and structure adapter counts differ: "
            f"weights.shape[-1]={weights.shape[-1]}, adapter_order={adapter_order}"
        )

    # Accumulate expert outputs without materializing a stacked expert tensor.
    mixed_out = None
    for i, delta_i in enumerate(deltas):
        weight_i = weights[..., i].to(dtype=delta_i.dtype, device=delta_i.device).unsqueeze(-1)
        contrib_i = weight_i * delta_i
        mixed_out = contrib_i if mixed_out is None else mixed_out + contrib_i

    result = result + mixed_out.to(dtype=torch_result_dtype)
    return result.to(torch_result_dtype)
