import torch
import torch.nn.functional as F


def _check_comol_ready(module):
    if not hasattr(module, "_get_gate_params"):
        raise RuntimeError("[CoMoL] current module has no _get_gate_params(); initialize it before forward.")

    if not hasattr(module, "_get_comol_params"):
        raise RuntimeError("[CoMoL] current module has no _get_comol_params(); initialize it before forward.")

    if not hasattr(module, "_adapter_order"):
        raise RuntimeError("[CoMoL] current module has no _adapter_order; initialize it before forward.")


def forward_comol(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    _check_comol_ready(self)

    result = self.base_layer(x, *args, **kwargs)
    torch_result_dtype = result.dtype

    refs = self._get_comol_params()
    adapter_order = getattr(self, "_adapter_order", [])

    if len(adapter_order) == 0:
        return result.to(torch_result_dtype)

    B_shared = refs["B_shared"]     # [m, r]
    A_shared = refs["A_shared"]     # [r, n]

    struct_device = A_shared.device
    struct_dtype = A_shared.dtype

    if B_shared.device != struct_device:
        raise RuntimeError("[CoMoL] B_shared and A_shared are on different devices.")

    x_struct = x.to(device=struct_device, dtype=struct_dtype)      # [..., n]
    x_hat = torch.matmul(x_struct, A_shared.transpose(0, 1))       # [..., r]

    W_gate, b_gate = self._get_gate_params()                       # [K, r], [K] or None

    gate_mode = getattr(self, "_comol_gate_mode", "vproj")
    x_gate = x_struct if gate_mode == "raw" else x_hat
    x_gate_flat = x_gate.reshape(-1, x_gate.shape[-1])
    if x_gate_flat.shape[-1] != W_gate.shape[-1]:
        raise RuntimeError(
            f"[CoMoL] gate dim mismatch: "
            f"layer_id={getattr(self, '_layer_id', None)}, "
            f"module_id={getattr(self, '_module_id', None)}, "
            f"gate_mode={gate_mode}, x_gate.shape={tuple(x_gate.shape)}, "
            f"W_gate.shape={tuple(W_gate.shape)}"
        )

    logits = torch.matmul(
        x_gate_flat.to(dtype=W_gate.dtype),
        W_gate.transpose(0, 1),
    )                                                              # [N, K]
    if b_gate is not None:
        logits = logits + b_gate

    weights = F.softmax(logits, dim=-1).view(*x_hat.shape[:-1], -1)  # [..., K]

    K = len(adapter_order)
    if weights.shape[-1] != K:
        raise RuntimeError(
            f"[CoMoL] gate and structure adapter counts differ: "
            f"weights.shape[-1]={weights.shape[-1]}, adapter_order={adapter_order}"
        )

    M_stack = torch.stack(
        [
            refs["per_adapter"][adapter_name]["M"].to(device=struct_device, dtype=struct_dtype)
            for adapter_name in adapter_order
        ],
        dim=0,
    )                                                              # [K, r, r]

    M_merged = torch.einsum(
        "...k,kab->...ab",
        weights.to(dtype=struct_dtype),
        M_stack,
    )                                                              # [..., r, r]

    mid = torch.matmul(
        x_hat.unsqueeze(-2),                                       # [..., 1, r]
        M_merged.transpose(-1, -2),                                # [..., r, r]
    ).squeeze(-2)                                                  # [..., r]

    delta = torch.matmul(mid, B_shared.transpose(0, 1))            # [..., m]

    result = result + delta.to(dtype=torch_result_dtype)
    return result.to(torch_result_dtype)
