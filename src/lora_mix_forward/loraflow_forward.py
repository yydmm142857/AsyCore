import torch
import torch.nn.functional as F


def _loraflow_rank0():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def _init_loraflow_diagnostics():
    if not hasattr(torch, "_loraflow_diag_stats"):
        torch._loraflow_diag_stats = {
            "sum_norm": {},             # key -> [K]
            "sum_norm_sq": {},          # key -> [K]
            "sum_gate": {},             # key -> [K]
            "sum_gate_sq": {},          # key -> [K]
            "sum_weighted_norm": {},    # key -> [K]

            "sum_entropy": {},          # key -> float
            "sum_entropy_sq": {},       # key -> float
            "sum_margin": {},           # key -> float
            "sum_margin_sq": {},        # key -> float

            "sum_cos_01": {},           # key -> float
            "sum_conflict_energy": {},  # key -> float
            "sum_shared_energy": {},    # key -> float

            "sum_merge_cos_0": {},      # key -> float
            "sum_merge_cos_1": {},      # key -> float

            "sum_switch_l1": {},        # key -> float
            "count_tokens": {},         # key -> int
            "count_switch": {},         # key -> int
        }


def _init_list_entry(dic, key, K):
    if key not in dic:
        dic[key] = [0.0 for _ in range(K)]


def _collect_loraflow_diagnostics(self, weights: torch.Tensor, deltas: list, weighted_sum: torch.Tensor):
    if not getattr(self, "_enable_loraflow_diagnostics", False):
        return
    if self.training:
        return
    if not _loraflow_rank0():
        return
    if len(deltas) == 0:
        return

    layer_id = getattr(self, "_layer_id", None)
    module_id = getattr(self, "_module_id", None)
    if layer_id is None or module_id is None or layer_id < 0:
        return

    _init_loraflow_diagnostics()
    st = torch._loraflow_diag_stats
    key = (int(layer_id), int(module_id))

    weights_f = weights.detach().float().reshape(-1, weights.shape[-1])   # [N, K]
    N, K = weights_f.shape

    deltas_f = [d.detach().float().reshape(-1, d.shape[-1]) for d in deltas]  # each [N, D_out]
    merge_f = weighted_sum.detach().float().reshape(-1, weighted_sum.shape[-1])  # [N, D_out]

    _init_list_entry(st["sum_norm"], key, K)
    _init_list_entry(st["sum_norm_sq"], key, K)
    _init_list_entry(st["sum_gate"], key, K)
    _init_list_entry(st["sum_gate_sq"], key, K)
    _init_list_entry(st["sum_weighted_norm"], key, K)

    norms = []
    for i in range(K):
        ni = deltas_f[i].norm(dim=-1)      # [N]
        wi = weights_f[:, i]               # [N]

        norms.append(ni)

        st["sum_norm"][key][i] += ni.sum().item()
        st["sum_norm_sq"][key][i] += (ni * ni).sum().item()

        st["sum_gate"][key][i] += wi.sum().item()
        st["sum_gate_sq"][key][i] += (wi * wi).sum().item()

        st["sum_weighted_norm"][key][i] += (wi * ni).sum().item()

    eps = 1e-12
    entropy = -(weights_f.clamp_min(eps) * weights_f.clamp_min(eps).log()).sum(dim=-1)  # [N]

    if K >= 2:
        top2 = torch.topk(weights_f, k=2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
    else:
        margin = torch.ones_like(entropy)

    st["sum_entropy"][key] = st["sum_entropy"].get(key, 0.0) + entropy.sum().item()
    st["sum_entropy_sq"][key] = st["sum_entropy_sq"].get(key, 0.0) + (entropy * entropy).sum().item()
    st["sum_margin"][key] = st["sum_margin"].get(key, 0.0) + margin.sum().item()
    st["sum_margin_sq"][key] = st["sum_margin_sq"].get(key, 0.0) + (margin * margin).sum().item()

    if weights.dim() >= 3 and weights.size(-2) > 1:
        diff = (weights[:, 1:, :] - weights[:, :-1, :]).detach().float().abs().sum(dim=-1)  # [B, T-1]
        st["sum_switch_l1"][key] = st["sum_switch_l1"].get(key, 0.0) + diff.sum().item()
        st["count_switch"][key] = st["count_switch"].get(key, 0) + diff.numel()

    if K >= 2:
        d0 = deltas_f[0]
        d1 = deltas_f[1]

        n0 = norms[0]
        n1 = norms[1]

        cos01 = F.cosine_similarity(d0, d1, dim=-1)   # [N]

        conflict_energy = 0.5 * (n0 + n1) * torch.relu(-cos01)
        shared_energy = 0.5 * (n0 + n1) * torch.relu(cos01)

        merge_cos_0 = F.cosine_similarity(merge_f, d0, dim=-1)
        merge_cos_1 = F.cosine_similarity(merge_f, d1, dim=-1)

        st["sum_cos_01"][key] = st["sum_cos_01"].get(key, 0.0) + cos01.sum().item()
        st["sum_conflict_energy"][key] = st["sum_conflict_energy"].get(key, 0.0) + conflict_energy.sum().item()
        st["sum_shared_energy"][key] = st["sum_shared_energy"].get(key, 0.0) + shared_energy.sum().item()
        st["sum_merge_cos_0"][key] = st["sum_merge_cos_0"].get(key, 0.0) + merge_cos_0.sum().item()
        st["sum_merge_cos_1"][key] = st["sum_merge_cos_1"].get(key, 0.0) + merge_cos_1.sum().item()

    st["count_tokens"][key] = st["count_tokens"].get(key, 0) + N


def forward_loraflow(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:

    result = self.base_layer(x, *args, **kwargs)
    torch_result_dtype = result.dtype

    x = x.to(next(iter(self.lora_A.values())).weight.dtype)

    W, b = self._get_gate_params()

    if not hasattr(self, "_layer_input_for_gate"):
        raise RuntimeError(
            f"[LORA-FLOW] Missing _layer_input_for_gate for "
            f"layer_id={getattr(self, '_layer_id', None)}, "
            f"module_id={getattr(self, '_module_id', None)}"
        )

    x_gate = self._layer_input_for_gate

    # Gate parameters are intentionally kept in FP32.
    # During BF16 inference, cast only the gate input to W.dtype
    # before matmul; this preserves the trained gate parameters.
    x_gate_for_gate = x_gate.to(
        device=W.device,
        dtype=W.dtype,
    )
    x_flat = x_gate_for_gate.reshape(
        -1,
        x_gate_for_gate.shape[-1],
    )  # [*, D]

    if x_flat.shape[-1] != W.shape[-1]:
        raise RuntimeError(
            f"[LORA-FLOW] gate dim mismatch: "
            f"layer_id={getattr(self, '_layer_id', None)}, "
            f"module_id={getattr(self, '_module_id', None)}, "
            f"x.shape={tuple(x.shape)}, "
            f"x_gate.shape={tuple(x_gate.shape)}, "
            f"W.shape={tuple(W.shape)}"
        )

    logits = torch.matmul(x_flat, W.T)   # [*, K]
    if b is not None:
        logits = logits + b

    weights = F.softmax(logits, dim=-1)
    weights = weights.view(*x_gate.shape[:-1], -1)  # [B,T,K] or [*,K]

    deltas = []
    for i, adapter in enumerate(self.active_adapters):
        if adapter not in self.lora_A:
            continue

        lora_A = self.lora_A[adapter]
        lora_B = self.lora_B[adapter]
        dropout = self.lora_dropout[adapter]
        scaling = self.scaling[adapter]

        delta = lora_B(lora_A(dropout(x))) * scaling
        deltas.append(delta)

    if getattr(self, "_enable_loraflow_diagnostics", False) and len(deltas) >= 2:
        layer_id = getattr(self, "_layer_id", None)
        module_id = getattr(self, "_module_id", None)

        if (
            layer_id is not None and layer_id >= 0 and
            module_id is not None and
            x_gate.dim() == 3 and x_gate.size(0) == 1 and x_gate.size(1) > 1
        ):
            import torch as _torch

            if not hasattr(_torch, "_loraflow_delta_stats"):
                _torch._loraflow_delta_stats = {
                    "sum_norm_0": {},
                    "sum_norm_1": {},
                    "sum_cos": {},
                    "count": {},
                }

            st = _torch._loraflow_delta_stats
            key = (int(layer_id), int(module_id))

            d0 = deltas[0].detach().float()
            d1 = deltas[1].detach().float()

            norm0 = d0.norm(dim=-1)
            norm1 = d1.norm(dim=-1)
            cos01 = F.cosine_similarity(d0, d1, dim=-1)

            st["sum_norm_0"][key] = st["sum_norm_0"].get(key, 0.0) + norm0.sum().item()
            st["sum_norm_1"][key] = st["sum_norm_1"].get(key, 0.0) + norm1.sum().item()
            st["sum_cos"][key] = st["sum_cos"].get(key, 0.0) + cos01.sum().item()
            st["count"][key] = st["count"].get(key, 0) + norm0.numel()

    stacked = torch.stack(deltas, dim=-2)      # [..., K, D]
    weights_exp = weights.unsqueeze(-1)        # [..., K, 1]
    weighted_sum = torch.sum(weights_exp * stacked, dim=-2)

    _collect_loraflow_diagnostics(self, weights, deltas, weighted_sum)

    result = result + weighted_sum
    return result.to(torch_result_dtype)