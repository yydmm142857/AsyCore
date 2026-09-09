import torch
import torch.nn.functional as F


def _loraall_rank0():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def _init_loraall_diagnostics():
    if not hasattr(torch, "_loraall_diag_stats"):
        torch._loraall_diag_stats = {
            "sum_norm": {},
            "sum_norm_sq": {},
            "sum_gate": {},
            "sum_gate_sq": {},
            "sum_weighted_norm": {},

            "sum_entropy": {},
            "sum_entropy_sq": {},
            "sum_margin": {},
            "sum_margin_sq": {},

            "sum_cos_01": {},
            "sum_conflict_energy": {},
            "sum_shared_energy": {},

            "sum_merge_cos_0": {},
            "sum_merge_cos_1": {},

            "sum_switch_l1": {},
            "count_tokens": {},
            "count_switch": {},
        }


def _init_list_entry(dic, key, K):
    if key not in dic:
        dic[key] = [0.0 for _ in range(K)]


def _collect_loraall_diagnostics(self, weights: torch.Tensor, deltas: list, weighted_sum: torch.Tensor):
    if not getattr(self, "_enable_loraall_diagnostics", False):
        return

    if self.training:
        return
    if not _loraall_rank0():
        return
    if len(deltas) == 0:
        return

    layer_id = getattr(self, "_layer_id", None)
    module_id = getattr(self, "_module_id", None)
    if layer_id is None or module_id is None or layer_id < 0:
        return

    _init_loraall_diagnostics()
    st = torch._loraall_diag_stats
    key = (int(layer_id), int(module_id))

    weights_f = weights.detach().float().reshape(-1, weights.shape[-1])
    N, K = weights_f.shape

    deltas_f = [d.detach().float().reshape(-1, d.shape[-1]) for d in deltas]
    merge_f = weighted_sum.detach().float().reshape(-1, weighted_sum.shape[-1])

    _init_list_entry(st["sum_norm"], key, K)
    _init_list_entry(st["sum_norm_sq"], key, K)
    _init_list_entry(st["sum_gate"], key, K)
    _init_list_entry(st["sum_gate_sq"], key, K)
    _init_list_entry(st["sum_weighted_norm"], key, K)

    norms = []
    for i in range(K):
        ni = deltas_f[i].norm(dim=-1)
        wi = weights_f[:, i]

        norms.append(ni)

        st["sum_norm"][key][i] += ni.sum().item()
        st["sum_norm_sq"][key][i] += (ni * ni).sum().item()

        st["sum_gate"][key][i] += wi.sum().item()
        st["sum_gate_sq"][key][i] += (wi * wi).sum().item()

        st["sum_weighted_norm"][key][i] += (wi * ni).sum().item()

    eps = 1e-12
    entropy = -(weights_f.clamp_min(eps) * weights_f.clamp_min(eps).log()).sum(dim=-1)

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
        diff = (weights[:, 1:, :] - weights[:, :-1, :]).detach().float().abs().sum(dim=-1)
        st["sum_switch_l1"][key] = st["sum_switch_l1"].get(key, 0.0) + diff.sum().item()
        st["count_switch"][key] = st["count_switch"].get(key, 0) + diff.numel()

    if K >= 2:
        d0 = deltas_f[0]
        d1 = deltas_f[1]

        n0 = norms[0]
        n1 = norms[1]

        cos01 = F.cosine_similarity(d0, d1, dim=-1)

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


def forward_loraall(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:

    result = self.base_layer(x, *args, **kwargs)
    torch_result_dtype = result.dtype

    W, b = self._get_gate_params()

    x_gate = x.to(W.dtype)
    x_flat = x_gate.reshape(-1, x_gate.shape[-1])

    if x_flat.shape[-1] != W.shape[-1]:
        raise RuntimeError(
            f"[LORA-ALL] gate dim mismatch: "
            f"layer_id={getattr(self, '_layer_id', None)}, "
            f"module_id={getattr(self, '_module_id', None)}, "
            f"x.shape={tuple(x.shape)}, "
            f"W.shape={tuple(W.shape)}"
        )

    logits = torch.matmul(x_flat, W.T)
    if b is not None:
        logits = logits + b

    weights = F.softmax(logits, dim=-1)
    weights = weights.view(*x_gate.shape[:-1], -1)

    if getattr(self, "is_generating", False) and getattr(self, "_enable_loraall_diagnostics", False):
        if weights.dim() == 3 and weights.shape[1] == 1:
            w = weights.detach().float().cpu().tolist()[0][0]
            self.loraall_cache.append(w)

    x_lora = x.to(next(iter(self.lora_A.values())).weight.dtype)

    deltas = []
    adapter_order = getattr(self, "_adapter_order", list(self.active_adapters))

    for adapter in adapter_order:
        if adapter not in self.active_adapters:
            continue
        if adapter not in self.lora_A:
            continue

        lora_A = self.lora_A[adapter]
        lora_B = self.lora_B[adapter]
        dropout = self.lora_dropout[adapter]
        scaling = self.scaling[adapter]

        delta = lora_B(lora_A(dropout(x_lora))) * scaling
        deltas.append(delta)

    if len(deltas) == 0:
        return result.to(torch_result_dtype)

    stacked = torch.stack(deltas, dim=-2)
    weights_exp = weights.unsqueeze(-1)
    weighted_sum = torch.sum(weights_exp * stacked, dim=-2)

    _collect_loraall_diagnostics(self, weights, deltas, weighted_sum)

    result = result + weighted_sum
    return result.to(torch_result_dtype)