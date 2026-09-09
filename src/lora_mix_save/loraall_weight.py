import os
import json
import math
import torch
import traceback
from collections import defaultdict


def reset_loraall_diagnostics():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return

    if hasattr(torch, "_loraall_diag_stats"):
        delattr(torch, "_loraall_diag_stats")


def _safe_mean(sum_v, cnt):
    return float(sum_v) / float(cnt) if cnt > 0 else 0.0


def _safe_std(sum_v, sum_sq_v, cnt):
    if cnt <= 0:
        return 0.0
    mean = float(sum_v) / float(cnt)
    var = max(float(sum_sq_v) / float(cnt) - mean * mean, 0.0)
    return math.sqrt(var)


def _effective_rank_from_singular_values(s: torch.Tensor) -> float:
    s = s.float()
    denom = torch.sum(s * s).item()
    if denom <= 0:
        return 0.0
    num = (torch.sum(s).item()) ** 2
    return float(num / denom)


def _energy_ratio(s: torch.Tensor, topk: int) -> float:
    if s.numel() == 0:
        return 0.0
    ss = s.float() ** 2
    total = ss.sum().item()
    if total <= 0:
        return 0.0
    top = ss[: min(topk, ss.numel())].sum().item()
    return float(top / total)


def _rank0_only():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def _infer_adapter_names(real_model):
    for _, module in real_model.named_modules():
        if hasattr(module, "lora_A"):
            try:
                if hasattr(module.lora_A, "keys") and len(module.lora_A) > 0:
                    return list(module.lora_A.keys())
            except Exception:
                pass
    return None


def _orthonormal_basis_from_tall(X: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    X = X.detach().float()
    if X.numel() == 0:
        return X.new_zeros((X.shape[0], 0))

    G = X.transpose(0, 1) @ X
    G = 0.5 * (G + G.transpose(0, 1))
    evals, evecs = torch.linalg.eigh(G)

    max_eval = float(torch.clamp(evals.max(), min=0.0).item()) if evals.numel() > 0 else 0.0
    tol = max(eps * max_eval, eps)

    keep = evals > tol
    if not bool(keep.any()):
        return X.new_zeros((X.shape[0], 0))

    evals_kept = evals[keep].clamp_min(tol)
    evecs_kept = evecs[:, keep]

    Q = X @ (evecs_kept / torch.sqrt(evals_kept).unsqueeze(0))
    return Q


def _build_low_rank_operator(A: torch.Tensor, B: torch.Tensor, scaling: float, eps: float = 1e-6):
    A = A.detach().float()
    B = B.detach().float()

    Q_left = _orthonormal_basis_from_tall(B, eps=eps)
    Q_right = _orthonormal_basis_from_tall(A.transpose(0, 1), eps=eps)

    if Q_left.shape[1] == 0 or Q_right.shape[1] == 0:
        core = A.new_zeros((Q_left.shape[1], Q_right.shape[1]))
    else:
        C_left = Q_left.transpose(0, 1) @ B
        C_right = Q_right.transpose(0, 1) @ A.transpose(0, 1)
        core = float(scaling) * (C_left @ C_right.transpose(0, 1))

    return {
        "left_basis": Q_left,
        "right_basis": Q_right,
        "core": core,
    }


def _principal_angle_stats_from_bases(Q1: torch.Tensor, Q2: torch.Tensor):
    if Q1.numel() == 0 or Q2.numel() == 0 or Q1.shape[1] == 0 or Q2.shape[1] == 0:
        return {"min_deg": 0.0, "mean_deg": 0.0, "max_deg": 0.0}

    M = Q1.transpose(0, 1) @ Q2
    s = torch.linalg.svdvals(M)
    s = torch.clamp(s, -1.0, 1.0)
    ang = torch.rad2deg(torch.arccos(s))

    return {
        "min_deg": float(ang.min().item()) if ang.numel() > 0 else 0.0,
        "mean_deg": float(ang.mean().item()) if ang.numel() > 0 else 0.0,
        "max_deg": float(ang.max().item()) if ang.numel() > 0 else 0.0,
    }


def _operator_inner_product(op0, op1) -> float:
    C0 = op0["core"]
    C1 = op1["core"]

    if C0.numel() == 0 or C1.numel() == 0:
        return 0.0

    L = op0["left_basis"].transpose(0, 1) @ op1["left_basis"]
    R = op1["right_basis"].transpose(0, 1) @ op0["right_basis"]

    tmp = L @ C1 @ R
    return float(torch.sum(C0 * tmp).item())


def _fro_norm_from_core(op) -> float:
    C = op["core"]
    if C.numel() == 0:
        return 0.0
    return float(torch.norm(C, p="fro").item())


def _projection_ratio_right(op_src, op_tgt) -> float:
    C = op_src["core"]
    denom = torch.norm(C, p="fro").item() ** 2
    if denom <= 0:
        return 0.0

    overlap = op_src["right_basis"].transpose(0, 1) @ op_tgt["right_basis"]
    proj_core = C @ overlap
    num = torch.norm(proj_core, p="fro").item() ** 2
    return float(num / denom)


def _projection_ratio_left(op_src, op_tgt) -> float:
    C = op_src["core"]
    denom = torch.norm(C, p="fro").item() ** 2
    if denom <= 0:
        return 0.0

    overlap = op_tgt["left_basis"].transpose(0, 1) @ op_src["left_basis"]
    proj_core = overlap @ C
    num = torch.norm(proj_core, p="fro").item() ** 2
    return float(num / denom)


def _pairwise_operator_stats(op0, op1):
    norm0 = _fro_norm_from_core(op0)
    norm1 = _fro_norm_from_core(op1)

    inner = _operator_inner_product(op0, op1)
    flat_cos = float(inner / (norm0 * norm1 + 1e-12))

    left_stats = _principal_angle_stats_from_bases(op0["left_basis"], op1["left_basis"])
    right_stats = _principal_angle_stats_from_bases(op0["right_basis"], op1["right_basis"])

    return {
        "flat_cosine": flat_cos,

        "left_subspace_min_angle_deg": left_stats["min_deg"],
        "left_subspace_mean_angle_deg": left_stats["mean_deg"],
        "left_subspace_max_angle_deg": left_stats["max_deg"],

        "right_subspace_min_angle_deg": right_stats["min_deg"],
        "right_subspace_mean_angle_deg": right_stats["mean_deg"],
        "right_subspace_max_angle_deg": right_stats["max_deg"],

        "right_proj_ratio_0_on_1": _projection_ratio_right(op0, op1),
        "right_proj_ratio_1_on_0": _projection_ratio_right(op1, op0),

        "left_proj_ratio_0_on_1": _projection_ratio_left(op0, op1),
        "left_proj_ratio_1_on_0": _projection_ratio_left(op1, op0),
    }


def save_loraall_gate_params(model, trainer):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return

    os.makedirs(trainer.args.output_dir, exist_ok=True)
    real_model = model.module if hasattr(model, "module") else model

    gate_layers = [
        (name, p)
        for name, p in real_model.named_parameters()
        if name.startswith("gate_W_")
    ]

    if not gate_layers:
        print("[LORA-ALL] No gate parameters found.")
        return

    params_dict = dict(real_model.named_parameters())

    def get_b(name):
        key = name.replace("gate_W_", "")
        return params_dict.get(f"gate_b_{key}", None)

    first_name, first_W = gate_layers[0]
    last_name, last_W = gate_layers[-1]

    first_b = get_b(first_name)
    last_b = get_b(last_name)

    def _save_single(name, W, b, suffix):
        record = {
            "step": int(trainer.state.global_step),
            "layer": name,
            "gate_W": W.detach().cpu().tolist(),
            "gate_b": b.detach().cpu().tolist() if b is not None else None,
        }

        path = os.path.join(trainer.args.output_dir, f"loraall_gate_{suffix}.json")

        data = []
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)

        data.append(record)

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"[LORA-ALL] Saved {suffix} gate params -> step {record['step']}")

    _save_single(first_name, first_W, first_b, "first")
    _save_single(last_name, last_W, last_b, "last")

    current_step = trainer.state.global_step
    total_steps = getattr(trainer.state, "max_steps", None)

    if total_steps is not None and current_step >= total_steps:
        print(
            f"[LORA-ALL] Final step {current_step} reached -- "
            f"saving all gate params for inference..."
        )

        all_gate = {}

        for name, W in gate_layers:
            b = get_b(name)
            key = name.replace("gate_W_", "")
            all_gate[key] = {
                "W": W.detach().cpu().tolist(),
                "b": b.detach().cpu().tolist() if b is not None else None,
            }

        final_path = os.path.join(
            trainer.args.output_dir,
            "loraall_final_gate_params.json"
        )

        record = {
            "step": current_step,
            "total_gates": len(all_gate),
            "gate_params": all_gate,
        }

        with open(final_path, "w") as f:
            json.dump(record, f, indent=4)

        print(
            f"[LORA-ALL] Saved final gate params "
            f"({len(all_gate)} gates) -> {final_path}"
        )


def save_loraall_diagnostics(trainer):
    if not _rank0_only():
        return

    real_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    if not getattr(real_model, "_enable_loraall_diagnostics", False):
        return

    os.makedirs(trainer.args.output_dir, exist_ok=True)

    id2name = {
        0: "q_proj",
        1: "k_proj",
        2: "v_proj",
        3: "o_proj",
        4: "gate_proj",
        5: "up_proj",
        6: "down_proj",
    }

    # ===============================
    # ===============================
    if not hasattr(torch, "_loraall_diag_stats"):
        print("[LORA-ALL] No detailed diagnostics found.")
        return

    st = torch._loraall_diag_stats
    token_records = []

    keys = sorted(st["count_tokens"].keys(), key=lambda x: (x[0], x[1]))
    for key in keys:
        layer_id, module_id = key
        cnt = st["count_tokens"].get(key, 0)
        if cnt <= 0:
            continue

        K = len(st["sum_norm"].get(key, []))
        per_adapter = []
        for i in range(K):
            per_adapter.append({
                "adapter_index": i,
                "norm_mean": _safe_mean(st["sum_norm"][key][i], cnt),
                "norm_std": _safe_std(st["sum_norm"][key][i], st["sum_norm_sq"][key][i], cnt),
                "gate_mean": _safe_mean(st["sum_gate"][key][i], cnt),
                "gate_std": _safe_std(st["sum_gate"][key][i], st["sum_gate_sq"][key][i], cnt),
                "weighted_norm_mean": _safe_mean(st["sum_weighted_norm"][key][i], cnt),
            })

        rec = {
            "layer_id": int(layer_id),
            "module_id": int(module_id),
            "module_name": id2name.get(int(module_id), f"module_{module_id}"),
            "count_tokens": int(cnt),
            "per_adapter": per_adapter,
            "entropy_mean": _safe_mean(st["sum_entropy"].get(key, 0.0), cnt),
            "entropy_std": _safe_std(st["sum_entropy"].get(key, 0.0), st["sum_entropy_sq"].get(key, 0.0), cnt),
            "margin_top1_top2_mean": _safe_mean(st["sum_margin"].get(key, 0.0), cnt),
            "margin_top1_top2_std": _safe_std(st["sum_margin"].get(key, 0.0), st["sum_margin_sq"].get(key, 0.0), cnt),
            "switch_l1_mean": _safe_mean(st["sum_switch_l1"].get(key, 0.0), st["count_switch"].get(key, 0)),
        }

        if K >= 2:
            rec.update({
                "cos_01_mean": _safe_mean(st["sum_cos_01"].get(key, 0.0), cnt),
                "conflict_energy_mean": _safe_mean(st["sum_conflict_energy"].get(key, 0.0), cnt),
                "shared_energy_mean": _safe_mean(st["sum_shared_energy"].get(key, 0.0), cnt),
                "merge_cos_0_mean": _safe_mean(st["sum_merge_cos_0"].get(key, 0.0), cnt),
                "merge_cos_1_mean": _safe_mean(st["sum_merge_cos_1"].get(key, 0.0), cnt),
            })

        token_records.append(rec)

    token_path = os.path.join(trainer.args.output_dir, "loraall_token_diagnostics.json")
    with open(token_path, "w") as f:
        json.dump({
            "num_records": len(token_records),
            "stats": token_records,
        }, f, indent=4)

    print(f"[LORA-ALL] Saved token diagnostics ({len(token_records)} records) -> {token_path}")

    # ===============================
    # ===============================
    by_layer = defaultdict(list)
    for rec in token_records:
        by_layer[rec["layer_id"]].append(rec)

    layer_records = []
    for layer_id in sorted(by_layer.keys()):
        recs = sorted(by_layer[layer_id], key=lambda x: x["module_id"])

        K = len(recs[0]["per_adapter"]) if len(recs) > 0 else 0
        module_winners = []

        per_adapter_gate_mean = [[] for _ in range(K)]
        attn_gate = [[] for _ in range(K)]
        mlp_gate = [[] for _ in range(K)]

        for rec in recs:
            gate_means = [x["gate_mean"] for x in rec["per_adapter"]]
            winner = int(max(range(len(gate_means)), key=lambda i: gate_means[i]))
            module_winners.append({
                "module_id": int(rec["module_id"]),
                "module_name": rec["module_name"],
                "winner_adapter": winner,
                "gate_means": gate_means,
            })

            for i in range(K):
                per_adapter_gate_mean[i].append(gate_means[i])
                if rec["module_id"] <= 3:
                    attn_gate[i].append(gate_means[i])
                else:
                    mlp_gate[i].append(gate_means[i])

        module_split = {
            "num_modules": len(module_winners),
            "winner_counts": {
                str(i): sum(1 for x in module_winners if x["winner_adapter"] == i)
                for i in range(K)
            },
            "module_winners": module_winners,
        }

        per_adapter_stats = []
        for i in range(K):
            values = per_adapter_gate_mean[i]
            per_adapter_stats.append({
                "adapter_index": i,
                "module_gate_mean_avg": float(sum(values) / len(values)) if len(values) > 0 else 0.0,
                "module_gate_mean_min": float(min(values)) if len(values) > 0 else 0.0,
                "module_gate_mean_max": float(max(values)) if len(values) > 0 else 0.0,
                "module_gate_mean_range": float(max(values) - min(values)) if len(values) > 0 else 0.0,
                "attn_gate_mean_avg": float(sum(attn_gate[i]) / len(attn_gate[i])) if len(attn_gate[i]) > 0 else 0.0,
                "mlp_gate_mean_avg": float(sum(mlp_gate[i]) / len(mlp_gate[i])) if len(mlp_gate[i]) > 0 else 0.0,
            })

        layer_records.append({
            "layer_id": int(layer_id),
            "per_adapter_module_summary": per_adapter_stats,
            "module_split_summary": module_split,
        })

    layer_path = os.path.join(trainer.args.output_dir, "loraall_layer_module_summary.json")
    with open(layer_path, "w") as f:
        json.dump({
            "num_layers": len(layer_records),
            "stats": layer_records,
        }, f, indent=4)

    print(f"[LORA-ALL] Saved layer-module summary ({len(layer_records)} layers) -> {layer_path}")

    # ===============================
    # ===============================
    try:
        adapter_names = _infer_adapter_names(real_model)
        if adapter_names is None or len(adapter_names) == 0:
            print("[LORA-ALL] Cannot infer adapter names, skip matrix diagnostics.")
            return

        matrix_records = []
        eps = 1e-6

        for module_name, module in real_model.named_modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue

            layer_id = getattr(module, "_layer_id", None)
            module_id = getattr(module, "_module_id", None)
            if layer_id is None or module_id is None or layer_id < 0:
                continue

            available_adapters = [a for a in adapter_names if a in module.lora_A and a in module.lora_B]
            if len(available_adapters) == 0:
                continue

            per_adapter_stats = []
            op_cache = {}

            for adapter in available_adapters:
                A = module.lora_A[adapter].weight.detach().float()
                B = module.lora_B[adapter].weight.detach().float()
                scaling = float(module.scaling[adapter])

                op = _build_low_rank_operator(A, B, scaling, eps=eps)
                core = op["core"]

                if core.numel() == 0:
                    svals = core.new_zeros((0,))
                else:
                    svals = torch.linalg.svdvals(core)

                op_cache[adapter] = op

                per_adapter_stats.append({
                    "adapter_name": adapter,

                    "left_subspace_dim": int(op["left_basis"].shape[1]),
                    "right_subspace_dim": int(op["right_basis"].shape[1]),

                    "fro_norm": _fro_norm_from_core(op),
                    "spectral_norm": float(svals[0].item()) if svals.numel() > 0 else 0.0,
                    "effective_rank": _effective_rank_from_singular_values(svals),

                    "top1_energy_ratio": _energy_ratio(svals, 1),
                    "top2_energy_ratio": _energy_ratio(svals, 2),
                    "top4_energy_ratio": _energy_ratio(svals, 4),
                })

            rec = {
                "layer_id": int(layer_id),
                "module_id": int(module_id),
                "module_name": id2name.get(int(module_id), f"module_{module_id}"),
                "full_module_name": module_name,
                "per_adapter": per_adapter_stats,
            }

            if len(available_adapters) >= 2:
                a0 = available_adapters[0]
                a1 = available_adapters[1]

                pair_stats = _pairwise_operator_stats(op_cache[a0], op_cache[a1])
                pair_stats.update({
                    "adapter_0": a0,
                    "adapter_1": a1,
                })
                rec["pair_01"] = pair_stats

            matrix_records.append(rec)
            del op_cache

        matrix_path = os.path.join(trainer.args.output_dir, "loraall_matrix_diagnostics.json")
        with open(matrix_path, "w") as f:
            json.dump({
                "num_records": len(matrix_records),
                "stats": matrix_records,
            }, f, indent=4)

        print(f"[LORA-ALL] Saved matrix diagnostics ({len(matrix_records)} records) -> {matrix_path}")

    except Exception as e:
        print(f"[LORA-ALL][ERROR] matrix diagnostics failed: {repr(e)}")
        traceback.print_exc()
        return