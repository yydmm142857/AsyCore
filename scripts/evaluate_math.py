#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate math accuracy from LLaMA-Factory generated_predictions.jsonl.
Rule: extract the last numeric answer from prediction and label, normalize with Decimal, compare numeric equality.
Outputs:
  new_acc.json
  predict_label.json
  updates all_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from decimal import Decimal, InvalidOperation, getcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

getcontext().prec = 80

NUMBER_PATTERN = re.compile(
    r"""
    (?<![\w.])
    [+-]?
    (?:
        \d+\s*/\s*\d+
        |\d+(?:,\d{3})*(?:\.\d+)?
        |\.\d+
    )
    (?:[eE][+-]?\d+)?
    (?![\w/])
    """,
    re.VERBOSE,
)
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")

LABEL_KEYS = ["label", "labels", "gold", "answer", "reference", "target", "ground_truth", "output"]
PRED_KEYS = ["predict", "prediction", "pred", "generated", "generated_text", "response", "output_text"]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(x) for x in value)
    return str(value)


def read_rows(path: Path) -> List[Tuple[int, Dict[str, Any]]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    rows: List[Tuple[int, Dict[str, Any]]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    obj = {"predict": obj}
                rows.append((idx, obj))
    else:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for k in ["predictions", "data", "results", "items"]:
                if isinstance(obj.get(k), list):
                    obj = obj[k]
                    break
        if isinstance(obj, list):
            for idx, item in enumerate(obj, start=1):
                rows.append((idx, item if isinstance(item, dict) else {"predict": item}))
        elif isinstance(obj, dict):
            rows.append((1, obj))
        else:
            raise ValueError(f"Unsupported JSON type: {type(obj)}")
    return rows


def first_key(obj: Dict[str, Any], candidates: Iterable[str], fuzzy: Iterable[str] = ()) -> Optional[str]:
    for k in candidates:
        if k in obj:
            return k
    low = {k.lower(): k for k in obj.keys()}
    for token in fuzzy:
        for lk, real in low.items():
            if token in lk:
                return real
    return None


def to_decimal(num: str) -> Optional[Decimal]:
    if not num:
        return None
    s = str(num).strip().strip("$\uffe5\u5143\u4e2a\uff0c\u3002\uff1b;:\uff1a!?\uff01\uff1f")
    s = s.replace(",", "").replace("\uff0c", "").replace(" ", "")
    try:
        if "/" in s and "e" not in s.lower() and "." not in s:
            frac = Fraction(s)
            return Decimal(frac.numerator) / Decimal(frac.denominator)
        return Decimal(s)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def decimal_to_str(x: Optional[Decimal]) -> str:
    if x is None:
        return ""
    if x == 0:
        return "0"
    s = format(x.normalize(), "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def extract_last_number(text: Any) -> Tuple[str, str, Optional[Decimal]]:
    s = normalize_text(text)
    # Prefer numbers inside the last \boxed{...}, then fallback to all numbers.
    boxed = BOXED_PATTERN.findall(s)
    candidates: List[str] = []
    for b in boxed:
        candidates.extend(m.group(0) for m in NUMBER_PATTERN.finditer(b))
    if not candidates:
        candidates = [m.group(0) for m in NUMBER_PATTERN.finditer(s)]
    raw = candidates[-1] if candidates else ""
    val = to_decimal(raw)
    return raw, decimal_to_str(val) if val is not None else raw, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", required=True, help="generated_predictions.jsonl/json")
    ap.add_argument("--out_dir", default=None, help="default: pred_file parent")
    ap.add_argument("--rtol", type=str, default="0", help="relative tolerance, default exact Decimal equality")
    ap.add_argument("--atol", type=str, default="0", help="absolute tolerance, default exact Decimal equality")
    args = ap.parse_args()

    pred_path = Path(args.pred_file).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(pred_path)
    total = correct = valid_pairs = 0
    details: List[Dict[str, Any]] = []
    pairs: List[Dict[str, Any]] = []
    atol = Decimal(args.atol)
    rtol = Decimal(args.rtol)

    for line_no, obj in rows:
        label_key = first_key(obj, LABEL_KEYS, fuzzy=["label", "target", "answer", "gold", "reference"])
        pred_key = first_key(obj, PRED_KEYS, fuzzy=["predict", "pred", "generate", "response"])
        if label_key is None:
            raise KeyError(f"Line {line_no} has no label field; keys={list(obj.keys())}")
        if pred_key is None:
            raise KeyError(f"Line {line_no} has no prediction field; keys={list(obj.keys())}")

        label_text = normalize_text(obj.get(label_key))
        pred_text = normalize_text(obj.get(pred_key))
        pred_raw, pred_norm, pred_val = extract_last_number(pred_text)
        label_raw, label_norm, label_val = extract_last_number(label_text)

        ok = False
        if pred_val is not None and label_val is not None:
            valid_pairs += 1
            diff = abs(pred_val - label_val)
            tol = atol + rtol * max(abs(label_val), Decimal("1"))
            ok = diff <= tol
        total += 1
        correct += int(ok)

        rec = {
            "line_no": line_no,
            "label_key": label_key,
            "predict_key": pred_key,
            "label_text": label_text,
            "predict_text": pred_text,
            "label_raw": label_raw,
            "predict_raw": pred_raw,
            "label": label_norm,
            "predict": pred_norm,
            "correct": int(ok),
        }
        details.append(rec)
        pairs.append({k: rec[k] for k in ["label_raw", "predict_raw", "label", "predict", "correct"]})

    result = {
        "source_file": str(pred_path),
        "total": total,
        "correct": correct,
        "acc": correct / total if total else 0.0,
        "acc_percent": 100 * correct / total if total else 0.0,
        "valid_pairs": valid_pairs,
        "invalid_pairs": total - valid_pairs,
        "rule": "extract final numeric answer from predict and label, normalize using Decimal/Fraction, compare with tolerance",
        "atol": str(atol),
        "rtol": str(rtol),
        "details": details,
    }

    (out_dir / "new_acc.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "predict_label.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    all_path = out_dir / "all_results.json"
    try:
        all_obj = json.loads(all_path.read_text(encoding="utf-8")) if all_path.exists() else {}
        if not isinstance(all_obj, dict):
            all_obj = {"old_all_results": all_obj}
    except Exception:
        all_obj = {}
    all_obj.update({
        "math_acc": result["acc"],
        "math_acc_percent": result["acc_percent"],
        "math_correct": correct,
        "math_total": total,
        "math_valid_pairs": valid_pairs,
        "math_invalid_pairs": total - valid_pairs,
    })
    all_path.write_text(json.dumps(all_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "total": total,
        "correct": correct,
        "acc": result["acc"],
        "acc_percent": result["acc_percent"],
        "valid_pairs": valid_pairs,
        "invalid_pairs": total - valid_pairs,
        "out_dir": str(out_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
