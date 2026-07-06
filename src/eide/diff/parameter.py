from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from eide.core.diff import ChangeType, DiffCategory, DiffChange


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k, v in sorted(d.items()):
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, Mapping) and not isinstance(v, (str, bytes)):
            result.update(_flatten_dict(v, key))
        else:
            result[key] = v
    return result


def diff_parameters(
    params_a: dict[str, Any],
    params_b: dict[str, Any],
    tolerances: dict[str, float] | None = None,
) -> list[DiffChange]:
    tolerances = tolerances or {}
    flat_a = _flatten_dict(params_a)
    flat_b = _flatten_dict(params_b)
    changes: list[DiffChange] = []

    all_keys = set(flat_a.keys()) | set(flat_b.keys())

    for key in sorted(all_keys):
        val_a = flat_a.get(key)
        val_b = flat_b.get(key)

        if key not in flat_a:
            changes.append(
                DiffChange(
                    category=DiffCategory.PARAMETER,
                    change_type=ChangeType.ADDED,
                    key=key,
                    old_value=None,
                    new_value=val_b,
                    description=f"Parameter added: {key} = {val_b}",
                )
            )
        elif key not in flat_b:
            changes.append(
                DiffChange(
                    category=DiffCategory.PARAMETER,
                    change_type=ChangeType.REMOVED,
                    key=key,
                    old_value=val_a,
                    new_value=None,
                    description=f"Parameter removed: {key} (was {val_a})",
                )
            )
        else:
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                tol = tolerances.get(key, 1e-3)
                if not math.isclose(val_a, val_b, rel_tol=tol):
                    pct = _safe_pct(val_b - val_a, val_a)
                    changes.append(
                        DiffChange(
                            category=DiffCategory.PARAMETER,
                            change_type=ChangeType.MODIFIED,
                            key=key,
                            old_value=val_a,
                            new_value=val_b,
                            significance=min(abs(pct) / 100.0, 1.0),
                            description=_format_num_change(key, val_a, val_b, pct),
                        )
                    )
            elif val_a != val_b:
                changes.append(
                    DiffChange(
                        category=DiffCategory.PARAMETER,
                        change_type=ChangeType.MODIFIED,
                        key=key,
                        old_value=val_a,
                        new_value=val_b,
                        significance=0.8,
                        description=f"Parameter changed: {key}: {val_a} -> {val_b}",
                    )
                )

    return changes


def _safe_pct(delta: float, reference: float) -> float:
    if reference == 0:
        return 0.0
    return (delta / abs(reference)) * 100.0


def _format_num_change(key: str, old: float, new: float, pct: float) -> str:
    direction = "↑" if new > old else "↓"
    return f"Parameter {key}: {old} -> {new} {direction} ({pct:+.2f}%)"
