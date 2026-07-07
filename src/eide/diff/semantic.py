from __future__ import annotations

import hashlib
from pathlib import Path

from eide.core.diff import ChangeType, DiffCategory, DiffChange
from eide.core.types import ExperimentOutputs


def _compute_statistical_significance(
    val_a: float,
    val_b: float,
    std_a: float | None = None,
    std_b: float | None = None,
    n_a: int = 1,
    n_b: int = 1,
) -> dict:
    """Compute statistical significance metrics between two values.

    Returns dict with: effect_size (Cohen's d), p_value (estimated),
    significance_score, test_type, meaningful.
    """
    raw_diff = abs(val_b - val_a)
    avg = (abs(val_a) + abs(val_b)) / 2
    rel_change = raw_diff / avg if avg > 0 else 0

    if std_a is not None and std_b is not None and n_a > 0 and n_b > 0:
        pooled_std = (((n_a - 1) * (std_a**2)) + ((n_b - 1) * (std_b**2))) / (n_a + n_b - 2)
        pooled_std = max(pooled_std, 0.001) ** 0.5
        cohens_d = raw_diff / pooled_std

        try:
            from scipy.stats import ttest_ind_from_stats

            t_stat, p_value = ttest_ind_from_stats(
                mean1=val_a,
                std1=std_a,
                nobs1=n_a,
                mean2=val_b,
                std2=std_b,
                nobs2=n_b,
            )
        except Exception:
            p_value = max(0.001, 1.0 - rel_change)
    else:
        cohens_d = rel_change * 3.0
        p_value = max(0.001, 1.0 - rel_change)

    effect_size = min(abs(cohens_d), 2.0)
    significance_score = min(effect_size / 1.5, 1.0)
    meaningful = effect_size > 0.2 and p_value < 0.05

    return {
        "effect_size": round(effect_size, 3),
        "p_value": min(p_value, 1.0),
        "significance_score": round(significance_score, 3),
        "test_type": "cohens_d+ttest" if std_a is not None else "heuristic",
        "meaningful": bool(meaningful),
    }


def diff_metric_history(
    history_a: dict[str, list],
    history_b: dict[str, list],
) -> list[DiffChange]:
    changes: list[DiffChange] = []
    all_keys = set(history_a.keys()) | set(history_b.keys())
    for key in sorted(all_keys):
        ha = history_a.get(key, [])
        hb = history_b.get(key, [])
        if ha and hb:
            final_a = ha[-1]["value"] if isinstance(ha[-1], dict) else ha[-1].value
            final_b = hb[-1]["value"] if isinstance(hb[-1], dict) else hb[-1].value
            if len(ha) != len(hb) or final_a != final_b:
                changes.append(
                    DiffChange(
                        category=DiffCategory.SEMANTIC,
                        change_type=ChangeType.MODIFIED,
                        key=f"metric_history.{key}",
                        old_value=f"{len(ha)} points, final={final_a}",
                        new_value=f"{len(hb)} points, final={final_b}",
                        significance=0.5,
                        description=(
                            f"Metric '{key}' history: {len(ha)} pts -> {len(hb)} pts"
                            f", final {final_a} -> {final_b}"
                        ),
                    )
                )
        elif ha and not hb:
            changes.append(
                DiffChange(
                    category=DiffCategory.OUTPUT,
                    change_type=ChangeType.REMOVED,
                    key=f"metric_history.{key}",
                    old_value=f"{len(ha)} points",
                    significance=0.5,
                    description=f"Metric history '{key}' removed ({len(ha)} points)",
                )
            )
        elif not ha and hb:
            changes.append(
                DiffChange(
                    category=DiffCategory.OUTPUT,
                    change_type=ChangeType.ADDED,
                    key=f"metric_history.{key}",
                    new_value=f"{len(hb)} points",
                    significance=0.5,
                    description=f"Metric history '{key}' added ({len(hb)} points)",
                )
            )
    return changes


def diff_metrics(
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    stds_a: dict[str, float] | None = None,
    stds_b: dict[str, float] | None = None,
    counts_a: dict[str, int] | None = None,
    counts_b: dict[str, int] | None = None,
) -> list[DiffChange]:
    changes: list[DiffChange] = []
    all_keys = set(metrics_a.keys()) | set(metrics_b.keys())

    for key in sorted(all_keys):
        val_a = metrics_a.get(key)
        val_b = metrics_b.get(key)

        if key not in metrics_a:
            changes.append(
                DiffChange(
                    category=DiffCategory.OUTPUT,
                    change_type=ChangeType.ADDED,
                    key=f"metrics.{key}",
                    old_value=None,
                    new_value=val_b,
                    significance=0.6,
                    description=f"Metric added: {key} = {val_b}",
                )
            )
        elif key not in metrics_b:
            changes.append(
                DiffChange(
                    category=DiffCategory.OUTPUT,
                    change_type=ChangeType.REMOVED,
                    key=f"metrics.{key}",
                    old_value=val_a,
                    new_value=None,
                    significance=0.6,
                    description=f"Metric removed: {key} (was {val_a})",
                )
            )
        elif val_a is not None and val_b is not None and val_a != val_b:
            pct = _safe_pct(val_b - val_a, val_a)
            direction = "↑" if val_b > val_a else "↓"

            std_a = (stds_a or {}).get(key)
            std_b = (stds_b or {}).get(key)
            n_a = (counts_a or {}).get(key, 1)
            n_b = (counts_b or {}).get(key, 1)

            stats = _compute_statistical_significance(val_a, val_b, std_a, std_b, n_a, n_b)
            significance = max(stats["significance_score"], min(abs(pct) / 100.0, 1.0))
            significance = min(significance, 1.0)

            extra = ""
            if stats["meaningful"]:
                extra = " [statistically significant]"
            elif stats["effect_size"] > 0.2:
                extra = f" [d={stats['effect_size']:.2f}]"

            changes.append(
                DiffChange(
                    category=DiffCategory.OUTPUT,
                    change_type=ChangeType.MODIFIED,
                    key=f"metrics.{key}",
                    old_value=val_a,
                    new_value=val_b,
                    significance=significance,
                    description=(
                        f"Metric {key}: {val_a} -> {val_b} {direction} ({pct:+.2f}%){extra}"
                    ),
                )
            )

    return changes


def _safe_pct(delta: float, reference: float) -> float:
    if reference == 0:
        return 0.0
    return (delta / abs(reference)) * 100.0


def diff_figures(
    figures_a: list[str],
    figures_b: list[str],
    output_dir: str | Path | None = None,
) -> list[DiffChange]:
    changes: list[DiffChange] = []
    all_figs = set(figures_a) | set(figures_b)

    for fig in sorted(all_figs):
        in_a = fig in figures_a
        in_b = fig in figures_b

        if not in_a:
            changes.append(
                DiffChange(
                    category=DiffCategory.OUTPUT,
                    change_type=ChangeType.ADDED,
                    key=f"figures.{fig}",
                    description=f"Figure added: {fig}",
                )
            )
        elif not in_b:
            changes.append(
                DiffChange(
                    category=DiffCategory.OUTPUT,
                    change_type=ChangeType.REMOVED,
                    key=f"figures.{fig}",
                    description=f"Figure removed: {fig}",
                )
            )
        else:
            if output_dir:
                path_a = Path(output_dir) / fig
                path_b = Path(output_dir) / fig
                if path_a.exists() and path_b.exists():
                    ssim_val = _compute_ssim(path_a, path_b)
                    if ssim_val is not None and ssim_val < 0.95:
                        changes.append(
                            DiffChange(
                                category=DiffCategory.SEMANTIC,
                                change_type=ChangeType.MODIFIED,
                                key=f"figures.{fig}",
                                old_value="SSIM=1.0 (original)",
                                new_value=f"SSIM={ssim_val:.4f}",
                                significance=1.0 - ssim_val,
                                description=(
                                    f"Figure {fig}: SSIM={ssim_val:.4f}"
                                    f" (visually {'similar' if ssim_val > 0.7 else 'different'})"
                                ),
                            )
                        )
            else:
                hash_a = _file_hash(fig)
                hash_b = _file_hash(fig)
                if hash_a != hash_b:
                    changes.append(
                        DiffChange(
                            category=DiffCategory.SEMANTIC,
                            change_type=ChangeType.MODIFIED,
                            key=f"figures.{fig}",
                            description=f"Figure {fig}: content changed (hash mismatch)",
                        )
                    )

    return changes


def _file_hash(path: str | Path) -> str:
    try:
        data = Path(path).read_bytes()
        return hashlib.sha256(data).hexdigest()[:16]
    except Exception:
        return ""


def _compute_ssim(path_a: Path, path_b: Path) -> float | None:
    try:
        from skimage.io import imread
        from skimage.metrics import structural_similarity as ssim

        img_a = imread(path_a)
        img_b = imread(path_b)

        if img_a.shape != img_b.shape:
            from skimage.transform import resize

            h, w = min(img_a.shape[0], img_b.shape[0]), min(img_a.shape[1], img_b.shape[1])
            img_a = resize(img_a, (h, w), anti_aliasing=True)
            img_b = resize(img_b, (h, w), anti_aliasing=True)
            if img_a.ndim == 3 and img_b.ndim == 3 and img_a.shape[2] != img_b.shape[2]:
                img_a = img_a[:, :, :3]
                img_b = img_b[:, :, :3]

        if img_a.ndim == 3:
            img_a = img_a[:, :, :3]
            img_b = img_b[:, :, :3]

        multichannel = img_a.ndim == 3
        return float(
            ssim(
                img_a,
                img_b,
                channel_axis=-1 if multichannel else None,
                data_range=img_a.max() - img_a.min() or 1.0,
            )
        )
    except Exception:
        return None


def diff_outputs(
    outputs_a: ExperimentOutputs,
    outputs_b: ExperimentOutputs,
    artifacts_root: str | Path | None = None,
    stds_a: dict[str, float] | None = None,
    stds_b: dict[str, float] | None = None,
) -> list[DiffChange]:
    changes: list[DiffChange] = []
    changes.extend(diff_metrics(outputs_a.metrics, outputs_b.metrics, stds_a, stds_b))
    changes.extend(diff_figures(outputs_a.figures, outputs_b.figures, artifacts_root))
    return changes
