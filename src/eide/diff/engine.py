from __future__ import annotations

from pathlib import Path

from eide.core.diff import ChangeType, DiffCategory, DiffChange, DiffReport
from eide.core.ir import ExperimentIR
from eide.diff.environment import diff_environment
from eide.diff.parameter import diff_parameters
from eide.diff.semantic import diff_metric_history, diff_outputs
from eide.diff.structural import diff_pipeline


class DiffEngine:
    def __init__(self, artifacts_root: str | Path | None = None):
        self.artifacts_root = Path(artifacts_root) if artifacts_root else None

    def diff(
        self,
        exp_a: ExperimentIR,
        exp_b: ExperimentIR,
        parameter_tolerances: dict[str, float] | None = None,
    ) -> DiffReport:
        changes = []

        changes.extend(diff_parameters(exp_a.parameters, exp_b.parameters, parameter_tolerances))
        changes.extend(diff_pipeline(exp_a.pipeline, exp_b.pipeline))
        changes.extend(diff_environment(exp_a.environment, exp_b.environment))
        changes.extend(diff_outputs(exp_a.outputs, exp_b.outputs, self.artifacts_root))
        changes.extend(
            diff_metric_history(exp_a.outputs.metric_history, exp_b.outputs.metric_history)
        )

        for key in sorted(set(exp_a.data_versions.keys()) | set(exp_b.data_versions.keys())):
            va = exp_a.data_versions.get(key)
            vb = exp_b.data_versions.get(key)
            if va != vb:
                changes.append(
                    DiffChange(
                        category=DiffCategory.DATA_VERSION,
                        change_type=ChangeType.MODIFIED,
                        key=f"data_versions.{key}",
                        old_value=va,
                        new_value=vb,
                        significance=0.7,
                        description=f"Data version {key}: {va} -> {vb}",
                    )
                )

        for key in sorted(set(exp_a.tags.keys()) | set(exp_b.tags.keys())):
            va = exp_a.tags.get(key)
            vb = exp_b.tags.get(key)
            if va != vb:
                ct = (
                    ChangeType.ADDED
                    if key not in exp_a.tags
                    else ChangeType.REMOVED
                    if key not in exp_b.tags
                    else ChangeType.MODIFIED
                )
                changes.append(
                    DiffChange(
                        category=DiffCategory.PARAMETER,
                        change_type=ct,
                        key=f"tags.{key}",
                        old_value=va,
                        new_value=vb,
                        significance=0.3,
                        description=f"Tag {key}: {va} -> {vb}"
                        if va and vb
                        else f"Tag {ct.value}: {key}={vb or va}",
                    )
                )

        summary_lines = _build_summary(changes)
        changelog_lines = _build_changelog(changes)

        report = DiffReport(
            experiment_a=exp_a.id,
            experiment_b=exp_b.id,
            changes=changes,
            summary="\n".join(summary_lines),
            changelog=changelog_lines,
        )

        report.impact_estimates = _estimate_impacts(changes)
        return report


def _build_summary(changes: list) -> list[str]:
    if not changes:
        return ["No differences found."]

    counts: dict[str, int] = {}
    for c in changes:
        counts[c.category.value] = counts.get(c.category.value, 0) + 1

    lines = [f"Found {len(changes)} difference(s):"]
    for cat, cnt in sorted(counts.items()):
        lines.append(f"  {cat}: {cnt}")
    return lines


def _build_changelog(changes: list) -> list[str]:
    return [c.description for c in changes if c.description]


def _estimate_impacts(changes: list) -> list:
    from eide.core.diff import ImpactEstimate

    metric_changes = [
        c for c in changes if c.category == DiffCategory.OUTPUT and c.key.startswith("metrics.")
    ]
    param_changes = [c for c in changes if c.category == DiffCategory.PARAMETER]
    struct_changes = [c for c in changes if c.category == DiffCategory.STRUCTURAL]
    env_changes = [c for c in changes if c.category == DiffCategory.ENVIRONMENT]
    data_changes = [c for c in changes if c.category == DiffCategory.DATA_VERSION]

    if not metric_changes:
        return []

    estimates = []
    for mc in metric_changes:
        metric_key = mc.key.replace("metrics.", "")
        old = mc.old_value or 0
        new = mc.new_value or 0
        change_percent = abs((new - old) / (old or 1)) * 100
        direction = "increase" if new > old else "decrease"

        likely_cause = ""
        confidence = 0.0
        correlation_reasons = []

        for pc in param_changes:
            if pc.significance > 0.3:
                if "lr" in pc.key.lower() or "learning" in pc.key.lower():
                    if metric_key in ("accuracy", "loss", "f1"):
                        correlation_reasons.append(
                            (pc, 0.7, f"Learning rate change ({pc.description[:40]})")
                        )
                elif "batch_size" in pc.key.lower():
                    if metric_key in ("accuracy", "loss"):
                        correlation_reasons.append(
                            (pc, 0.5, f"Batch size change ({pc.description[:40]})")
                        )
                elif "epoch" in pc.key.lower():
                    correlation_reasons.append(
                        (pc, 0.4, f"Epoch count change ({pc.description[:40]})")
                    )
                else:
                    correlation_reasons.append((pc, 0.35, f"Parameter '{pc.key}' changed"))

        for sc in struct_changes:
            correlation_reasons.append((sc, 0.6, "Pipeline structure changed"))

        for ec in env_changes:
            if "python" in ec.key.lower() or "packages" in ec.key.lower():
                correlation_reasons.append((ec, 0.5, f"Environment change ({ec.description[:50]})"))

        for dc in data_changes:
            correlation_reasons.append((dc, 0.7, "Data version changed"))

        if mc.significance > 0.3:
            correlation_reasons.append(
                (mc, mc.significance * 0.3, "Direct metric change magnitude")
            )

        if correlation_reasons:
            best = max(correlation_reasons, key=lambda x: x[1])
            likely_cause = best[2]
            confidence = min(best[1] + (mc.significance * 0.3), 0.95)
            if len(correlation_reasons) > 1:
                second = sorted(correlation_reasons, key=lambda x: -x[1])[1]
                if second[1] > 0.3:
                    confidence = min(confidence + 0.1, 0.98)

        estimates.append(
            ImpactEstimate(
                metric=metric_key,
                change_direction=direction,
                change_percent=round(change_percent, 2),
                likely_cause=likely_cause or "No clear cause identified",
                confidence=max(round(confidence, 2), 0.05),
            )
        )
    return estimates


def diff_experiments(
    exp_a: ExperimentIR,
    exp_b: ExperimentIR,
    artifacts_root: str | Path | None = None,
    parameter_tolerances: dict[str, float] | None = None,
) -> DiffReport:
    """Convenience function: diff two experiments"""
    engine = DiffEngine(artifacts_root=artifacts_root)
    return engine.diff(exp_a, exp_b, parameter_tolerances)
