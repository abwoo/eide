from __future__ import annotations

from eide.core.diff import DiffCategory, DiffReport
from eide.core.ir import ExperimentIR
from eide.diff.engine import DiffEngine
from eide.storage.filestore import FileStore


class CausalAttribution:
    def __init__(self, store: FileStore | None = None):
        self.store = store
        self._engine = DiffEngine()

    def explain(
        self,
        exp_a: ExperimentIR,
        exp_b: ExperimentIR,
    ) -> dict:
        report = self._engine.diff(exp_a, exp_b)
        return self._build_explanation(report, exp_a, exp_b)

    def find_baseline(
        self,
        target: ExperimentIR,
        max_results: int = 5,
    ) -> list[dict]:
        if not self.store:
            return []
        candidates = []
        for entry in self.store.list_experiments():
            if entry["id"] == target.id:
                continue
            try:
                candidate = self.store.load(entry["id"])
            except Exception:
                continue

            report = self._engine.diff(target, candidate)
            similarity = self._compute_similarity(report)
            candidates.append({
                "experiment_id": candidate.id,
                "name": candidate.name,
                "similarity": similarity,
                "change_count": len(report.changes),
                "parameter_diff_count": len(
                    [c for c in report.changes if c.category == DiffCategory.PARAMETER]
                ),
            })

        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:max_results]

    def _build_explanation(
        self,
        report: DiffReport,
        exp_a: ExperimentIR,
        exp_b: ExperimentIR,
    ) -> dict:
        param_changes = [c for c in report.changes if c.category == DiffCategory.PARAMETER]
        struct_changes = [c for c in report.changes if c.category == DiffCategory.STRUCTURAL]
        env_changes = [c for c in report.changes if c.category == DiffCategory.ENVIRONMENT]
        metric_changes = [c for c in report.changes if c.category == DiffCategory.OUTPUT and c.key.startswith("metrics.")]

        explanation = {
            "experiment_a": exp_a.id,
            "experiment_b": exp_b.id,
            "summary": report.summary,
            "changelog": report.changelog,
            "change_statistics": {
                "total": len(report.changes),
                "parameters": len(param_changes),
                "structural": len(struct_changes),
                "environment": len(env_changes),
                "metrics": len(metric_changes),
            },
            "likely_cause": self._rank_causes(param_changes, struct_changes, env_changes, metric_changes),
        }
        return explanation

    def _rank_causes(self, param_changes: list, struct_changes: list, env_changes: list, metric_changes: list) -> list[dict]:
        causes = []

        if param_changes:
            avg_sig = sum(c.significance for c in param_changes) / max(len(param_changes), 1)
            causes.append({
                "cause": "parameter_change",
                "confidence": min(avg_sig + 0.2, 1.0),
                "count": len(param_changes),
                "reasoning": f"{len(param_changes)} parameter(s) changed",
            })

        if struct_changes:
            causes.append({
                "cause": "pipeline_change",
                "confidence": 0.6,
                "count": len(struct_changes),
                "reasoning": f"Pipeline structure changed ({len(struct_changes)} difference(s))",
            })

        if env_changes:
            causes.append({
                "cause": "environment_change",
                "confidence": 0.4,
                "count": len(env_changes),
                "reasoning": f"Environment changed ({len(env_changes)} difference(s))",
            })

        metric_impact = len(metric_changes) > 0
        if metric_impact:
            for cause in causes:
                cause["confidence"] = min(cause["confidence"] + 0.15, 1.0)
                cause["reasoning"] += " (metric impact confirmed)"

        causes.sort(key=lambda x: x["confidence"], reverse=True)
        return causes

    def _compute_similarity(self, report: DiffReport) -> float:
        total = len(report.changes)
        if total == 0:
            return 1.0
        param_weight = sum(0.3 for c in report.changes if c.category == DiffCategory.PARAMETER)
        struct_weight = sum(0.5 for c in report.changes if c.category == DiffCategory.STRUCTURAL)
        env_weight = sum(0.2 for c in report.changes if c.category == DiffCategory.ENVIRONMENT)
        weighted = param_weight + struct_weight + env_weight
        return max(0.0, 1.0 - (weighted / max(total, 1)))
