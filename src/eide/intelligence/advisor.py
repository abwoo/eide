from __future__ import annotations

from eide.causal.attribution import CausalAttribution
from eide.core.diff import DiffCategory
from eide.core.ir import ExperimentIR
from eide.diff.engine import DiffEngine
from eide.graph.builder import ExperimentGraph
from eide.graph.queries import GraphQuery
from eide.storage.filestore import FileStore


class ExperimentAdvisor:
    def __init__(self, store: FileStore | None = None):
        self.store = store
        self.engine = DiffEngine()
        self.causal = CausalAttribution(store=store)

    def compare(self, exp_a: ExperimentIR, exp_b: ExperimentIR) -> ComparisonInsight:
        """Full comparison with explanation and recommendations."""
        report = self.engine.diff(exp_a, exp_b)

        explanation = self.causal.explain(exp_a, exp_b)

        actions = self._generate_actions(report, exp_a, exp_b)

        return ComparisonInsight(
            experiment_a=exp_a.id,
            experiment_b=exp_b.id,
            diff_summary=report.summary,
            changes=report.changes,
            likely_causes=explanation["likely_cause"],
            impact_estimates=report.impact_estimates,
            recommended_actions=actions,
        )

    def recommend_baseline(self, target: ExperimentIR) -> BaselineRecommendation:
        """Recommend the best baseline experiment for comparison."""
        if not self.store:
            return BaselineRecommendation(target_id=target.id, best_match=None)

        from eide.causal.attribution import CausalAttribution

        causal = CausalAttribution(store=self.store)
        baselines = causal.find_baseline(target, max_results=5)

        if not baselines:
            return BaselineRecommendation(
                target_id=target.id,
                best_match=None,
                message="No baseline experiments found in the store.",
            )

        best = baselines[0]
        best_exp = self.store.load(best["experiment_id"])

        report = self.engine.diff(target, best_exp)
        significant_changes = [c for c in report.changes if c.significance > 0.3]

        return BaselineRecommendation(
            target_id=target.id,
            best_match={
                "experiment_id": best["experiment_id"],
                "name": best.get("name", ""),
                "similarity": best["similarity"],
            },
            significant_differences=significant_changes,
        )

    def assess_risk(self, experiment: ExperimentIR) -> RiskAssessment:
        """Assess risk of an experiment based on historical patterns."""
        if not self.store:
            return RiskAssessment(experiment_id=experiment.id, risk_level="unknown")

        try:
            eg = ExperimentGraph.build(self.store)
        except Exception:
            return RiskAssessment(experiment_id=experiment.id, risk_level="unknown")

        q = GraphQuery(eg)
        risks = []

        if "replay_of" in experiment.tags:
            risks.append({
                "type": "replay",
                "severity": "info",
                "message": "This is a replay of another experiment",
            })

        for key, value in experiment.parameters.items():
            impact = q.parameter_impact(key)
            if len(impact) > 1:
                risks.append({
                    "type": "parameter_volatility",
                    "severity": "warning" if len(impact) > 2 else "info",
                    "message": f"Parameter '{key}' has {len(impact)} different values across experiments",
                    "parameter": key,
                    "values_count": len(impact),
                })

        risk_level = "low"
        warnings = [r for r in risks if r["severity"] == "warning"]
        if len(warnings) >= 3:
            risk_level = "high"
        elif len(warnings) >= 1:
            risk_level = "medium"

        return RiskAssessment(
            experiment_id=experiment.id,
            risk_level=risk_level,
            risks=risks,
        )

    def _generate_actions(self, report, exp_a, exp_b) -> list[dict]:
        actions = []
        metric_changes = [c for c in report.changes if c.category == DiffCategory.OUTPUT
                         and c.key.startswith("metrics.")]
        param_changes = [c for c in report.changes if c.category == DiffCategory.PARAMETER]

        if metric_changes and param_changes:
            top_metric = max(metric_changes, key=lambda c: abs(c.significance))
            top_param = max(param_changes, key=lambda c: abs(c.significance))
            key_metric = top_metric.key.replace("metrics.", "")
            actions.append({
                "type": "investigate",
                "priority": "high",
                "message": f"Metric '{key_metric}' changed by {abs(top_metric.new_value - top_metric.old_value):.4f}. "
                           f"Likely related to parameter '{top_param.key}' change.",
            })

        if metric_changes:
            for mc in metric_changes:
                key = mc.key.replace("metrics.", "")
                actions.append({
                    "type": "verify",
                    "priority": "medium",
                    "message": f"Verify if '{key}' change ({mc.old_value} -> {mc.new_value}) is expected.",
                })

        if not actions:
            actions.append({
                "type": "ok",
                "priority": "low",
                "message": "No significant changes detected.",
            })

        return actions


class ComparisonInsight:
    def __init__(self, experiment_a: str, experiment_b: str, diff_summary: str,
                 changes: list, likely_causes: list, impact_estimates: list,
                 recommended_actions: list):
        self.experiment_a = experiment_a
        self.experiment_b = experiment_b
        self.diff_summary = diff_summary
        self.changes = changes
        self.likely_causes = likely_causes
        self.impact_estimates = impact_estimates
        self.recommended_actions = recommended_actions

    def to_dict(self) -> dict:
        return {
            "experiment_a": self.experiment_a,
            "experiment_b": self.experiment_b,
            "diff_summary": self.diff_summary,
            "likely_causes": self.likely_causes,
            "impact_estimates": [ie.model_dump() for ie in self.impact_estimates],
            "recommended_actions": self.recommended_actions,
        }


class BaselineRecommendation:
    def __init__(self, target_id: str, best_match: dict | None = None,
                 significant_differences: list | None = None, message: str = ""):
        self.target_id = target_id
        self.best_match = best_match
        self.significant_differences = significant_differences or []
        self.message = message


class RiskAssessment:
    def __init__(self, experiment_id: str, risk_level: str = "low",
                 risks: list[dict] | None = None):
        self.experiment_id = experiment_id
        self.risk_level = risk_level
        self.risks = risks or []
