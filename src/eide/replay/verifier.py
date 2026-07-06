from __future__ import annotations

from eide.core.diff import DiffReport
from eide.core.ir import ExperimentIR
from eide.diff.engine import DiffEngine


class ReplayVerifier:
    def __init__(self, artifacts_root: str | None = None):
        self.engine = DiffEngine(artifacts_root=artifacts_root)

    def verify(
        self,
        original: ExperimentIR,
        replay_result: ExperimentIR,
    ) -> ReplayVerdict:
        report = self.engine.diff(original, replay_result)

        output_changes = [c for c in report.changes if c.category.value in ("output", "semantic")]
        param_changes = [c for c in report.changes if c.category.value == "parameter"]
        env_changes = [c for c in report.changes if c.category.value == "environment"]
        struct_changes = [c for c in report.changes if c.category.value == "structural"]

        is_identical = len(report.changes) == 0

        return ReplayVerdict(
            original_id=original.id,
            replay_id=replay_result.id,
            is_identical=is_identical,
            total_changes=len(report.changes),
            output_changes=len(output_changes),
            param_changes=len(param_changes),
            env_changes=len(env_changes),
            struct_changes=len(struct_changes),
            report=report,
        )


class ReplayVerdict:
    def __init__(
        self,
        original_id: str,
        replay_id: str,
        is_identical: bool,
        total_changes: int,
        output_changes: int,
        param_changes: int,
        env_changes: int,
        struct_changes: int,
        report: DiffReport,
    ):
        self.original_id = original_id
        self.replay_id = replay_id
        self.is_identical = is_identical
        self.total_changes = total_changes
        self.output_changes = output_changes
        self.param_changes = param_changes
        self.env_changes = env_changes
        self.struct_changes = struct_changes
        self.report = report

    @property
    def diverged(self) -> bool:
        return not self.is_identical

    @property
    def summary(self) -> str:
        if self.is_identical:
            return "✅ Replay produced identical output"
        divergence_causes = []
        if self.output_changes:
            divergence_causes.append(f"{self.output_changes} output change(s)")
        if self.env_changes:
            divergence_causes.append(f"{self.env_changes} environment change(s)")
        if self.struct_changes:
            divergence_causes.append(f"{self.struct_changes} structural change(s)")
        causes = ", ".join(divergence_causes) or "unknown"
        return f"❌ Replay diverged: {causes}"
