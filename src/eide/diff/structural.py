from __future__ import annotations

import difflib

from eide.core.diff import ChangeType, DiffCategory, DiffChange
from eide.core.types import PipelineSpec, PipelineStep


def _step_signature(step: PipelineStep) -> str:
    return f"{step.name}:{step.type}"


def _lcs_align(
    steps_a: list[PipelineStep],
    steps_b: list[PipelineStep],
) -> list[tuple[int | None, int | None]]:
    sigs_a = [_step_signature(s) for s in steps_a]
    sigs_b = [_step_signature(s) for s in steps_b]
    matcher = difflib.SequenceMatcher(None, sigs_a, sigs_b)
    alignment: list[tuple[int | None, int | None]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for idx in range(i2 - i1):
                alignment.append((i1 + idx, j1 + idx))
        elif tag == "replace":
            for idx in range(max(i2 - i1, j2 - j1)):
                a_idx = i1 + idx if idx < i2 - i1 else None
                b_idx = j1 + idx if idx < j2 - j1 else None
                alignment.append((a_idx, b_idx))
        elif tag == "delete":
            for idx in range(i1, i2):
                alignment.append((idx, None))
        elif tag == "insert":
            for idx in range(j1, j2):
                alignment.append((None, idx))
    return alignment


def diff_pipeline(pipeline_a: PipelineSpec, pipeline_b: PipelineSpec) -> list[DiffChange]:
    changes: list[DiffChange] = []

    if pipeline_a.name != pipeline_b.name:
        changes.append(
            DiffChange(
                category=DiffCategory.STRUCTURAL,
                change_type=ChangeType.MODIFIED,
                key="pipeline.name",
                old_value=pipeline_a.name,
                new_value=pipeline_b.name,
                description=f"Pipeline name: {pipeline_a.name} -> {pipeline_b.name}",
            )
        )

    steps_a = pipeline_a.steps
    steps_b = pipeline_b.steps
    alignment = _lcs_align(steps_a, steps_b)

    for a_idx, b_idx in alignment:
        if a_idx is None and b_idx is not None:
            step = steps_b[b_idx]
            changes.append(
                DiffChange(
                    category=DiffCategory.STRUCTURAL,
                    change_type=ChangeType.ADDED,
                    key=f"pipeline.steps[{b_idx}]",
                    old_value=None,
                    new_value=_step_signature(step),
                    description=f"Step added: {_step_signature(step)}",
                )
            )
        elif a_idx is not None and b_idx is None:
            step = steps_a[a_idx]
            changes.append(
                DiffChange(
                    category=DiffCategory.STRUCTURAL,
                    change_type=ChangeType.REMOVED,
                    key=f"pipeline.steps[{a_idx}]",
                    old_value=_step_signature(step),
                    new_value=None,
                    description=f"Step removed: {_step_signature(step)}",
                )
            )
        elif a_idx is not None and b_idx is not None:
            step_a = steps_a[a_idx]
            step_b = steps_b[b_idx]
            step_changes = _diff_steps(step_a, step_b, a_idx)
            changes.extend(step_changes)

    return changes


def _diff_steps(step_a: PipelineStep, step_b: PipelineStep, index: int) -> list[DiffChange]:
    changes: list[DiffChange] = []
    prefix = f"pipeline.steps[{index}]"

    if step_a.type != step_b.type:
        changes.append(
            DiffChange(
                category=DiffCategory.STRUCTURAL,
                change_type=ChangeType.MODIFIED,
                key=f"{prefix}.type",
                old_value=step_a.type,
                new_value=step_b.type,
                description=f"Step type: {step_a.type} -> {step_b.type}",
            )
        )

    if step_a.parameters != step_b.parameters:
        from eide.diff.parameter import diff_parameters

        param_changes = diff_parameters(step_a.parameters, step_b.parameters)
        for c in param_changes:
            c.key = f"{prefix}.parameters.{c.key}"
            c.category = DiffCategory.STRUCTURAL
        changes.extend(param_changes)

    return changes
