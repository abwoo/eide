from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any

from eide.core.ir import ExperimentIR


class FileStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self._experiments_dir = self.root / "experiments"
        self._artifacts_dir = self.root / "artifacts"
        self._index_path = self.root / "index.json"
        self._init()

    def _init(self):
        self._experiments_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        if not self._index_path.exists():
            self._write_index([])

    def _write_index(self, entries: list[dict[str, Any]]):
        self._index_path.write_text(json.dumps(entries, indent=2, default=str), encoding="utf-8")

    def _read_index(self) -> list[dict[str, Any]]:
        if not self._index_path.exists():
            return []
        return json.loads(self._index_path.read_text(encoding="utf-8"))

    def save(self, experiment: ExperimentIR) -> Path:
        exp_path = self._experiments_dir / f"{experiment.id}.json"
        exp_path.write_text(
            experiment.model_dump_json(indent=2, exclude_none=True),
            encoding="utf-8",
        )
        index = self._read_index()
        entry = {
            "id": experiment.id,
            "name": experiment.name,
            "project": experiment.project,
            "timestamp": experiment.timestamp.isoformat(),
            "tags": experiment.tags,
            "pipeline_name": experiment.pipeline.name,
        }
        existing = [e for e in index if e["id"] != experiment.id]
        existing.append(entry)
        self._write_index(existing)
        return exp_path

    def load(self, experiment_id: str) -> ExperimentIR:
        exp_path = self._experiments_dir / f"{experiment_id}.json"
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment {experiment_id} not found at {exp_path}")
        data = json.loads(exp_path.read_text(encoding="utf-8"))
        return ExperimentIR.model_validate(data)

    def delete(self, experiment_id: str):
        exp_path = self._experiments_dir / f"{experiment_id}.json"
        if exp_path.exists():
            exp_path.unlink()
        index = self._read_index()
        self._write_index([e for e in index if e["id"] != experiment_id])

    def list_experiments(self) -> list[dict[str, Any]]:
        return self._read_index()

    def find_by_tag(self, key: str, value: str) -> list[ExperimentIR]:
        return [
            self.load(e["id"])
            for e in self._read_index()
            if e.get("tags", {}).get(key) == value
        ]

    def search(self, query: str, field: str = "all") -> list[ExperimentIR]:
        """Search experiments by keyword across name, tags, parameters, and metadata.
        Supports glob-like wildcards (*, ?)."""
        results = []
        for entry in self._read_index():
            exp = self.load(entry["id"])
            if self._matches_query(exp, query, field):
                results.append(exp)
        return results

    @staticmethod
    def _matches_query(exp: ExperimentIR, query: str, field: str) -> bool:
        q = query.lower()
        if field in ("all", "name"):
            if exp.name and q in exp.name.lower():
                return True
            if fnmatch.fnmatch((exp.name or "").lower(), q):
                return True
        if field in ("all", "tag"):
            for k, v in exp.tags.items():
                if q in f"{k}:{v}".lower():
                    return True
        if field in ("all", "param"):
            for k, v in _flatten_params(exp.parameters).items():
                if q in k.lower() or q in str(v).lower():
                    return True
        if field in ("all", "metadata"):
            for k, v in (exp.metadata or {}).items():
                if q in k.lower() or q in str(v).lower():
                    return True
        if field in ("all", "id"):
            if q in exp.id.lower():
                return True
        return False

    def find_by_metadata(self, key: str, value: Any) -> list[ExperimentIR]:
        """Find experiments where metadata[key] == value"""
        results = []
        for e in self._read_index():
            exp = self.load(e["id"])
            if exp.metadata and exp.metadata.get(key) == value:
                results.append(exp)
        return results

    def diff_with_baseline(self, experiment_id: str) -> dict | None:
        """Find the most similar experiment and diff against it.
        Returns a dict with 'baseline_id', 'change_count', 'summary' or None."""
        try:
            from eide.diff.engine import DiffEngine
            from eide.intelligence.advisor import ExperimentAdvisor

            target = self.load(experiment_id)
            advisor = ExperimentAdvisor(store=self)
            rec = advisor.recommend_baseline(target)
            bm = rec.best_match if rec else None
            bm_id = bm.get("experiment_id") if isinstance(bm, dict) else bm
            if bm_id:
                baseline = self.load(bm_id)
                engine = DiffEngine()
                report = engine.diff(target, baseline)
                return {
                    "baseline_id": bm_id,
                    "baseline_name": baseline.name,
                    "change_count": len(report.changes),
                    "summary": report.summary,
                }
        except Exception:
            return None

    def get_artifact_path(self, experiment_id: str) -> Path:
        path = self._artifacts_dir / experiment_id
        path.mkdir(parents=True, exist_ok=True)
        return path


def _flatten_params(params: dict, prefix: str = "") -> dict:
    flat = {}
    for k, v in params.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_params(v, key))
        else:
            flat[key] = v
    return flat
