from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import networkx as nx

from eide.core.ir import ExperimentIR
from eide.diff.engine import DiffEngine
from eide.storage.filestore import FileStore


class ExperimentGraph:
    NODE_EXPERIMENT = "experiment"
    NODE_PARAMETER = "parameter"
    NODE_DATASET = "dataset"
    NODE_PIPELINE = "pipeline"
    NODE_METRIC = "metric"
    NODE_TAG = "tag"

    EDGE_HAS_PARAM = "has_parameter"
    EDGE_USES_DATA = "uses_dataset"
    EDGE_HAS_PIPELINE = "has_pipeline"
    EDGE_PRODUCED = "produced"
    EDGE_SIMILAR_TO = "similar_to"
    EDGE_DERIVED_FROM = "derived_from"
    EDGE_TAGGED = "tagged_with"
    EDGE_CONFLICTS_WITH = "conflicts_with"

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self._engine: DiffEngine | None = None

    def _get_engine(self) -> DiffEngine:
        if self._engine is None:
            self._engine = DiffEngine()
        return self._engine

    @classmethod
    def build(cls, store: FileStore, max_experiments: int | None = None) -> ExperimentGraph:
        eg = cls()
        entries = store.list_experiments()
        if max_experiments:
            entries = entries[:max_experiments]

        experiments = []
        for entry in entries:
            try:
                exp = store.load(entry["id"])
                experiments.append(exp)
                eg._add_experiment(exp)
            except Exception:
                continue

        eg._link_derived(experiments)
        return eg

    def _add_experiment(self, exp: ExperimentIR):
        exp_id = exp.id
        self.graph.add_node(exp_id, type=self.NODE_EXPERIMENT, label=exp.name or exp_id,
                            timestamp=str(exp.timestamp))

        for k, v in exp.parameters.items():
            param_id = self._param_id(k, v)
            self.graph.add_node(param_id, type=self.NODE_PARAMETER, label=f"{k}={v}", key=k, value=str(v))
            self.graph.add_edge(exp_id, param_id, type=self.EDGE_HAS_PARAM)

        for k, v in exp.data_versions.items():
            ds_id = self._dataset_id(k, v)
            self.graph.add_node(ds_id, type=self.NODE_DATASET, label=f"{k}:{v}", key=k, version=v)
            self.graph.add_edge(exp_id, ds_id, type=self.EDGE_USES_DATA)

        pipeline_name = exp.pipeline.name
        if pipeline_name:
            pipe_id = f"pipeline:{pipeline_name}"
            self.graph.add_node(pipe_id, type=self.NODE_PIPELINE, label=pipeline_name)
            self.graph.add_edge(exp_id, pipe_id, type=self.EDGE_HAS_PIPELINE)

        for k, v in exp.outputs.metrics.items():
            metric_id = self._metric_id(k, v, exp_id)
            self.graph.add_node(metric_id, type=self.NODE_METRIC, label=f"{k}={v}", key=k, value=v)
            self.graph.add_edge(exp_id, metric_id, type=self.EDGE_PRODUCED)

        for k, v in exp.tags.items():
            tag_id = f"tag:{k}={v}"
            self.graph.add_node(tag_id, type=self.NODE_TAG, label=f"{k}={v}", key=k, value=v)
            self.graph.add_edge(exp_id, tag_id, type=self.EDGE_TAGGED)

    def _link_derived(self, experiments: list[ExperimentIR]):
        tag_index: dict[str, list[ExperimentIR]] = {}
        for exp in experiments:
            replay_of = exp.tags.get("replay_of")
            if replay_of:
                target = next((e for e in experiments if e.id == replay_of), None)
                if target:
                    self.graph.add_edge(exp.id, replay_of, type=self.EDGE_DERIVED_FROM)
            for k, v in exp.tags.items():
                key = f"{k}={v}"
                tag_index.setdefault(key, []).append(exp)

        for key, group in tag_index.items():
            if len(group) > 1:
                sim = self._compute_similarity(group[0], group[1])
                if sim > 0.5:
                    self.graph.add_edge(group[0].id, group[1].id, type=self.EDGE_SIMILAR_TO,
                                        similarity=round(sim, 3))

        for i, a in enumerate(experiments):
            for b in experiments[i + 1:]:
                if self._has_conflicting_results(a, b):
                    self.graph.add_edge(a.id, b.id, type=self.EDGE_CONFLICTS_WITH)

    def _compute_similarity(self, a: ExperimentIR, b: ExperimentIR) -> float:
        report = self._get_engine().diff(a, b)
        total = len(report.changes)
        if total == 0:
            return 1.0
        return max(0.0, 1.0 - (total * 0.15))

    def _has_conflicting_results(self, a: ExperimentIR, b: ExperimentIR) -> bool:
        common_metrics = set(a.outputs.metrics.keys()) & set(b.outputs.metrics.keys())
        if not common_metrics:
            return False
        for key in common_metrics:
            va = a.outputs.metrics[key]
            vb = b.outputs.metrics[key]
            if va != 0 and abs((vb - va) / va) > 0.5:
                return True
        return False

    def save(self, path: str | Path):
        data = nx.node_link_data(self.graph)
        Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ExperimentGraph:
        eg = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        eg.graph = nx.node_link_graph(data, multigraph=True, directed=True)
        return eg

    @staticmethod
    def _param_id(key: str, value: Any) -> str:
        raw = f"param:{key}={value}"
        return f"param:{hashlib.md5(raw.encode()).hexdigest()[:8]}"

    @staticmethod
    def _dataset_id(key: str, version: str) -> str:
        return f"dataset:{key}:{version[:16] if version else key}"

    @staticmethod
    def _metric_id(key: str, value: float, exp_id: str) -> str:
        return f"metric:{key}={value}:{exp_id[:4]}"
