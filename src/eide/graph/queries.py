from __future__ import annotations

import networkx as nx

from eide.graph.builder import ExperimentGraph


class GraphQuery:
    def __init__(self, graph: ExperimentGraph):
        self.g = graph.graph

    def duplicates(self, experiment_id: str) -> list[dict]:
        """Find experiments that are likely duplicates of the given one."""
        results = []
        for _, other, data in self.g.edges(experiment_id, data=True):
            if data.get("type") == ExperimentGraph.EDGE_SIMILAR_TO:
                sim = data.get("similarity", 0)
                label = self.g.nodes[other].get("label", other)
                results.append({"id": other, "label": label, "similarity": sim})
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    def conflicts(self, experiment_id: str) -> list[dict]:
        """Find experiments with conflicting results."""
        results = []
        for _, other, data in self.g.edges(experiment_id, data=True):
            if data.get("type") == ExperimentGraph.EDGE_CONFLICTS_WITH:
                label = self.g.nodes[other].get("label", other)
                results.append({"id": other, "label": label})
        return results

    def lineage(self, experiment_id: str) -> list[dict]:
        """Get the replay lineage (chain of derived_from)."""
        chain = []
        current = experiment_id
        visited = set()
        while current and current not in visited:
            visited.add(current)
            label = self.g.nodes[current].get("label", current)
            chain.append({"id": current, "label": label})
            successors = list(self.g.successors(current))
            next_id = None
            for succ in successors:
                edge_data = self.g.get_edge_data(current, succ)
                if edge_data:
                    for ed in edge_data.values():
                        if ed.get("type") == ExperimentGraph.EDGE_DERIVED_FROM:
                            next_id = succ
                            break
            current = next_id
        return chain

    def most_used_parameters(self, top_n: int = 10) -> list[dict]:
        """Find the most commonly used parameter values across experiments."""
        param_counts: dict[str, dict] = {}
        for node, data in self.g.nodes(data=True):
            if data.get("type") == ExperimentGraph.NODE_PARAMETER:
                key = data.get("key", "")
                value = data.get("value", "")
                param_key = f"{key}={value}"
                if param_key not in param_counts:
                    param_counts[param_key] = {"key": key, "value": value, "count": 0}
                param_counts[param_key]["count"] += 1
        sorted_params = sorted(param_counts.values(), key=lambda x: x["count"], reverse=True)
        return sorted_params[:top_n]

    def parameter_impact(self, param_key: str) -> list[dict]:
        """Find how different values of a parameter correlate with metrics."""
        experiments_with_param = set()
        param_values: dict[str, set] = {}
        for node, data in self.g.nodes(data=True):
            if data.get("type") == ExperimentGraph.NODE_PARAMETER and data.get("key") == param_key:
                val = data.get("value", "")
                param_values.setdefault(val, set())
                for exp_id, _ in self.g.in_edges(node):
                    if self.g.nodes[exp_id].get("type") == ExperimentGraph.NODE_EXPERIMENT:
                        param_values[val].add(exp_id)
                        experiments_with_param.add(exp_id)

        results = []
        for val, exp_ids in param_values.items():
            metrics_summary: dict[str, list[float]] = {}
            for exp_id in exp_ids:
                for _, metric_node, data in self.g.out_edges(exp_id, data=True):
                    if data.get("type") == ExperimentGraph.EDGE_PRODUCED:
                        mn = self.g.nodes[metric_node]
                        mk = mn.get("key", "")
                        mv = mn.get("value")
                        if mk and mv is not None:
                            metrics_summary.setdefault(mk, []).append(float(mv))
            stats = {}
            for mk, mv_list in metrics_summary.items():
                if mv_list:
                    stats[mk] = {
                        "mean": round(sum(mv_list) / len(mv_list), 4),
                        "count": len(mv_list),
                    }
            results.append({"value": val, "experiment_count": len(exp_ids), "metrics": stats})
        return results

    def experiments_by_parameter(self, key: str, value: str | None = None) -> list[dict]:
        """Find experiments that used a specific parameter value."""
        results = []
        for node, data in self.g.nodes(data=True):
            if data.get("type") == ExperimentGraph.NODE_PARAMETER and data.get("key") == key:
                if value is not None and data.get("value") != value:
                    continue
                for exp_id, _ in self.g.in_edges(node):
                    label = self.g.nodes[exp_id].get("label", exp_id)
                    results.append({"id": exp_id, "label": label, "param_value": data.get("value")})
        return results

    def most_volatile_parameters(self, top_n: int = 5) -> list[dict]:
        """Find parameters that change most frequently across experiments."""
        param_values: dict[str, set] = {}
        for node, data in self.g.nodes(data=True):
            if data.get("type") == ExperimentGraph.NODE_PARAMETER:
                key = data.get("key", "")
                val = data.get("value", "")
                if key:
                    param_values.setdefault(key, set()).add(val)
        sorted_params = sorted(
            [{"key": k, "unique_values": len(v), "values": list(v)} for k, v in param_values.items()],
            key=lambda x: x["unique_values"], reverse=True,
        )
        return sorted_params[:top_n]

    def export_graphml(self, path: str):
        nx.write_graphml(self.g, path)
