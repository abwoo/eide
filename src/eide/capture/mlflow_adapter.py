from __future__ import annotations

from eide.capture.base import CaptureAdapter
from eide.core.ir import ExperimentIR
from eide.core.types import EnvironmentInfo, ExperimentOutputs


class MLflowCapture(CaptureAdapter):
    def capture(self, run_id: str | None = None, **kwargs) -> ExperimentIR:
        try:
            import mlflow
        except ImportError:
            raise ImportError("mlflow is required. Install: pip install eide-core[mlflow]")

        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        data = run.data

        params = dict(data.params) if data.params else {}
        metrics = dict(data.metrics) if data.metrics else {}
        tags = dict(data.tags) if data.tags else {}

        exp = ExperimentIR(
            name=run.info.run_name or "",
            description=tags.pop("mlflow.note.content", ""),
            parameters=params,
            tags=tags,
            outputs=ExperimentOutputs(metrics=metrics),
            data_versions={
                "experiment_id": run.info.experiment_id,
                "run_id": run_id or "",
            },
            metadata={
                "mlflow_run_id": run_id or "",
                "mlflow_experiment_id": run.info.experiment_id,
                "mlflow_status": run.info.status,
                "mlflow_artifact_uri": run.info.artifact_uri,
            },
        )
        env = EnvironmentInfo()
        if run.data.tags:
            env.os = tags.get("mlflow.source.type", "")
        exp.environment = env
        return exp


def from_mlflow(run_id: str) -> ExperimentIR:
    """Capture an MLflow run as an ExperimentIR"""
    return MLflowCapture().capture(run_id=run_id)
