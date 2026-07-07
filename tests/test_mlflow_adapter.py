from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from eide.capture.mlflow_adapter import MLflowCapture, from_mlflow


@pytest.fixture(scope="module")
def mlflow_tracking_uri():
    tmp = tempfile.mkdtemp(prefix="eide_mlflow_test_")
    tracking_uri = f"sqlite:///{Path(tmp, 'mlflow.db').as_posix()}"
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    with patch.dict("os.environ", {"MLFLOW_TRACKING_URI": tracking_uri}):
        yield tracking_uri


@pytest.fixture(autouse=True)
def reset_mlflow():
    yield
    import mlflow

    mlflow.end_run()


class TestMLflowAdapter:
    def test_from_mlflow_basic(self, mlflow_tracking_uri):
        import mlflow

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        exp_name = "eide_test_adapter"
        mlflow.set_experiment(exp_name)
        with mlflow.start_run(run_name="adapter-test-run") as run:
            mlflow.log_param("lr", 0.01)
            mlflow.log_param("epochs", 10)
            mlflow.log_metric("acc", 0.95)
            mlflow.log_metric("loss", 0.05)
            mlflow.set_tag("model", "resnet")
            mlflow.set_tag("mlflow.note.content", "Test from EIDE")
            run_id = run.info.run_id

        result = from_mlflow(run_id)
        assert result.name == "adapter-test-run"
        assert result.parameters == {"lr": "0.01", "epochs": "10"}
        assert result.outputs.metrics == {"acc": 0.95, "loss": 0.05}
        assert result.tags.get("model") == "resnet"
        assert result.metadata["mlflow_run_id"] == run_id
        assert result.metadata["mlflow_experiment_id"] is not None

    def test_from_mlflow_with_experiment_id(self, mlflow_tracking_uri):
        import mlflow

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        exp_name = "eide_test_exp_id"
        exp_id = mlflow.create_experiment(exp_name)
        with mlflow.start_run(experiment_id=exp_id, run_name="exp-id-test") as run:
            mlflow.log_param("batch_size", 64)
            mlflow.log_metric("f1", 0.89)
            run_id = run.info.run_id

        result = from_mlflow(run_id)
        assert result.parameters == {"batch_size": "64"}
        assert result.outputs.metrics == {"f1": 0.89}
        from eide.core.types import ExperimentOutputs

        assert isinstance(result.outputs, ExperimentOutputs)
        assert result.data_versions["experiment_id"] == exp_id

    def test_capture_raises_without_mlflow(self):
        with patch("eide.capture.mlflow_adapter.mlflow", None):
            with pytest.raises(ImportError, match="mlflow is required"):
                from_mlflow("fake-run-id")

    def test_capture_raises_without_run_id(self):
        capture = MLflowCapture()
        with pytest.raises(ValueError, match="run_id is required"):
            capture.capture()

    def test_from_mlflow_empty_run(self, mlflow_tracking_uri):
        import mlflow

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        with mlflow.start_run(run_name="empty-run") as run:
            run_id = run.info.run_id

        result = from_mlflow(run_id)
        assert result.name == "empty-run"
        assert result.parameters == {}
        assert result.outputs.metrics == {}
