from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from eide.capture.mlflow_adapter import from_mlflow


def _mock_run(
    run_id="test-run-123",
    run_name="test-run",
    experiment_id="exp-1",
    status="FINISHED",
    artifact_uri="s3://bucket/artifacts",
    params=None,
    metrics=None,
    tags=None,
):
    mock_run = MagicMock()
    mock_run.info.run_id = run_id
    mock_run.info.run_name = run_name
    mock_run.info.experiment_id = experiment_id
    mock_run.info.status = status
    mock_run.info.artifact_uri = artifact_uri
    mock_data = MagicMock()
    mock_data.params = params or {}
    mock_data.metrics = metrics or {}
    mock_data.tags = tags or {}
    mock_run.data = mock_data
    return mock_run


class TestMLflowAdapter:
    def _patch_run(self, mock_run):
        """Patch eide.capture.mlflow_adapter.mlflow with a mock that returns run."""
        fake_mlflow = MagicMock()
        fake_mlflow.tracking.MlflowClient.return_value.get_run.return_value = mock_run
        return patch("eide.capture.mlflow_adapter.mlflow", fake_mlflow)

    def test_capture_basic(self):
        mock_run = _mock_run(
            params={"lr": "0.01", "batch_size": "32"},
            metrics={"accuracy": 0.95, "loss": 0.05},
            tags={"mlflow.source.type": "NOTEBOOK", "user": "alice"},
        )
        with self._patch_run(mock_run):
            exp = from_mlflow("test-run-123")
        assert exp.name == "test-run"
        assert exp.parameters == {"lr": "0.01", "batch_size": "32"}
        assert exp.outputs.metrics == {"accuracy": 0.95, "loss": 0.05}
        assert exp.tags["user"] == "alice"
        assert exp.tags.get("mlflow.source.type") == "NOTEBOOK"
        assert exp.metadata["mlflow_run_id"] == "test-run-123"

    def test_capture_empty(self):
        mock_run = _mock_run(params={}, metrics={}, tags={})
        with self._patch_run(mock_run):
            exp = from_mlflow("empty-run")
        assert exp.parameters == {}
        assert exp.outputs.metrics == {}
        assert exp.tags == {}

    def test_capture_with_mlflow_note(self):
        mock_run = _mock_run(tags={"mlflow.note.content": "My experiment note", "env": "prod"})
        with self._patch_run(mock_run):
            exp = from_mlflow("note-run")
        assert exp.description == "My experiment note"
        assert "mlflow.note.content" not in exp.tags
        assert exp.tags["env"] == "prod"

    def test_capture_without_name(self):
        mock_run = _mock_run(run_name="")
        with self._patch_run(mock_run):
            exp = from_mlflow("no-name")
        assert exp.name == ""

    def test_capture_metadata(self):
        mock_run = _mock_run(
            run_id="specific-id",
            experiment_id="exp-99",
            status="RUNNING",
            artifact_uri="/local/artifacts",
        )
        with self._patch_run(mock_run):
            exp = from_mlflow("specific-id")
        assert exp.metadata["mlflow_run_id"] == "specific-id"
        assert exp.metadata["mlflow_experiment_id"] == "exp-99"
        assert exp.metadata["mlflow_status"] == "RUNNING"
        assert exp.metadata["mlflow_artifact_uri"] == "/local/artifacts"
