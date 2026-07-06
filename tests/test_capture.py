import tempfile
from pathlib import Path

import pytest

from eide.capture.context import ExperimentContext
from eide.capture.decorator import ExperimentLogger, track
from eide.core.ir import ExperimentIR


class TestExperimentLogger:
    def test_log_metric(self):
        exp = ExperimentIR()
        logger = ExperimentLogger(exp)
        logger.log_metric("accuracy", 0.95)
        assert exp.outputs.metrics["accuracy"] == 0.95

    def test_log_metrics(self):
        exp = ExperimentIR()
        logger = ExperimentLogger(exp)
        logger.log_metrics({"acc": 0.9, "loss": 0.1})
        assert exp.outputs.metrics["acc"] == 0.9

    def test_log_param(self):
        exp = ExperimentIR()
        logger = ExperimentLogger(exp)
        logger.log_param("lr", 0.01)
        assert exp.parameters["lr"] == 0.01

    def test_log_figure(self):
        exp = ExperimentIR()
        logger = ExperimentLogger(exp)
        logger.log_figure("plot.png")
        assert "plot.png" in exp.outputs.figures

    def test_log_tag(self):
        exp = ExperimentIR()
        logger = ExperimentLogger(exp)
        logger.log_tag("dataset", "cifar10")
        assert exp.tags["dataset"] == "cifar10"


class TestTrackDecorator:
    def test_track_basic(self):
        @track(name="test", auto_save=False, capture_env=False)
        def train(lr=0.01, epochs=10):
            return {"accuracy": 0.92}

        exp, result = train()
        assert exp.name == "test"
        assert exp.parameters["lr"] == 0.01
        assert exp.parameters["epochs"] == 10
        assert exp.outputs.metrics["accuracy"] == 0.92

    def test_track_no_return(self):
        @track(name="side-effect", auto_save=False, capture_env=False)
        def train(lr=0.01):
            pass

        exp, result = train()
        assert result is None
        assert exp.parameters["lr"] == 0.01

    def test_track_with_exception(self):
        @track(name="failing", auto_save=False, capture_env=False)
        def train():
            raise ValueError("bad param")

        with pytest.raises(ValueError):
            train()

    def test_track_default_name(self):
        @track(auto_save=False, capture_env=False)
        def my_train_func(lr=0.01):
            return {"acc": 0.9}

        exp, result = my_train_func()
        assert exp.name == "my_train_func"

    def test_track_empty_decorator(self):
        @track(auto_save=False, capture_env=False)
        def train():
            return 0.95

        exp, result = train()
        assert exp.outputs.metrics["result"] == 0.95


class TestExperimentContext:
    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with ExperimentContext("test-run", store_root=str(root), auto_save=True, capture_env=False) as logger:
                logger.log_param("lr", 0.01)
                logger.log_metric("accuracy", 0.92)
            store_path = root / "experiments"
            files = list(store_path.glob("*.json"))
            assert len(files) == 1

    def test_context_manager_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            try:
                with ExperimentContext("failing", store_root=str(root), auto_save=True, capture_env=False) as logger:
                    logger.log_param("x", 1)
                    raise RuntimeError("boom")
            except RuntimeError:
                pass
            store_path = root / "experiments"
            files = list(store_path.glob("*.json"))
            assert len(files) == 1
