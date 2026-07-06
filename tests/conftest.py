from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from eide.core.ir import ExperimentIR
from eide.core.types import ExperimentOutputs
from eide.storage.filestore import FileStore


@pytest.fixture
def tmp_store():
    """Create a FileStore in a temporary directory, automatically cleaned up."""
    tmp = Path(tempfile.mkdtemp())
    store = FileStore(tmp)
    yield store


@pytest.fixture
def sample_experiments(tmp_store):
    """Create 3 sample experiments for testing."""
    store = tmp_store
    a = ExperimentIR(
        name="alpha",
        parameters={"lr": 0.01, "batch_size": 32},
        outputs=ExperimentOutputs(metrics={"acc": 0.9, "loss": 0.1}),
        tags={"dataset": "mnist", "model": "cnn"},
    )
    b = ExperimentIR(
        name="beta",
        parameters={"lr": 0.02, "batch_size": 64},
        outputs=ExperimentOutputs(metrics={"acc": 0.85, "loss": 0.2}),
        tags={"dataset": "cifar", "model": "cnn"},
    )
    c = ExperimentIR(
        name="gamma",
        parameters={"lr": 0.01, "batch_size": 32},
        outputs=ExperimentOutputs(metrics={"acc": 0.88, "loss": 0.15}),
        tags={"dataset": "mnist", "model": "mlp"},
        data_versions={"train": "v2", "test": "v1"},
        metadata={"user": "alice"},
    )
    store.save(a)
    store.save(b)
    store.save(c)
    return store


@pytest.fixture
def exp_alpha(sample_experiments):
    return sample_experiments.load(
        [e["id"] for e in sample_experiments.list_experiments() if e["name"] == "alpha"][0]
    )


@pytest.fixture
def exp_beta(sample_experiments):
    return sample_experiments.load(
        [e["id"] for e in sample_experiments.list_experiments() if e["name"] == "beta"][0]
    )
