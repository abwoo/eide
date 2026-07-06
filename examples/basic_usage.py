"""EIDE basic usage example."""

from eide import ExperimentIR, FileStore
from eide.capture.manual import capture_manual
from eide.diff.engine import diff_experiments

store = FileStore(".eide_example")

exp1 = capture_manual(
    name="baseline-cnn",
    description="Baseline CNN with LR=0.01",
    parameters={
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 10,
        "optimizer": "adam",
        "model": "cnn",
    },
    pipeline_steps=[
        {"name": "load_data", "type": "data"},
        {"name": "train", "type": "training", "parameters": {"epochs": 10}},
        {"name": "evaluate", "type": "eval"},
    ],
    metrics={"accuracy": 0.92, "loss": 0.35, "f1_score": 0.91},
    tags={"dataset": "cifar10", "experimenter": "alice"},
)

store.save(exp1)
print(f"Created experiment: {exp1.id}")

exp2 = capture_manual(
    name="changed-lr",
    description="Changed learning rate to 0.001",
    parameters={
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 10,
        "optimizer": "adam",
        "model": "cnn",
    },
    pipeline_steps=[
        {"name": "load_data", "type": "data"},
        {"name": "train", "type": "training", "parameters": {"epochs": 10}},
        {"name": "evaluate", "type": "eval"},
    ],
    metrics={"accuracy": 0.88, "loss": 0.42, "f1_score": 0.86},
    tags={"dataset": "cifar10", "experimenter": "alice"},
)

store.save(exp2)
print(f"Created experiment: {exp2.id}")

report = diff_experiments(exp1, exp2)
print("\n" + "=" * 60)
print("DIFF REPORT")
print("=" * 60)
print(report.summary)
print("\nChanges:")
for c in report.changes:
    print(f"  [{c.category.value:12s}] [{c.change_type.value:8s}] {c.description}")

print("\nImpact Estimates:")
for ie in report.impact_estimates:
    print(f"  {ie.metric}: {ie.change_percent:.1f}% {ie.change_direction}")

from eide.causal.attribution import CausalAttribution
causal = CausalAttribution(store=store)
explanation = causal.explain(exp1, exp2)
print("\n" + "=" * 60)
print("EXPLANATION")
print("=" * 60)
for cause in explanation["likely_cause"]:
    print(f"  {cause['cause']}: {cause['confidence']:.0%} confidence")
    print(f"    {cause['reasoning']}")
