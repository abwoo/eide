# EIDE Core

**Experiment Intelligence & Diff Engine** — Understand why experiments differ.

EIDE is not an experiment tracking tool. It's an **experiment understanding infrastructure** that answers:

- ❓ Why did accuracy drop?
- ❓ Which parameter change caused the result shift?
- ❓ What's the difference between experiment A and B?
- ❓ Which experiment is the most reliable baseline?
- ❓ Can this experiment be reproduced?

## Quick Start

```bash
pip install eide-core

# Initialize a store
eide init

# Create experiments
eide create --name "baseline" --tag lr=0.01 --tag model=cnn

# Auto-capture with decorator
python -c "
from eide import track

@track(name='my-run')
def train(lr=0.01, epochs=10):
    return {'accuracy': 0.92}

exp, result = train()
"

# Capture from MLflow
eide capture mlflow <run_id>

# Diff two experiments
eide diff <exp_id_a> <exp_id_b>

# Explain differences
eide explain <exp_id_a> <exp_id_b>

# Intelligent comparison with recommendations
eide advise compare <exp_id_a> <exp_id_b>

# Assess experiment risk
eide advise risk <exp_id>

# Replay an experiment
eide replay <exp_id>

# Build and query the knowledge graph
eide graph build
eide graph duplicates <exp_id>
eide graph conflicts <exp_id>
eide graph lineage <exp_id>
eide graph top-params
eide graph param-impact <key>

# Search experiments
eide search <query>
eide search --field param "0.01"

# Manage tags
eide tag add <id> <key> <value>
eide tag remove <id> <key>
eide tag list <id>

# Filter by tag or project
eide list --tag dataset=mnist
eide list --project my-project

# Create with project
eide create --name "run" --project my-project

# Auto-diff on capture
eide create --name "v2" --param lr=0.02 --metric acc=0.85 --diff
eide capture manual --name "v2" --param lr=0.02 --metric acc=0.85 --diff

# Run from YAML config
eide run experiment.yml
eide run experiment.yml --diff

# Export/import experiments
eide export <exp_id> backup.json
eide import backup.json

# Start web dashboard (port 8765)
eide dashboard
eide dashboard --port 8765 --host 0.0.0.0

# Start MCP server (for Claude/Cursor)
eide serve
```

## Python Library

```python
from eide import ExperimentIR, FileStore, track, ExperimentContext
from eide.diff.engine import DiffEngine
from eide.intelligence.advisor import ExperimentAdvisor

# Auto-capture decorator
@track(name="training-run")
def train(lr=0.01, batch_size=32):
    # ... training code ...
    return {"accuracy": 0.92, "loss": 0.35}

exp, result = train()

# Context manager with project + resource monitoring
with ExperimentContext("eval-run", project="my-project", monitor_resources=True) as logger:
    logger.log_param("model", "cnn")
    logger.log_metric("f1", 0.89)

# Manual capture with project
from eide.capture.manual import capture_manual
exp = capture_manual(name="run", project="my-project", parameters={"lr": 0.01})

# Diff
engine = DiffEngine()
report = engine.diff(exp_a, exp_b)
print(report.changelog)

# Intelligent comparison
advisor = ExperimentAdvisor(store=store)
insight = advisor.compare(exp_a, exp_b)
print(insight.recommended_actions)
```

## All CLI Commands

All commands support `--root <path>` to specify a custom store location.

| Command | Arguments | Options | Description |
|---|---|---|---|
| `eide init` | — | `--root` | Initialize experiment store |
| `eide create` | — | `--name`, `--project`, `--tag` (multiple), `--diff`, `--root` | Create experiment |
| `eide list` | — | `--tag` (multiple), `--project`, `--root` | List experiments |
| `eide show` | `<id>` | `--root` | Show experiment details |
| `eide delete` | `<id>` | `--root` | Delete experiment |
| `eide search` | `<query>` | `--field` (name/tag/param/metadata/id/all), `--json`, `--root` | Search experiments |
| `eide tag add` | `<id>` `<key>` `<value>` | `--root` | Add a tag |
| `eide tag remove` | `<id>` `<key>` | `--root` | Remove a tag |
| `eide tag list` | `<id>` | `--root` | List tags |
| `eide diff` | `<a>` `<b>` | `--format` (text/json), `--artifacts`, `--root` | Diff two experiments |
| `eide capture manual` | — | `--name`, `--description`, `--project`, `--param`, `--metric`, `--tag`, `--step`, `--data-version`, `--decision`, `--metadata`, `--diff`, `--root` | Manually capture experiment |
| `eide capture mlflow` | `<run_id>` | `--name`, `--project`, `--root` | Capture MLflow run |
| `eide run` | `<config.yml>` | `--diff`, `--dry-run`, `--root` | Run from YAML config |
| `eide export` | `<id>` `<path>` | `--pretty`, `--root` | Export as JSON |
| `eide import-exp` | `<path>` | `--id` (override), `--root` | Import from JSON |
| `eide explain` | `<a>` `<b>` | `--root` | Explain differences |
| `eide replay` | `<id>` | `--no-env`, `--no-data`, `--no-save`, `--artifacts`, `--work-dir`, `--root` | Replay experiment |
| `eide graph build` | — | `--max`, `--save`, `--root` | Build knowledge graph |
| `eide graph duplicates` | `<id>` | `--save`, `--root` | Find duplicates |
| `eide graph conflicts` | `<id>` | `--save`, `--root` | Find conflicts |
| `eide graph lineage` | `<id>` | `--save`, `--root` | Show lineage |
| `eide graph top-params` | — | `--top` (default 10), `--save`, `--root` | Most used parameters |
| `eide graph param-impact` | `<key>` | `--save`, `--root` | Parameter-metric correlation |
| `eide advise compare` | `<a>` `<b>` | `--root` | Intelligent comparison |
| `eide advise risk` | `<id>` | `--root` | Assess risk |
| `eide advise baseline` | `<id>` | `--root` | Find best baseline |
| `eide serve` | — | `--host` (127.0.0.1), `--port` (8080), `--root` | Start MCP server |
| `eide dashboard` | — | `--host` (127.0.0.1), `--port` (8765), `--root` | Start web dashboard |

## Architecture

```
Observation Layer → ExperimentIR → Diff Engine → Causal Engine
                     ↓
                Knowledge Graph ← → Replay Engine
                     ↓
              Intelligence Layer (explain / warn / recommend)
```

### Key Features

| Feature | Description |
|---|---|
| **Auto-capture** | `@track()` decorator + `ExperimentContext` context manager |
| **Logging API** | `logger.log_param()`, `logger.log_metric()`, `logger.log_figure()` |
| **3-layer Diff** | Parameter (numeric tolerance), Structural (LCS pipeline alignment), Semantic (SSIM + statistical effect size) |
| **Causal Attribution** | Change ranking + baseline similarity search |
| **Replay** | Venv + Docker environment restoration, pipeline execution, output verification |
| **Knowledge Graph** | NetworkX-backed experiment graph with duplicate/conflict/lineage/parameter-impact queries |
| **Intelligence** | Risk assessment, baseline recommendation, action generation |
| **MCP Server** | 13+ tools for Claude/Cursor integration |
| **Web Dashboard** | FastAPI + Jinja2 dashboard with charts, graph visualization, diff UI |
| **Search & Tags** | Full-text search across name/tags/params/metadata, tag management CLI |
| **Enhanced Environment** | GPU detection (nvidia-smi + torch), git commit, conda, CPU/RAM |
| **Config-driven runs** | `eide run <config.yml>` — YAML-defined experiments with auto-capture |
| **Resource Monitoring** | CPU/GPU/RAM tracking in `ExperimentContext` with timeline charts |
| **Pipeline DAG** | Visual pipeline graph in dashboard using vis-network |
| **Export/Import** | Portable JSON roundtrip for cross-system sharing |
| **Project Groups** | `project` field on experiments for multi-project organization |

## MCP Server

EIDE exposes all capabilities via the Model Context Protocol (MCP).

```bash
eide serve
```

Available tools: `eide_capture`, `eide_diff`, `eide_explain`, `eide_list`, `eide_show`, `eide_delete`, `eide_search`, `eide_replay`, `eide_graph_duplicates`, `eide_graph_conflicts`, `eide_graph_lineage`, `eide_graph_top_params`, `eide_graph_param_impact`, `eide_advise_risk`, `eide_advise_baseline`

Available resources: `eide://experiments` (list), `eide://experiments/{id}` (detail)

## Development

```bash
pip install -e ".[full]"
pytest tests/
```
