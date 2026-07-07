from __future__ import annotations

import fnmatch
import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from eide.core.ir import ExperimentIR

_STALE_LOCK_SECONDS = 30
_WAL_NAME = ".index.wal"


class FileStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self._experiments_dir = self.root / "experiments"
        self._artifacts_dir = self.root / "artifacts"
        self._index_path = self.root / "index.json"
        self._lock_path = self.root / ".index.lock"
        self._wal_path = self.root / _WAL_NAME
        self._lock = threading.Lock()
        self._init()

    def _init(self):
        self._experiments_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        if not self._index_path.exists():
            self._write_index([])
        self._recover_from_wal()

    @staticmethod
    def _write_json_atomic(path: Path, data: Any):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        os.replace(str(tmp), str(path))

    def _write_index(self, entries: list[dict[str, Any]]):
        self._write_json_atomic(self._index_path, entries)

    def _read_index(self) -> list[dict[str, Any]]:
        if not self._index_path.exists():
            return []
        return json.loads(self._index_path.read_text(encoding="utf-8"))

    @contextmanager
    def _exclusive_index(self) -> Iterator[None]:
        with self._lock:
            lock_path = self._lock_path
            deadline = time.monotonic() + 10.0
            while True:
                try:
                    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                    payload = f"{os.getpid()} {time.time()}\n"
                    os.write(fd, payload.encode())
                    os.close(fd)
                    break
                except FileExistsError:
                    if self._is_lock_stale(lock_path):
                        try:
                            lock_path.unlink()
                        except FileNotFoundError:
                            pass
                        continue
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Could not acquire index lock at {lock_path}")
                    time.sleep(0.01)
            try:
                yield
            finally:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass

    @staticmethod
    def _is_lock_stale(lock_path: Path) -> bool:
        try:
            content = lock_path.read_text("utf-8").strip()
            parts = content.split(None, 1)
            if len(parts) == 2:
                pid = int(parts[0])
                try:
                    os.kill(pid, 0)
                    return False
                except ProcessLookupError:
                    return True
                except PermissionError:
                    return time.time() - float(parts[1]) > _STALE_LOCK_SECONDS
                except OSError:
                    pass
        except (OSError, ValueError, IndexError):
            pass
        return False

    def _write_wal(self, op: str, exp_id: str):
        wal = {
            "op": op,
            "experiment_id": exp_id,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        self._wal_path.write_text(json.dumps(wal), encoding="utf-8")

    def _clear_wal(self):
        try:
            self._wal_path.unlink()
        except FileNotFoundError:
            pass

    def _recover_from_wal(self):
        if not self._wal_path.exists():
            return
        try:
            wal = json.loads(self._wal_path.read_text("utf-8"))
            op = wal.get("op")
            exp_id = wal.get("experiment_id")
            if not op or not exp_id:
                self._clear_wal()
                return
            if op == "save":
                exp_path = self._experiments_dir / f"{exp_id}.json"
                if exp_path.exists():
                    index = self._read_index()
                    if not any(e["id"] == exp_id for e in index):
                        exp = self.load(exp_id)
                        entry = {
                            "id": exp.id,
                            "name": exp.name,
                            "project": exp.project,
                            "timestamp": exp.timestamp.isoformat(),
                            "tags": exp.tags,
                            "pipeline_name": exp.pipeline.name,
                        }
                        index.append(entry)
                        self._write_index(index)
            elif op == "delete":
                index = self._read_index()
                filtered = [e for e in index if e["id"] != exp_id]
                if len(filtered) < len(index):
                    self._write_index(filtered)
        except Exception:
            pass
        finally:
            self._clear_wal()

    def save(self, experiment: ExperimentIR) -> Path:
        with self._exclusive_index():
            exp_path = self._experiments_dir / f"{experiment.id}.json"
            self._write_wal("save", experiment.id)
            self._write_json_atomic(
                exp_path, json.loads(experiment.model_dump_json(exclude_none=True))
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
            self._clear_wal()
            return exp_path

    def load(self, experiment_id: str) -> ExperimentIR:
        exp_path = self._experiments_dir / f"{experiment_id}.json"
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment {experiment_id} not found at {exp_path}")
        data = json.loads(exp_path.read_text(encoding="utf-8"))
        return ExperimentIR.model_validate(data)

    def delete(self, experiment_id: str):
        with self._exclusive_index():
            exp_path = self._experiments_dir / f"{experiment_id}.json"
            if not exp_path.exists():
                raise FileNotFoundError(f"Experiment {experiment_id} not found")
            self._write_wal("delete", experiment_id)
            exp_path.unlink()
            index = self._read_index()
            self._write_index([e for e in index if e["id"] != experiment_id])
            self._clear_wal()

    def list_experiments(self) -> list[dict[str, Any]]:
        return self._read_index()

    def find_by_tag(self, key: str, value: str) -> list[ExperimentIR]:
        return [
            self.load(e["id"]) for e in self._read_index() if e.get("tags", {}).get(key) == value
        ]

    def search(self, query: str, field: str = "all") -> list[ExperimentIR]:
        """Search experiments by keyword across name, tags, parameters, and metadata.
        Supports glob-like wildcards (*, ?).
        Fast path: index-level search for name/id/tag fields — no full experiment load.
        For param/metadata, loads all experiments (these fields are not in the index)."""
        q = query.lower()
        entries = self._read_index()

        if field in ("name", "id", "tag", "project"):
            return self._search_index_only(entries, q, field)

        candidates = entries
        if field == "all":
            candidates = [e for e in entries if self._quick_match_index(e, q)]

        results = []
        for entry in candidates:
            exp = self.load(entry["id"])
            if self._full_match(exp, q, field):
                results.append(exp)
        return results

    def _search_index_only(self, entries: list[dict], q: str, field: str) -> list[ExperimentIR]:
        results = []
        for entry in entries:
            if self._quick_match_index(entry, q, field=field):
                results.append(self.load(entry["id"]))
        return results

    @staticmethod
    def _quick_match_index(entry: dict, q: str, field: str = "all") -> bool:
        name = entry.get("name") or ""
        if field in ("all", "name"):
            if q in name.lower():
                return True
            if fnmatch.fnmatch(name.lower(), q):
                return True
        if field in ("all", "id"):
            if q in (entry.get("id") or "").lower():
                return True
        if field in ("all", "tag"):
            tags = entry.get("tags") or {}
            for k, v in tags.items():
                if q in f"{k}:{v}".lower():
                    return True
        if field in ("all", "project"):
            proj = entry.get("project") or ""
            if q in proj.lower():
                return True
        return False

    @staticmethod
    def _full_match(exp: ExperimentIR, q: str, field: str) -> bool:
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
        results: list[ExperimentIR] = []
        for e in self._read_index():
            exp = self.load(e["id"])
            if exp.metadata and exp.metadata.get(key) == value:
                results.append(exp)
        return results

    def diff_with_baseline(self, experiment_id: str) -> dict | None:
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
            return None
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
