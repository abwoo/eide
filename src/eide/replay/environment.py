from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from eide.core.types import EnvironmentInfo


class EnvironmentManager:
    def __init__(self, work_dir: str | Path | None = None):
        self.work_dir = Path(work_dir or tempfile.mkdtemp(prefix="eide_replay_"))
        self.venv_path: Path | None = None
        self._restored = False

    def restore(self, env_info: EnvironmentInfo) -> Path:
        """Restore a Python virtual environment matching env_info packages."""
        self.venv_path = self.work_dir / ".venv"
        if self.venv_path.exists():
            shutil.rmtree(self.venv_path)

        self._create_venv()
        self._install_packages(env_info.packages)
        self._restored = True
        return self.venv_path

    def _create_venv(self):
        import venv

        builder = venv.EnvBuilder(with_pip=True, clear=True)
        builder.create(self.venv_path)

    def _install_packages(self, packages: dict[str, str]):
        if not packages:
            return
        assert self.venv_path is not None
        pip = self.venv_path / ("Scripts" if platform.system() == "Windows" else "bin") / "pip"
        req_lines = [f"{pkg}=={ver}" if ver else pkg for pkg, ver in sorted(packages.items())]
        req_file = self.work_dir / "_eide_requirements.txt"
        req_file.write_text("\n".join(req_lines), encoding="utf-8")

        try:
            result = subprocess.run(
                [str(pip), "install", "-r", str(req_file)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                print(f"[warn] Some packages may not have been installed: {result.stderr[:200]}")
        except FileNotFoundError:
            print(f"[warn] pip not found at {pip}")
        except subprocess.TimeoutExpired:
            print("[warn] pip install timed out after 120s")
        finally:
            req_file.unlink(missing_ok=True)

    def detect_current(self) -> EnvironmentInfo:
        """Detect the current runtime environment."""
        info = EnvironmentInfo(
            os=f"{platform.system()} {platform.release()}",
            python_version=sys.version.split()[0],
        )
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                import json

                info.packages = {pkg["name"]: pkg["version"] for pkg in json.loads(result.stdout)}
        except Exception:
            pass
        return info

    def python_exe(self) -> str | None:
        if not self.venv_path:
            return None
        name = "python.exe" if platform.system() == "Windows" else "python"
        exe = self.venv_path / ("Scripts" if platform.system() == "Windows" else "bin") / name
        return str(exe) if exe.exists() else None

    def cleanup(self):
        if self.venv_path and self.venv_path.exists():
            shutil.rmtree(self.venv_path, ignore_errors=True)


class DockerEnvironmentManager:
    """Restore environment using Docker containers for full reproducibility."""

    def __init__(self, image_tag: str = "eide-replay:latest"):
        self.image_tag = image_tag
        self.container_id: str | None = None

    def restore(self, env_info: EnvironmentInfo, work_dir: str | Path) -> str:
        """Build a Docker image from env_info packages and return image tag."""
        dockerfile = self._generate_dockerfile(env_info)
        dockerfile_path = Path(work_dir) / "Dockerfile.eide"
        dockerfile_path.write_text(dockerfile, encoding="utf-8")

        try:
            result = subprocess.run(
                [
                    "docker",
                    "build",
                    "-t",
                    self.image_tag,
                    "-f",
                    str(dockerfile_path),
                    str(Path(work_dir).parent),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Docker build failed: {result.stderr[:500]}")
        except FileNotFoundError:
            raise RuntimeError("Docker not found. Install Docker or disable container replay.")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Docker build timed out after 300s")
        return self.image_tag

    def run(
        self, image_tag: str, command: str, work_dir: str | Path, timeout: int = 3600
    ) -> tuple[str, int]:
        """Run a command inside a Docker container."""
        try:
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{Path(work_dir).resolve()}:/workspace",
                    "-w",
                    "/workspace",
                    image_tag,
                    "sh",
                    "-c",
                    command,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout + result.stderr
            return output, result.returncode
        except FileNotFoundError:
            raise RuntimeError("Docker not found. Install Docker or disable container replay.")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Docker run timed out after {timeout}s")

    def run_python(
        self, image_tag: str, script_path: str | Path, work_dir: str | Path
    ) -> tuple[str, int]:
        """Run a Python script inside a Docker container."""
        return self.run(image_tag, f"python {script_path}", work_dir)

    def _generate_dockerfile(self, env_info: EnvironmentInfo) -> str:
        base_image = "python:3.11-slim"
        if "python" in env_info.runtime_versions:
            ver = env_info.runtime_versions["python"]
            base_image = f"python:{ver}-slim"

        lines = [f"FROM {base_image}", "WORKDIR /workspace", "COPY . /workspace"]

        if env_info.packages:
            req_lines = [
                f"{pkg}=={ver}" if ver else pkg
                for pkg, ver in sorted(env_info.packages.items())
                if pkg != "python"
            ]
            if req_lines:
                lines.append("RUN pip install --no-cache-dir " + " ".join(req_lines))

        lines.append('CMD ["python"]')
        return "\n".join(lines)

    def check_available(self) -> bool:
        """Check if Docker is available on this system."""
        try:
            result = subprocess.run(
                ["docker", "--version"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
