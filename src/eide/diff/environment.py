from __future__ import annotations

from eide.core.diff import ChangeType, DiffCategory, DiffChange
from eide.core.types import EnvironmentInfo


def diff_environment(env_a: EnvironmentInfo, env_b: EnvironmentInfo) -> list[DiffChange]:
    changes: list[DiffChange] = []

    if env_a.os != env_b.os:
        changes.append(
            DiffChange(
                category=DiffCategory.ENVIRONMENT,
                change_type=ChangeType.MODIFIED,
                key="environment.os",
                old_value=env_a.os,
                new_value=env_b.os,
                description=f"OS: {env_a.os} -> {env_b.os}",
            )
        )

    if env_a.python_version != env_b.python_version:
        changes.append(
            DiffChange(
                category=DiffCategory.ENVIRONMENT,
                change_type=ChangeType.MODIFIED,
                key="environment.python_version",
                old_value=env_a.python_version,
                new_value=env_b.python_version,
                description=f"Python: {env_a.python_version} -> {env_b.python_version}",
            )
        )

    all_pkgs = set(env_a.packages.keys()) | set(env_b.packages.keys())
    for pkg in sorted(all_pkgs):
        va = env_a.packages.get(pkg)
        vb = env_b.packages.get(pkg)
        if va is None:
            changes.append(
                DiffChange(
                    category=DiffCategory.ENVIRONMENT,
                    change_type=ChangeType.ADDED,
                    key=f"environment.packages.{pkg}",
                    new_value=vb,
                    description=f"Package added: {pkg}=={vb}",
                )
            )
        elif vb is None:
            changes.append(
                DiffChange(
                    category=DiffCategory.ENVIRONMENT,
                    change_type=ChangeType.REMOVED,
                    key=f"environment.packages.{pkg}",
                    old_value=va,
                    description=f"Package removed: {pkg}=={va}",
                )
            )
        elif va != vb:
            changes.append(
                DiffChange(
                    category=DiffCategory.ENVIRONMENT,
                    change_type=ChangeType.MODIFIED,
                    key=f"environment.packages.{pkg}",
                    old_value=va,
                    new_value=vb,
                    description=f"Package {pkg}: {va} -> {vb}",
                )
            )

    return changes
