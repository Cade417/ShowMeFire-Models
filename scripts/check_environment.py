"""
Single-command environment preflight for model-training - checks
everything currently checked piecemeal (spatial/env_check.py's CUDA-only
check) or not at all (package version drift, SMF_DATA_ROOT's silent
fallback, disk space, optional API tokens), before a multi-hour download
or training run starts.

Warnings do not block (exit 0); only genuine failures do (exit 1) - CPU-
only workflows are valid (legacy XGBoost, risk_fusion's GLM), so missing
CUDA is a warning, not a failure, and SYNOPTIC_API_TOKEN/SMF_GITHUB_REPO
are only needed by specific downstream scripts, not everything.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"
MIN_FREE_DISK_GB = 20.0


class CheckResult(NamedTuple):
    name: str
    level: str  # "pass", "warn", "fail"
    detail: str


def _parse_pinned_requirements(path: Path) -> Dict[str, str]:
    """{distribution_name: pinned_version} for every == pin whose environment marker matches this platform.

    Uses packaging.requirements.Requirement for real PEP 508 marker
    evaluation - a naive string split would flag eccodeslib as "missing"
    on Windows even though requirements.txt's own marker
    (`; platform_system != "Windows"`) correctly excludes it there.
    """
    try:
        from packaging.requirements import InvalidRequirement, Requirement
    except ImportError:
        return _parse_pinned_requirements_naive(path)

    pinned = {}
    if not path.exists():
        return pinned
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "==" not in line:
            continue
        try:
            requirement = Requirement(line)
        except InvalidRequirement:
            continue
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        for specifier in requirement.specifier:
            if specifier.operator == "==":
                pinned[requirement.name] = specifier.version
                break
    return pinned


def _parse_pinned_requirements_naive(path: Path) -> Dict[str, str]:
    """Fallback when the `packaging` library is unavailable - cannot evaluate environment markers, so platform-conditional pins may be misreported."""
    pinned = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "==" not in line:
            continue
        name_part = line.split(";", 1)[0].strip()
        name, _, version = name_part.partition("==")
        pinned[name.strip()] = version.strip()
    return pinned


def check_package_versions(requirements_path: Path = REQUIREMENTS_PATH) -> List[CheckResult]:
    pinned = _parse_pinned_requirements(requirements_path)
    if not pinned:
        return [CheckResult("package_versions", "warn", f"no pinned packages found in {requirements_path}")]
    results = []
    for name, expected_version in sorted(pinned.items()):
        try:
            installed_version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            results.append(CheckResult(f"package:{name}", "fail", f"not installed (requirements.txt pins {expected_version})"))
            continue
        if installed_version != expected_version:
            results.append(CheckResult(f"package:{name}", "warn",
                                       f"installed {installed_version}, requirements.txt pins {expected_version}"))
        else:
            results.append(CheckResult(f"package:{name}", "pass", installed_version))
    return results


def check_cuda() -> CheckResult:
    """Same torch/CUDA/VRAM check as spatial/env_check.py, reused rather than duplicated - CPU-only is a warn, not a fail."""
    try:
        import torch
    except ImportError:
        return CheckResult("cuda", "warn",
                           "torch not installed - spatial (V4/V5/station-sequence) training unavailable; "
                           "CPU-only workflows (legacy XGBoost, risk_fusion GLM) unaffected")
    if not torch.cuda.is_available():
        return CheckResult("cuda", "warn", f"torch {torch.__version__} installed, CUDA not available - spatial training will run on CPU or be unavailable")
    props = torch.cuda.get_device_properties(0)
    return CheckResult("cuda", "pass", f"torch {torch.__version__}, {props.name}, {props.total_memory / 1024**3:.1f} GB VRAM")


def check_data_root() -> CheckResult:
    """Flags the exact silent-fallback risk in paths.py: SMF_DATA_ROOT unset -> falls back to an in-repo data/ dir with no warning anywhere else."""
    configured = os.getenv("SMF_DATA_ROOT", "").strip()
    if not configured:
        return CheckResult("data_root", "warn", f"SMF_DATA_ROOT is not set - falling back to {paths.DATA_ROOT} (in-repo). Is this intentional?")
    resolved = Path(configured).resolve()
    if not resolved.exists():
        return CheckResult("data_root", "fail", f"SMF_DATA_ROOT={configured} does not exist")
    if not os.access(resolved, os.W_OK):
        return CheckResult("data_root", "fail", f"SMF_DATA_ROOT={resolved} is not writable")
    return CheckResult("data_root", "pass", str(resolved))


def check_disk_space(min_free_gb: float = MIN_FREE_DISK_GB) -> CheckResult:
    try:
        usage = shutil.disk_usage(paths.DATA_ROOT)
    except OSError as error:
        return CheckResult("disk_space", "fail", f"could not stat {paths.DATA_ROOT}: {error}")
    free_gb = usage.free / 1024 ** 3
    if free_gb < min_free_gb:
        return CheckResult("disk_space", "warn",
                           f"{free_gb:.1f} GB free at {paths.DATA_ROOT} (below the {min_free_gb:.0f} GB guideline - HRRR/RTMA backfills are data-heavy)")
    return CheckResult("disk_space", "pass", f"{free_gb:.1f} GB free at {paths.DATA_ROOT}")


def check_optional_token(env_var: str, purpose: str) -> CheckResult:
    if os.getenv(env_var, "").strip():
        return CheckResult(env_var, "pass", "set")
    return CheckResult(env_var, "warn", f"not set - only needed for {purpose}")


def run_all() -> List[CheckResult]:
    results = [check_data_root(), check_disk_space(), check_cuda()]
    results.extend(check_package_versions())
    results.append(check_optional_token("SYNOPTIC_API_TOKEN", "Synoptic station backfill"))
    results.append(check_optional_token("SMF_GITHUB_REPO", "publishing model releases"))
    return results


def summarize(results: List[CheckResult]) -> int:
    """Prints a pass/warn/fail table; returns the process exit code (0 unless something failed - warnings never block)."""
    width = max((len(r.name) for r in results), default=10)
    for result in results:
        print(f"[{result.level.upper():4}] {result.name:<{width}}  {result.detail}")
    fails = [r for r in results if r.level == "fail"]
    warns = [r for r in results if r.level == "warn"]
    passes = len(results) - len(fails) - len(warns)
    print(f"\n{passes} passed, {len(warns)} warnings, {len(fails)} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preflight check: is this environment ready to download data and train models?")
    parser.parse_args()
    sys.exit(summarize(run_all()))
