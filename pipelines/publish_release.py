"""
Publish a locally-trained beta model candidate as a GitHub pre-release, so
it can reach the server via import_model.py without any shared filesystem.

Requires the `gh` CLI, authenticated (`gh auth login`, or reuse the
server's existing GITHUB_TOKEN as GH_TOKEN in the environment).

Usage:
    python pipelines/publish_release.py --model fuel_moisture --repo youruser/ShowMeFire-Models
    python pipelines/publish_release.py --model fuel_moisture --version 1.5.0-beta.1 --repo youruser/ShowMeFire-Models
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.versioning import get_model_entry, load_active_model_path
from models.model_types import KNOWN_MODEL_TYPES
import paths


def publish(model_type, version=None, repo=None):
    repo = repo or os.getenv("SMF_GITHUB_REPO")
    if not repo:
        raise SystemExit("No target repo - pass --repo or set SMF_GITHUB_REPO (e.g. youruser/ShowMeFire-Models)")

    entry = get_model_entry(model_type)
    beta = entry.get("beta")
    if not beta:
        raise SystemExit(f"No beta candidate registered for {model_type!r} - train one first (pipelines/train_model.py)")
    if version and beta["version"] != version:
        raise SystemExit(f"Requested version {version!r} does not match current beta {beta['version']!r}")

    tag = f"{model_type}-v{beta['version']}"

    with tempfile.TemporaryDirectory() as tmp:
        meta_path = Path(tmp) / "metadata.json"
        asset_paths = []
        published_assets = {}
        if beta.get("assets"):
            for role, asset in beta["assets"].items():
                path = paths.DATA_ROOT / asset["file"]
                asset_paths.append(str(path))
                published_assets[role] = {**asset, "filename": path.name}
        else:
            # Only single-file candidates (no `assets` dict) rely on beta["file"]
            # via load_active_model_path - multi-asset bundles that lack a
            # model/checkpoint/static_bundle role leave beta["file"] as None,
            # so calling this unconditionally would crash before ever reaching
            # the assets branch above.
            asset_paths.append(str(load_active_model_path(model_type, channel="beta")))
        meta_path.write_text(json.dumps({
            "model_type": model_type,
            "version": beta["version"],
            "performance": beta.get("performance", {}),
            "trained_at": beta.get("trained_at"),
            # Preserve the complete training contract for the API-side
            # promotion gates. Without this, imported candidates appear to
            # have good scores but are missing all required metadata.
            "model_metadata": beta.get("metadata", {}),
            "assets": published_assets,
        }, indent=2))

        exists = subprocess.run(
            ["gh", "release", "view", tag, "--repo", repo],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
        if exists:
            # The artifact/version is unchanged; refresh the existing release
            # so its metadata.json gains the repaired promotion contract.
            cmd = ["gh", "release", "upload", tag, *asset_paths, str(meta_path),
                   "--repo", repo, "--clobber"]
        else:
            cmd = [
                "gh", "release", "create", tag,
                *asset_paths, str(meta_path),
                "--repo", repo,
                "--prerelease",
                "--title", f"{model_type} {beta['version']}",
                "--notes", f"Beta candidate for {model_type}. Performance: {json.dumps(beta.get('performance', {}))}",
            ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    print(f"\nPublished {tag} as a pre-release on {repo}.")
    print(f"Promote when ready with: gh release edit {tag} --repo {repo} --prerelease=false")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish a beta model candidate as a GitHub pre-release")
    parser.add_argument("--model", required=True, choices=sorted(KNOWN_MODEL_TYPES))
    parser.add_argument("--version", default=None, help="Beta version to publish (defaults to the current beta)")
    parser.add_argument("--repo", default=None, help="owner/repo (defaults to SMF_GITHUB_REPO env var)")
    args = parser.parse_args()

    publish(args.model, version=args.version, repo=args.repo)
