#!/usr/bin/env bash
# Pull newly-bundled archive zips (api/data_archive_day/*.zip on the server)
# down to this machine's training data folder, then unpack them.
#
# Plain bash + rsync + ssh - works on Ubuntu (including WSL), macOS, or any
# Linux box with those installed. On a fresh Ubuntu/WSL install you may need:
#   sudo apt update && sudo apt install -y rsync openssh-client python3-pip
#
# Usage:
#   ./scripts/pull_archives.sh user@server /remote/path/to/api/data_archive_day/
#
# Or set env vars once (e.g. in ~/.bashrc) and just run with no args:
#   export SMF_SSH_TARGET=user@server
#   export SMF_REMOTE_ARCHIVE_DIR=/remote/path/to/api/data_archive_day/
#   ./scripts/pull_archives.sh
#
# SMF_DATA_ROOT (optional) - where training data lives locally, same as
# paths.py; defaults to ./data next to this repo if unset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

SSH_TARGET="${1:-${SMF_SSH_TARGET:-}}"
REMOTE_ARCHIVE_DIR="${2:-${SMF_REMOTE_ARCHIVE_DIR:-}}"

if [ -z "$SSH_TARGET" ] || [ -z "$REMOTE_ARCHIVE_DIR" ]; then
  echo "Usage: $0 user@host /remote/path/to/api/data_archive_day/" >&2
  echo "   (or set SMF_SSH_TARGET / SMF_REMOTE_ARCHIVE_DIR)" >&2
  exit 1
fi

# Ensure the remote path ends in / so rsync copies contents, not the dir itself.
case "$REMOTE_ARCHIVE_DIR" in
  */) ;;
  *) REMOTE_ARCHIVE_DIR="${REMOTE_ARCHIVE_DIR}/" ;;
esac

DATA_ROOT="${SMF_DATA_ROOT:-$REPO_ROOT/data}"
ZIPS_DIR="$DATA_ROOT/archive_zips"
mkdir -p "$ZIPS_DIR"

echo "Pulling ${SSH_TARGET}:${REMOTE_ARCHIVE_DIR} -> ${ZIPS_DIR}/"
rsync -av --inplace --partial --progress \
  -e ssh \
  "${SSH_TARGET}:${REMOTE_ARCHIVE_DIR}" \
  "${ZIPS_DIR}/"

echo ""
echo "Unpacking..."
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python3}"
"$PYTHON" pipelines/unpack_archive_zip.py

echo ""
echo "Done."
