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

# Auto-load SMF_SSH_TARGET/SMF_REMOTE_ARCHIVE_DIR/SMF_DATA_ROOT from .env if
# not already set in the environment - python-dotenv does the equivalent for
# every Python script in this repo (see paths.py). Not a blind `source .env`:
# this repo's .env has `KEY = 'value'` spacing elsewhere (see
# SYNOPTIC_API_TOKEN) that isn't valid bash assignment syntax, and sourcing
# would also execute anything else unlucky enough to land in that file.
read_env_var() {
  local key="$1" env_file="$REPO_ROOT/.env" line value
  [ -f "$env_file" ] || return 0
  line="$(grep -E "^[[:space:]]*${key}[[:space:]]*=" "$env_file" | tail -n1)"
  [ -n "$line" ] || return 0
  value="${line#*=}"
  value="$(echo "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  value="${value%\"}"; value="${value#\"}"
  value="${value%\'}"; value="${value#\'}"
  printf '%s' "$value"
}

: "${SMF_SSH_TARGET:=$(read_env_var SMF_SSH_TARGET)}"
: "${SMF_REMOTE_ARCHIVE_DIR:=$(read_env_var SMF_REMOTE_ARCHIVE_DIR)}"
: "${SMF_DATA_ROOT:=$(read_env_var SMF_DATA_ROOT)}"

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

# Windows (Git Bash) quirks, confirmed live 2026-10-03 - both are no-ops on
# native Linux/macOS where cygpath doesn't exist:
#
# 1. Git Bash auto-translates POSIX-looking arguments (e.g. the remote
#    path /mnt/adrive/...) into Windows paths before handing them to a
#    non-MSYS-aware exe, which corrupts the "host:path" remote spec and
#    makes rsync think both sides are remote. MSYS_NO_PATHCONV=1 disables
#    that auto-translation for this script's rsync call.
# 2. Chocolatey's "rsync" package is a Cygwin build. Pairing it with Git
#    Bash's own MSYS-built ssh (whatever "ssh" resolves to on PATH) fails
#    with a native "dup() in/out/err failed" error - the two POSIX
#    runtimes handle file-descriptor inheritance differently. Fix: use the
#    ssh.exe bundled right next to rsync.exe instead (same Cygwin
#    runtime). That ssh.exe also keeps its own known_hosts under a
#    runtime-specific HOME, so point it at /dev/null and auto-accept
#    rather than fighting cross-runtime home-directory resolution.
RSYNC_SSH="ssh"
if command -v cygpath >/dev/null 2>&1; then
  export MSYS_NO_PATHCONV=1
  # "command -v rsync" on this machine resolves to Chocolatey's shim
  # (ProgramData/chocolatey/bin/rsync.exe, a tiny .NET forwarder with no
  # ssh.exe next to it) rather than the real Cygwin binary it forwards to
  # (.../lib/rsync/tools/bin/rsync.exe, which DOES ship a matching
  # ssh.exe) - so check both the PATH rsync's own directory and the
  # known Chocolatey lib layout.
  CANDIDATE_SSH_DIRS=()
  RSYNC_BIN="$(command -v rsync || true)"
  [ -n "$RSYNC_BIN" ] && CANDIDATE_SSH_DIRS+=("$(dirname "$RSYNC_BIN")")
  CANDIDATE_SSH_DIRS+=("/c/ProgramData/chocolatey/lib/rsync/tools/bin")
  for dir in "${CANDIDATE_SSH_DIRS[@]}"; do
    if [ -f "$dir/ssh.exe" ]; then
      WIN_SSH_PATH="$(cygpath -w "$dir/ssh.exe" | sed 's#\\#/#g')"
      RSYNC_SSH="$WIN_SSH_PATH -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null"
      break
    fi
  done
fi

ZIPS_DIR="$DATA_ROOT/archive_zips"
mkdir -p "$ZIPS_DIR"

echo "Pulling ${SSH_TARGET}:${REMOTE_ARCHIVE_DIR} -> ${ZIPS_DIR}/"
# Destination given as "." from inside ZIPS_DIR, not an absolute path:
# Cygwin rsync and MSYS bash don't share a mount table, so an absolute
# MSYS-form path (e.g. /d/foo) as the LOCAL destination fails to resolve
# the same way the remote spec did above. A relative path works because
# the process's working directory is inherited directly from Windows,
# bypassing path-translation entirely. No-op cd on Linux/macOS.
(
  cd "$ZIPS_DIR"
  rsync -av --partial --delay-updates --progress \
    -e "$RSYNC_SSH" \
    "${SSH_TARGET}:${REMOTE_ARCHIVE_DIR}" \
    .
)

echo ""
echo "Unpacking..."
cd "$REPO_ROOT"
if [ -z "${PYTHON:-}" ]; then
  # Windows ships a "python3" App Execution Alias that's present on PATH
  # (command -v finds it) but not functional - running it just prints a
  # Microsoft Store redirect message and exits. Verify a candidate
  # actually runs rather than merely exists; "python" is the real,
  # venv-activated interpreter here.
  PYTHON=python
  for candidate in python3 python; do
    if "$candidate" --version >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  done
fi
"$PYTHON" pipelines/unpack_archive_zip.py

echo ""
echo "Done."
