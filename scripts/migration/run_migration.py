#!/usr/bin/env python3
"""One-command entrypoint for the R2 + server -> local HDD migration.

    python scripts/migration/run_migration.py inventory
    python scripts/migration/run_migration.py pull
    python scripts/migration/run_migration.py pull --limit 50     # smoke test
    python scripts/migration/run_migration.py dashboard           # serve the progress UI

Deletion from R2 is deliberately NOT part of this orchestrator - once the
dashboard shows everything pulled and verified, run r2_delete.py yourself
as a separate, explicit, reviewed step (dry run by default).

Point SMF_DATA_ROOT (in model-training/.env) at the external HDD before
running anything here - every path in paths.py, including this tool's
migration/ and r2_pull/ subfolders, is anchored under it.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[2]))
import r2_inventory
import r2_pull
import server_pull


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("inventory", help="List everything in the R2 bucket into inventory.json")

    pull_parser = sub.add_parser("pull", help="Pull R2 objects + the server's archive dir to local disk")
    pull_parser.add_argument("--limit", type=int, help="Only pull the first N R2 objects (smoke test)")
    pull_parser.add_argument("--skip-r2", action="store_true", help="Skip the R2 pull")
    pull_parser.add_argument("--skip-server", action="store_true", help="Skip the SSH server archive pull")

    dashboard_parser = sub.add_parser("dashboard", help="Serve the local progress dashboard")
    dashboard_parser.add_argument("--port", type=int, default=8765)

    args = parser.parse_args()

    if args.command == "inventory":
        r2_inventory.build_inventory()
    elif args.command == "pull":
        if not args.skip_server:
            server_pull.run()
        if not args.skip_r2:
            r2_pull.pull(limit=args.limit)
    elif args.command == "dashboard":
        import dashboard
        print(f"Dashboard at http://127.0.0.1:{args.port}")
        dashboard.serve(port=args.port)


if __name__ == "__main__":
    main()
