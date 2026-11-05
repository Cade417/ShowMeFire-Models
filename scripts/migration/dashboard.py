"""Local-only progress dashboard for the R2 + server migration.

Read-only: this process never touches R2 or the data being pulled - it
just polls migration_status.json (written by r2_inventory.py, r2_pull.py,
server_pull.py, r2_delete.py) and serves it, following the same pattern as
api/services/forecast_v1_job.py + api/public/dashboard.html.

Run with:
    python scripts/migration/run_migration.py dashboard
    (or directly: python scripts/migration/dashboard.py)
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[2]))
import status as mstatus

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()
HTML_PATH = Path(__file__).resolve().parent / "dashboard.html"


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PATH.read_text(encoding="utf-8")


@app.get("/status")
def status_endpoint():
    return mstatus.read_status()


def serve(port=8765):
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    serve()
