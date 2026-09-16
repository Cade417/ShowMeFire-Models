"""Print the latest V4 gate without mutating a registry."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent));import paths
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--evaluation",type=Path,default=paths.REPORTS_DIR/"v4_prospective_evaluation.json");args=parser.parse_args()
    report=json.loads(args.evaluation.read_text());value={key:report.get(key) for key in ("status","pass","beta_registration_allowed","checks","days","runs","samples","manifest_sha256")}
    print(json.dumps(value,indent=2));raise SystemExit(0 if report.get("pass") else 2)
if __name__=="__main__":main()
