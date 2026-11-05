import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from spatial.evaluate_v4 import main
if __name__=="__main__":main()
