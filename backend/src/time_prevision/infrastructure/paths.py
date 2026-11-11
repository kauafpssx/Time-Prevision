from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = BACKEND_ROOT.parent
DATA_DIR = BACKEND_ROOT / "data"
DATABASE_PATH = DATA_DIR / "time_prevision.db"
MODEL_DIR = DATA_DIR / "model"
