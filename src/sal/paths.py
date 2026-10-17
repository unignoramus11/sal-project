"""Project paths. Large artefacts live under data/ and outputs/ (both gitignored)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"
