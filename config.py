import os
from pathlib import Path

# data/ (landmarks, split.csv) and runs/ live under ROOT. Default: the folder above this repo; on Colab set WLASL_ROOT.
ROOT = Path(os.environ.get("WLASL_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"
RAW = Path(os.environ.get("WLASL_RAW", Path(__file__).resolve().parent / "wlasl-full"))  # WLASL100 videos + json, made by fetch_wlasl100.py
LM = DATA / "landmarks"
RUNS = ROOT / "runs"
META_CSV = DATA / "metadata.csv"
SPLIT_CSV = DATA / "split.csv"

# RAW (videos + json) is only needed by build_metadata.py, fetch_wlasl100.py and the I3D frame cache.
LM.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)


def face_indices():
    """Return (brows+eyes indices, lips indices) of the 468-pt face mesh (frozen in mp_constants.py)."""
    from mp_constants import BROWS_EYES, LIPS

    return BROWS_EYES, LIPS
