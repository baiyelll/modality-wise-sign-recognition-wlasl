from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = ROOT / "wlasl-processed"  # where the Kaggle download lives
LM = DATA / "landmarks"
RUNS = ROOT / "runs"
META_CSV = DATA / "metadata.csv"
SPLIT_CSV = DATA / "split.csv"

if not RAW.exists():
    raise SystemExit(f"Dataset folder not found: {RAW}")
LM.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)


def face_indices():
    """Return (brows+eyes indices, lips indices) of the 468-pt face mesh."""
    import mediapipe as mp

    fm = mp.solutions.face_mesh

    def idx(*conns):
        return sorted({i for c in conns for e in c for i in e})

    lips = idx(fm.FACEMESH_LIPS)
    be = idx(
        fm.FACEMESH_LEFT_EYE,
        fm.FACEMESH_RIGHT_EYE,
        fm.FACEMESH_LEFT_EYEBROW,
        fm.FACEMESH_RIGHT_EYEBROW,
    )
    return be, lips
