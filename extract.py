import argparse, os, time, traceback
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
from config import LM, SPLIT_CSV, RUNS

H = None


def init():
    global H
    import cv2, mediapipe as mp
    cv2.setNumThreads(1)
    H = mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1, refine_face_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)


def arr(lms, n):
    if lms is None:
        return np.full((n, 3), np.nan, np.float32)
    return np.array([[p.x, p.y, p.z] for p in lms.landmark], np.float32)


def _work(a):
    import cv2
    vid, path, fs, fe = a
    out = LM / f"{vid}.npz"
    if out.exists():
        return vid, "skip", 0, 0.0
    t0 = time.time()
    try:
        H.reset()
    except Exception:
        pass
    start = max(int(fs) - 1, 0)                       # metadata is 1-indexed
    end = int(fe) - 1 if int(fe) > 0 else None        # -1 means until the end
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return vid, "error: cannot open video", 0, 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    d = {k: [] for k in ("lh", "rh", "pose", "face")}
    i = 0
    while True:
        ok, f = cap.read()
        if not ok or (end is not None and i > end):
            break
        if i >= start:
            r = H.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            d["lh"].append(arr(r.left_hand_landmarks, 21))
            d["rh"].append(arr(r.right_hand_landmarks, 21))
            d["pose"].append(arr(r.pose_landmarks, 33)[:25])   # upper body
            d["face"].append(arr(r.face_landmarks, 468))
        i += 1
    cap.release()
    n = len(d["pose"])
    if n == 0:
        return vid, "empty", 0, time.time() - t0
    tmp = LM / f"{vid}.tmp.npz"
    np.savez_compressed(tmp, fps=fps, **{k: np.stack(v).astype(np.float16) for k, v in d.items()})
    os.replace(tmp, out)
    return vid, "ok", n, time.time() - t0


def work(a):
    try:
        return _work(a)
    except Exception:
        return a[0], "error: " + traceback.format_exc().strip().splitlines()[-1], 0, 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    df = pd.read_csv(SPLIT_CSV, dtype={"video_id": str})
    if a.limit:
        df = df.head(a.limit)
    jobs = list(zip(df.video_id, df.path, df.frame_start, df.frame_end))
    print(f"jobs: {len(jobs)} | workers: {a.workers} | output: {LM}", flush=True)

    t0, res = time.time(), []
    if a.workers <= 1:
        init()
        for j in tqdm(jobs):
            res.append(work(j))
    else:
        with Pool(a.workers, initializer=init) as p:
            for r in tqdm(p.imap_unordered(work, jobs), total=len(jobs)):
                res.append(r)
    wall = time.time() - t0

    log = pd.DataFrame(res, columns=["video_id", "status", "frames", "seconds"])
    log.to_csv(RUNS / "extract_log.csv", index=False)
    print(log.status.value_counts().to_string())
    ok = log[log.frames > 0]
    if len(ok):
        print(f"wall {wall/60:.1f} min | frames {ok.frames.sum()} | "
              f"{ok.frames.sum()/max(wall,1e-9):.1f} fps overall with {a.workers} workers | "
              f"{ok.seconds.sum()/ok.frames.sum()*1000:.0f} ms/frame per core")