import argparse, csv, json, math, subprocess, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from config import SPLIT_CSV, RUNS, LM
from dataset import SignDataset
from model import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--mods", default="H")            # letters from H P F L D
ap.add_argument("--model", default="transformer")  # transformer | lstm
ap.add_argument("--T", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--overfit", action="store_true", help="sanity: overfit 32 train clips, no aug")
ap.add_argument("--shuffle_labels", action="store_true", help="sanity: should give chance acc")
a = ap.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
name = f"{a.model}_{a.mods}_T{a.T}_s{a.seed}" + ("_overfit" if a.overfit else "") + ("_shuf" if a.shuffle_labels else "")
out = RUNS / name
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SPLIT_CSV, dtype={"video_id": str})
have = df.video_id.map(lambda v: (LM / f"{v}.npz").exists())
print(f"clips with landmarks: {have.sum()}/{len(df)}")
df = df[have]
tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
if a.overfit:
    tr = tr.head(32)
ds_tr = SignDataset(tr, a.mods, a.T, train=not a.overfit, seed=a.seed, shuffle_labels=a.shuffle_labels)
ds_va = SignDataset(va, a.mods, a.T)
ds_te = SignDataset(te, a.mods, a.T)
dl_tr = DataLoader(ds_tr, a.bs, shuffle=True, drop_last=len(ds_tr) > a.bs)
dl_va, dl_te = DataLoader(ds_va, 64), DataLoader(ds_te, 64)

NC = 100
model = build_model(a.model, ds_tr.dims, NC, a.T).to(dev)
n_params = sum(p.numel() for p in model.parameters())
print(name, "| device", dev, "| params", n_params, "| dims", ds_tr.dims)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
warm = 5
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda e: (e + 1) / warm if e < warm else 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, a.epochs - warm))))
crit = nn.CrossEntropyLoss(label_smoothing=0.1)


def to_dev(x):
    return {k: v.to(dev) for k, v in x.items()}


@torch.no_grad()
def evaluate(dl):
    model.eval()
    L, Y = [], []
    for x, y in dl:
        L.append(model(to_dev(x)).cpu()); Y.append(y)
    return torch.cat(L), torch.cat(Y)


def metrics(logits, y):
    top5 = logits.topk(5, 1).indices
    c1, c5 = top5[:, 0] == y, (top5 == y[:, None]).any(1)
    cls = y.unique().tolist()
    return dict(inst_top1=c1.float().mean().item(), inst_top5=c5.float().mean().item(),
                cls_top1=float(np.mean([c1[y == k].float().mean().item() for k in cls])),
                cls_top5=float(np.mean([c5[y == k].float().mean().item() for k in cls])),
                loss=nn.functional.cross_entropy(logits, y).item())


best, best_ep, t0 = -1, -1, time.time()
log = open(out / "log.csv", "w", newline=""); w = csv.writer(log)
w.writerow(["epoch", "train_loss", "train_acc", "val_top1", "val_loss"])
for ep in range(a.epochs):
    model.train(); tl, tc, n = 0, 0, 0
    for x, y in dl_tr:
        x, y = to_dev(x), y.to(dev)
        lo = model(x); loss = crit(lo, y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tl += loss.item() * len(y); tc += (lo.argmax(1) == y).sum().item(); n += len(y)
    sched.step()
    lv, yv = evaluate(dl_va); mv = metrics(lv, yv)
    w.writerow([ep, tl / n, tc / n, mv["inst_top1"], mv["loss"]]); log.flush()
    if mv["inst_top1"] > best:
        best, best_ep = mv["inst_top1"], ep
        torch.save(model.state_dict(), out / "best.pt")
    if ep % 10 == 0 or ep == a.epochs - 1:
        print(f"ep {ep:3d} train_loss {tl/n:.3f} train_acc {tc/n:.3f} val_top1 {mv['inst_top1']:.3f}")
train_time = time.time() - t0
if a.overfit:
    raise SystemExit(f"overfit check: final train acc {tc/n:.3f} (want ~1.0)")

model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
lt, yt = evaluate(dl_te); mt = metrics(lt, yt)
mt.update(val_top1=best, best_epoch=best_ep, n_params=n_params, train_time_s=round(train_time, 1))
json.dump(mt, open(out / "metrics.json", "w"), indent=2)
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    commit = "n/a"
json.dump({**vars(a), "commit": commit, "device": dev}, open(out / "config.json", "w"), indent=2)
pd.DataFrame(dict(video_id=ds_te.vids, signer_id=te.signer_id.values, gloss=te.gloss.values,
                  label=yt.numpy(), pred=lt.argmax(1).numpy(),
                  top5=[" ".join(map(str, r)) for r in lt.topk(5, 1).indices.tolist()])
             ).to_csv(out / "test_preds.csv", index=False)
res = RUNS / "results.csv"
new = not res.exists()
with open(res, "a", newline="") as f:
    ww = csv.writer(f)
    if new:
        ww.writerow(["run", "mods", "model", "T", "seed", "val_top1", "test_inst_top1", "test_cls_top1",
                     "test_inst_top5", "test_cls_top5", "best_epoch", "params", "train_s"])
    ww.writerow([name, a.mods, a.model, a.T, a.seed, f"{best:.4f}", f"{mt['inst_top1']:.4f}", f"{mt['cls_top1']:.4f}",
                 f"{mt['inst_top5']:.4f}", f"{mt['cls_top5']:.4f}", best_ep, n_params, round(train_time, 1)])
print(json.dumps(mt, indent=2))