"""Pull only the WLASL100 videos out of the remote gazquez zip (ranged requests, no full download)."""
import struct, sys, json, zlib, urllib.request, pathlib, shutil
from concurrent.futures import ThreadPoolExecutor

API = "https://www.kaggle.com/api/v1/datasets/download/gazquez/wlasl-processed"
HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / "wlasl-processed"  # only used to copy the two json files
OUT = HERE / "wlasl-full"
(OUT / "videos").mkdir(parents=True, exist_ok=True)
for f in ("WLASL_v0.3.json", "nslt_100.json"):
    shutil.copy(SRC / f, OUT / f)


def resolve():
    r = urllib.request.urlopen(urllib.request.Request(API, headers={"Range": "bytes=0-0"}))
    return r.geturl(), int(r.headers["Content-Range"].split("/")[1])


url, size = resolve()
get = lambda a, b: urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})).read()
tail = get(size - 12 * 1024 * 1024, size - 1)
i = tail.rfind(b"PK\x05\x06")
_, _, _, _, n_tot, cd_size, cd_off, _ = struct.unpack("<IHHHHIIH", tail[i:i + 22])
cd = tail[len(tail) - (size - cd_off):][:cd_size]

ids = {k.zfill(5) for k in json.load(open(SRC / "nslt_100.json"))}
todo, p = {}, 0
while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
    method = struct.unpack("<H", cd[p + 10:p + 12])[0]
    csz, usz = struct.unpack("<II", cd[p + 20:p + 28])
    nl, el, cl = struct.unpack("<HHH", cd[p + 28:p + 34])
    off = struct.unpack("<I", cd[p + 42:p + 46])[0]
    name = cd[p + 46:p + 46 + nl].decode("utf8", "replace")
    ex = cd[p + 46 + nl:p + 46 + nl + el]
    if 0xFFFFFFFF in (usz, csz, off):
        q = 0
        while q + 4 <= len(ex):
            hid, ln = struct.unpack("<HH", ex[q:q + 4])
            if hid == 1:
                vals = list(struct.unpack("<" + "Q" * (ln // 8), ex[q + 4:q + 4 + ln])); k = 0
                if usz == 0xFFFFFFFF: usz = vals[k]; k += 1
                if csz == 0xFFFFFFFF: csz = vals[k]; k += 1
                if off == 0xFFFFFFFF: off = vals[k]
                break
            q += 4 + ln
    if name.lower().endswith(".mp4") and name.split("/")[-1][:-4].zfill(5) in ids and "/videos/" in name:
        todo[name.split("/")[-1]] = (off, csz, usz, method, nl)
    p += 46 + nl + el + cl
print("to fetch:", len(todo), "of", len(ids), flush=True)


def fetch(item):
    global url
    fn, (off, csz, usz, method, nl) = item
    dest = OUT / "videos" / fn
    if dest.exists() and dest.stat().st_size == usz:
        return fn, "skip"
    for attempt in range(4):
        try:
            b = get(off, off + 30 + nl + 512 + csz)
            assert b[:4] == b"PK\x03\x04", "bad local header"
            lnl, lel = struct.unpack("<HH", b[26:30])
            data = b[30 + lnl + lel:30 + lnl + lel + csz]
            assert len(data) == csz, "short read"
            raw = data if method == 0 else zlib.decompressobj(-15).decompress(data)
            assert len(raw) == usz, "size mismatch"
            tmp = dest.with_suffix(".part"); tmp.write_bytes(raw); tmp.replace(dest)
            return fn, "ok"
        except Exception as e:
            err = str(e)
            if "403" in err or "401" in err:  # signed link expired
                url, _ = resolve()
    return fn, "FAIL " + err


done = bad = 0
with ThreadPoolExecutor(8) as ex:
    for fn, st in ex.map(fetch, todo.items()):
        done += 1
        if st.startswith("FAIL"):
            bad += 1; print(fn, st, flush=True)
        if done % 50 == 0:
            print(f"{done}/{len(todo)} (failed {bad})", flush=True)
print("finished", done, "failed", bad)
