"""
Downloads and extracts the Paderborn University (KAt) bearing dataset.

Source: https://groups.uni-paderborn.de/kat/BearingDataCenter/
License: CC BY-NC 4.0 (non-commercial use; cite Lessmeier et al., PHM Society
European Conference 2016). Each bearing is one ~150-180 MB .rar archive of
MATLAB files (4 operating conditions x 20 measurements of 4 s at 64 kHz).

The three bearings with combined inner + outer race damage (KB23, KB24, KB27)
are skipped because the classifiers here assign a single damage class.
Extraction uses 7-Zip if installed, otherwise the system bsdtar.
"""
import os
import shutil
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "https://groups.uni-paderborn.de/kat/BearingDataCenter/{}.rar"
SAVE_DIR = os.path.join("paderborn_data", "raw")

BEARINGS = (
    [f"K00{i}" for i in range(1, 7)]                                         # healthy
    + ["KA01", "KA03", "KA04", "KA05", "KA06", "KA07", "KA08", "KA09",
       "KA15", "KA16", "KA22", "KA30"]                                       # outer ring damage
    + ["KI01", "KI03", "KI04", "KI05", "KI07", "KI08",
       "KI14", "KI16", "KI17", "KI18", "KI21"]                               # inner ring damage
)
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def extractor():
    for exe in ("7z", r"C:\Program Files\7-Zip\7z.exe"):
        path = shutil.which(exe) or (exe if os.path.exists(exe) else None)
        if path:
            return lambda archive, out: [path, "x", "-y", f"-o{out}", archive]
    tar = shutil.which("tar")
    if tar:
        return lambda archive, out: [tar, "-xf", archive, "-C", out]
    raise RuntimeError("Need 7-Zip or bsdtar to extract .rar archives.")


def fetch(bearing, extract_cmd):
    archive = os.path.join(SAVE_DIR, f"{bearing}.rar")
    if os.path.isdir(os.path.join(SAVE_DIR, bearing)) and os.listdir(os.path.join(SAVE_DIR, bearing)):
        return f"[SKIP] {bearing} already extracted"
    if not (os.path.exists(archive) and os.path.getsize(archive) > 100_000_000):
        req = urllib.request.Request(BASE_URL.format(bearing), headers=HEADERS)
        tmp = archive + ".part"
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 20)
        os.replace(tmp, archive)
    subprocess.run(extract_cmd(archive, SAVE_DIR), check=True, stdout=subprocess.DEVNULL)
    os.remove(archive)
    return f"[OK] {bearing} ({len(os.listdir(os.path.join(SAVE_DIR, bearing)))} files)"


def download_all(workers=3):
    os.makedirs(SAVE_DIR, exist_ok=True)
    cmd = extractor()
    with ThreadPoolExecutor(workers) as pool:
        for msg in pool.map(lambda b: fetch(b, cmd), BEARINGS):
            print(msg, flush=True)


if __name__ == "__main__":
    download_all()
