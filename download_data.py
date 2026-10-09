import os
import urllib.request

SAVE_DIR = "cwru_fault_data"

# CWRU 12k Drive End Bearing Fault Data + Normal Data
FILES = {
    # Normal Baseline (Healthy)
    "Normal_0.mat": "https://engineering.case.edu/sites/default/files/97.mat",
    "Normal_1.mat": "https://engineering.case.edu/sites/default/files/98.mat",
    "Normal_2.mat": "https://engineering.case.edu/sites/default/files/99.mat",
    "Normal_3.mat": "https://engineering.case.edu/sites/default/files/100.mat",
    
    # Inner Race Fault (0.007")
    "IR007_0.mat": "https://engineering.case.edu/sites/default/files/105.mat",
    "IR007_1.mat": "https://engineering.case.edu/sites/default/files/106.mat",
    "IR007_2.mat": "https://engineering.case.edu/sites/default/files/107.mat",
    "IR007_3.mat": "https://engineering.case.edu/sites/default/files/108.mat",
    
    # Ball Fault (0.007")
    "B007_0.mat": "https://engineering.case.edu/sites/default/files/118.mat",
    "B007_1.mat": "https://engineering.case.edu/sites/default/files/119.mat",
    "B007_2.mat": "https://engineering.case.edu/sites/default/files/120.mat",
    "B007_3.mat": "https://engineering.case.edu/sites/default/files/121.mat",
    
    # Outer Race Fault (0.007", Centered @ 6:00)
    "OR007_6_0.mat": "https://engineering.case.edu/sites/default/files/130.mat",
    "OR007_6_1.mat": "https://engineering.case.edu/sites/default/files/131.mat",
    "OR007_6_2.mat": "https://engineering.case.edu/sites/default/files/132.mat",
    "OR007_6_3.mat": "https://engineering.case.edu/sites/default/files/133.mat",
}

# Larger defect sizes (0.014" and 0.021"), same fault types and loads
_BASE = "https://engineering.case.edu/sites/default/files/{}.mat"
for _prefix, _ids in {
    "IR014": [169, 170, 171, 172],
    "B014": [185, 186, 187, 188],
    "OR014_6": [197, 198, 199, 200],
    "IR021": [209, 210, 211, 212],
    "B021": [222, 223, 224, 225],
    "OR021_6": [234, 235, 236, 237],
}.items():
    for _load, _rid in enumerate(_ids):
        FILES[f"{_prefix}_{_load}.mat"] = _BASE.format(_rid)

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}


def recording_id(filename):
    """CWRU recording number for a local file, e.g. 'Normal_2.mat' -> '099'."""
    return os.path.splitext(os.path.basename(FILES[filename]))[0].zfill(3)


def download_all():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Total files to download: {len(FILES)}")
    for filename, url in FILES.items():
        filepath = os.path.join(SAVE_DIR, filename)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
            print(f"[SKIP] {filename} already exists ({os.path.getsize(filepath)} bytes)")
            continue
        print(f"[DOWNLOADING] {filename} from {url}...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
                out_file.write(response.read())
            print(f"[OK] {filename} downloaded successfully ({os.path.getsize(filepath)} bytes)")
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}")

    print("\nDownload process completed!")


if __name__ == "__main__":
    download_all()
