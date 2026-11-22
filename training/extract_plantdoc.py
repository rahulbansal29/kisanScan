"""Unzip PlantDoc train/test photos into ../datasets/plantdoc with Windows-safe file names.

Usage (from kisanscan/): .venv/Scripts/python -I training/extract_plantdoc.py <zip> <out_dir>
"""
import re
import sys
import zipfile
from pathlib import Path

zip_path, out = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
n = 0
with zipfile.ZipFile(zip_path) as z:
    for m in z.infolist():
        parts = m.filename.split("/")[1:]  # drop the top "PlantDoc-Dataset-master" folder
        if not parts or parts[0] not in ("train", "test") or m.is_dir():
            continue
        parts = [re.sub(r'[<>:"|?*\\]', "_", p).strip() or "_" for p in parts]
        dest = (out / "/".join(parts)).resolve()
        if out not in dest.parents:  # never write outside out_dir
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(z.read(m))
        n += 1
print("extracted", n)
