"""KisanScan - AI crop disease detector (prototype).

Run:  .venv/Scripts/python app.py   then open http://localhost:5000
"""
import io
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from transformers import AutoModelForImageClassification, MobileNetV2ImageProcessor

from diseases import CROPS, DISEASES

MODEL_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
LOW_CONFIDENCE = 0.50  # below this we ask the farmer to retake the photo
MIN_LEAF_SHARE = 0.08  # real leaf photos score 0.2-0.6; noise / non-plant photos ~0
MIN_CROP_SHARE = 0.10  # chosen crop gets <0.06 on wrong-crop samples, >0.8 on right ones; below this, ask to check the crop
DB_PATH = Path(__file__).with_name("scans.db")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

print("Loading AI model (first run downloads ~10 MB)...")
processor = MobileNetV2ImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID).eval()
print("Model ready.")


def db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, class_id INTEGER,
        confidence REAL, crop TEXT, disease TEXT, healthy INTEGER)""")
    return con


def leaf_share(image: Image.Image) -> float:
    """Share of plant-coloured pixels (green to yellow-brown, not grey). Used to reject non-leaf photos."""
    hsv = np.asarray(image.convert("RGB").resize((128, 128)).convert("HSV")).astype(float)
    h, s, v = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    return float(((h >= 20) & (h <= 170) & (s > 0.18) & (v > 0.15)).mean())


def predict(image: Image.Image, crop: str | None = None):
    """Top-3 (class_id, prob). With a crop chosen by the farmer, only that crop's classes are
    considered; also returns how much of the model's belief fell on that crop (crop_share)."""
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        probs = model(**inputs).logits.softmax(-1)[0]
    crop_share = 1.0
    if crop:
        mask = torch.tensor([DISEASES[i]["crop"] == crop for i in range(len(probs))])
        crop_share = float(probs[mask].sum())
        probs = torch.where(mask, probs, torch.zeros_like(probs)) / max(crop_share, 1e-9)
    k = min(3, int((probs > 0).sum()))
    top = torch.topk(probs, k)
    return [(int(i), float(p)) for p, i in zip(top.values, top.indices)], crop_share


@app.get("/")
def index():
    return render_template("index.html", crops=CROPS)


SAMPLES_DIR = Path(__file__).with_name("sample_leaves")


@app.get("/api/samples")
def api_samples():
    return jsonify(sorted(p.name for p in SAMPLES_DIR.glob("*.jpg")))


@app.get("/samples/<path:name>")
def sample_file(name):
    return send_from_directory(SAMPLES_DIR, name)


@app.post("/api/predict")
def api_predict():
    file = request.files.get("image")
    if not file:
        return jsonify(error="No image uploaded"), 400
    try:
        image = Image.open(io.BytesIO(file.read()))
    except UnidentifiedImageError:
        return jsonify(error="That file is not an image. Please upload a JPG or PNG photo."), 400

    crop = request.form.get("crop") or None
    if crop == "other":
        return jsonify(unsupported=True, crops=[{"en": c, "hi": h} for c, h in CROPS.items()])
    if crop and crop not in CROPS:
        return jsonify(error="Unknown crop"), 400

    if leaf_share(image) < MIN_LEAF_SHARE:
        return jsonify(not_leaf=True)

    top3, crop_share = predict(image, crop)
    if crop and crop_share < MIN_CROP_SHARE:
        # The photo does not look like the crop the farmer chose (often a crop we don't support yet).
        guess, _ = predict(image)
        g = DISEASES[guess[0][0]]
        return jsonify(wrong_crop=True, crop=crop, crop_hi=CROPS[crop],
                       looks_like=g["crop"], looks_like_hi=g["crop_hi"])
    best_id, best_p = top3[0]
    info = DISEASES[best_id]

    with db() as con:
        con.execute("INSERT INTO scans (ts, class_id, confidence, crop, disease, healthy) VALUES (?,?,?,?,?,?)",
                    (datetime.now().isoformat(timespec="seconds"), best_id, best_p,
                     info["crop"], info["name_en"], int(info["healthy"])))

    return jsonify(
        result={**info, "confidence": round(best_p * 100, 1)},
        low_confidence=best_p < LOW_CONFIDENCE,
        alternatives=[{"name_en": DISEASES[i]["name_en"], "name_hi": DISEASES[i]["name_hi"],
                       "confidence": round(p * 100, 1)} for i, p in top3[1:]],
    )


@app.get("/api/history")
def api_history():
    with db() as con:
        rows = con.execute("SELECT ts, crop, disease, confidence, healthy FROM scans "
                           "ORDER BY id DESC LIMIT 20").fetchall()
        stats = con.execute("SELECT disease, COUNT(*) FROM scans WHERE healthy=0 "
                            "GROUP BY disease ORDER BY 2 DESC LIMIT 5").fetchall()
        total, sick = con.execute("SELECT COUNT(*), COALESCE(SUM(1-healthy),0) FROM scans").fetchone()
    return jsonify(
        scans=[dict(ts=r[0], crop=r[1], disease=r[2], confidence=round(r[3] * 100, 1), healthy=bool(r[4]))
               for r in rows],
        top_diseases=[dict(disease=d, count=c) for d, c in stats],
        total=total, diseased=sick,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
