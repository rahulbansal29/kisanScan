"""Compare base, fine-tuned and combined (average of both) models on PlantDoc test + demo samples.

Run from kisanscan/:  .venv/Scripts/python training/compare.py
"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from transformers import AutoModelForImageClassification, MobileNetV2ImageProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diseases import DISEASES  # noqa: E402
from training.common import BASE_MODEL, FINETUNED, ROOT, SIZE, Normalizer, centre, load_split  # noqa: E402

norm = Normalizer(MobileNetV2ImageProcessor.from_pretrained(BASE_MODEL))
base = AutoModelForImageClassification.from_pretrained(BASE_MODEL).eval()
new = AutoModelForImageClassification.from_pretrained(FINETUNED).eval()
crops = [DISEASES[i]["crop"] for i in range(len(DISEASES))]


@torch.no_grad()
def probs(x):
    pb, pn = base(pixel_values=x).logits.softmax(-1), new(pixel_values=x).logits.softmax(-1)
    return {"base": pb, "new": pn, "combined": (pb + pn) / 2}


def pick(p, crop=None):
    if crop:
        p = p * torch.tensor([c == crop for c in crops])
    return int(p.argmax())


train, _ = load_split("train")
test, _ = load_split("test", exclude_hashes={h for _, _, h in train})
score = {k: [0, 0] for k in ("base", "new", "combined")}
for i in range(0, len(test), 64):
    chunk = test[i:i + 64]
    out = probs(torch.stack([norm(centre(im)) for im, _, _ in chunk]))
    for k, p in out.items():
        for row, (_, cid, _) in zip(p, chunk):
            score[k][0] += pick(row) == cid
            score[k][1] += pick(row, crops[cid]) == cid

result = {"field_test_photos": len(test)}
for k, (a, b) in score.items():
    result[k] = {"field_acc": round(a / len(test), 4), "field_acc_crop_chosen": round(b / len(test), 4)}
    print(f"{k:9s} field photos {a / len(test):.1%}   with crop chosen {b / len(test):.1%}")

result["samples"] = {}
for p in sorted((ROOT / "sample_leaves").glob("*.jpg")):
    true_crop = DISEASES[[i for i in range(len(DISEASES))
                          if p.name.lower().startswith(DISEASES[i]["crop"].split()[0].lower())][0]]["crop"]
    x = norm(centre(ImageOps.fit(Image.open(p).convert("RGB"), (SIZE, SIZE), Image.BILINEAR)))[None]
    out = probs(x)
    row = {}
    for k, pr in out.items():
        j = pick(pr[0], true_crop)
        q = pr[0] * torch.tensor([c == true_crop for c in crops])
        row[k] = f"{DISEASES[j]['name_en']} {float(q[j] / q.sum()):.0%}"
    result["samples"][p.name] = row
    print(p.name, row)

json.dump(result, open(FINETUNED / "compare.json", "w"), indent=2)
