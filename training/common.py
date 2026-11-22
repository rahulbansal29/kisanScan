"""Shared helpers for testing and fine-tuning KisanScan on PlantDoc (real field photos).

PlantDoc: https://github.com/pratikkayal/PlantDoc-Dataset (CC-BY-4.0), 27 classes, all of
which exist among the 38 PlantVillage classes the base model already predicts.
"""
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps

BASE_MODEL = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
ROOT = Path(__file__).resolve().parents[1]                     # kisanscan/
PLANTDOC = ROOT.parent / "datasets" / "plantdoc"               # unleash/datasets/plantdoc
FINETUNED = ROOT / "models" / "kisanscan-plantdoc"

# PlantDoc folder name -> class id of the base model (see diseases.py / model config)
PLANTDOC_TO_ID = {
    "Apple Scab Leaf": 0, "Apple leaf": 3, "Apple rust leaf": 2,
    "Bell_pepper leaf spot": 18, "Bell_pepper leaf": 19, "Blueberry leaf": 4, "Cherry leaf": 6,
    "Corn Gray leaf spot": 7, "Corn leaf blight": 9, "Corn rust leaf": 8, "Peach leaf": 17,
    "Potato leaf early blight": 20, "Potato leaf late blight": 21, "Raspberry leaf": 23,
    "Soyabean leaf": 24, "Squash Powdery mildew leaf": 25, "Strawberry leaf": 27,
    "Tomato Early blight leaf": 29, "Tomato Septoria leaf spot": 32, "Tomato leaf bacterial spot": 28,
    "Tomato leaf late blight": 30, "Tomato leaf mosaic virus": 36, "Tomato leaf yellow virus": 35,
    "Tomato leaf": 37, "Tomato mold leaf": 31, "Tomato two spotted spider mites leaf": 33, "grape leaf black rot": 11, "grape leaf": 14,
}
SIZE, CROP = 256, 224


def load_split(split: str, exclude_hashes: set | None = None):
    """Return [(PIL image 256x256, class_id, file_hash)] for a PlantDoc split; skips broken files
    and (for test) any photo whose bytes also appear in train, so the test is fair."""
    items, skipped = [], 0
    for folder in sorted((PLANTDOC / split).iterdir()):
        cid = PLANTDOC_TO_ID.get(folder.name)
        if cid is None or not folder.is_dir():
            continue
        for f in sorted(folder.iterdir()):
            try:
                data = f.read_bytes()
                h = hashlib.md5(data).hexdigest()
                if exclude_hashes and h in exclude_hashes:
                    skipped += 1
                    continue
                img = Image.open(f)
                img = ImageOps.exif_transpose(img).convert("RGB")
            except Exception:
                skipped += 1
                continue
            img = ImageOps.fit(img, (SIZE, SIZE), Image.BILINEAR)  # short side 256 + centre crop
            items.append((img, cid, h))
    return items, skipped


class Normalizer:
    def __init__(self, processor):
        self.mean = torch.tensor(processor.image_mean).view(3, 1, 1)
        self.std = torch.tensor(processor.image_std).view(3, 1, 1)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return (t - self.mean) / self.std


def augment(img: Image.Image) -> Image.Image:
    """Field-photo style augmentation: random crop/zoom, flip, rotation, light and colour changes."""
    s = random.randint(CROP, SIZE)
    x, y = random.randint(0, SIZE - s), random.randint(0, SIZE - s)
    img = img.crop((x, y, x + s, y + s)).resize((CROP, CROP), Image.BILINEAR)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.3:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = img.rotate(random.uniform(-20, 20), resample=Image.BILINEAR)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))
    return img


def centre(img: Image.Image) -> Image.Image:
    o = (SIZE - CROP) // 2
    return img.crop((o, o, o + CROP, o + CROP))


@torch.no_grad()
def evaluate(model, items, norm, crop_given: bool = False, batch: int = 64):
    """Accuracy on items. crop_given=True mimics the app's crop picker: only the true crop's
    classes are allowed (crop taken from the label), which is how the app now works."""
    from diseases import DISEASES  # noqa: E402  (needs kisanscan on sys.path)
    crops = [DISEASES[i]["crop"] for i in range(len(DISEASES))]
    model.eval()
    correct = crop_correct = 0
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        x = torch.stack([norm(centre(im)) for im, _, _ in chunk])
        logits = model(pixel_values=x).logits
        for row, (_, cid, _) in zip(logits, chunk):
            correct += int(row.argmax()) == cid
            mask = torch.tensor([c == crops[cid] for c in crops])
            crop_correct += int(row.masked_fill(~mask, -1e9).argmax()) == cid
    n = len(items)
    return correct / n, crop_correct / n
