"""Measure the base model on PlantDoc field photos, fine-tune it, and save the result if better.

Run from kisanscan/:  .venv/Scripts/python training/finetune.py
Writes models/kisanscan-plantdoc/ (model + results.json).
"""
import copy
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageClassification, MobileNetV2ImageProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.common import (BASE_MODEL, FINETUNED, ROOT, Normalizer, augment, evaluate,  # noqa: E402
                             load_split)

EPOCHS, BATCH, LR = 10, 32, 1e-4  # 5e-4 made the loss rise (too fast for a pretrained model)
# "Learning without forgetting": a small pull towards the base model's answers so the 11 classes
# PlantDoc lacks are not forgotten. Kept small: the base model is often wrong on field photos.
DISTILL_W, T = 0.2, 2.0
random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(max(1, torch.get_num_threads()))

t0 = time.time()
processor = MobileNetV2ImageProcessor.from_pretrained(BASE_MODEL)
norm = Normalizer(processor)
base = AutoModelForImageClassification.from_pretrained(BASE_MODEL).eval()

train_all, bad_tr = load_split("train")
test, dup = load_split("test", exclude_hashes={h for _, _, h in train_all})
random.shuffle(train_all)
n_val = int(len(train_all) * 0.15)
val, train = train_all[:n_val], train_all[n_val:]
print(f"train {len(train)}  val {len(val)}  test {len(test)}  "
      f"(skipped {bad_tr} broken train files, {dup} test photos also in train)", flush=True)

base_test = evaluate(base, test, norm)
print(f"BASE model on field photos: {base_test[0]:.1%} (with crop chosen: {base_test[1]:.1%})", flush=True)

model = copy.deepcopy(base)
# Freeze the first half of the network (general edge/texture features); train the rest.
blocks = list(model.mobilenet_v2.layer)
for p in list(model.mobilenet_v2.conv_stem.parameters()) + [p for b in blocks[:8] for p in b.parameters()]:
    p.requires_grad = False
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
steps = EPOCHS * ((len(train) + BATCH - 1) // BATCH)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.15)


def set_train_mode(m):
    m.train()
    for mod in m.modules():  # keep BatchNorm statistics from the big PlantVillage training
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.eval()


best_val, best_state, history = evaluate(model, val, norm)[0], copy.deepcopy(model.state_dict()), []
print(f"epoch 0  val {best_val:.1%}", flush=True)
for ep in range(1, EPOCHS + 1):
    set_train_mode(model)
    random.shuffle(train)
    total = hits = 0.0
    for i in range(0, len(train), BATCH):
        chunk = train[i:i + BATCH]
        x = torch.stack([norm(augment(im)) for im, _, _ in chunk])
        y = torch.tensor([c for _, c, _ in chunk])
        logits = model(pixel_values=x).logits
        with torch.no_grad():
            teacher = base(pixel_values=x).logits
        loss = F.cross_entropy(logits, y, label_smoothing=0.1) + DISTILL_W * F.kl_div(
            F.log_softmax(logits / T, -1), F.softmax(teacher / T, -1), reduction="batchmean") * T * T
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        total += loss.item() * len(chunk)
        hits += (logits.argmax(-1) == y).sum().item()
    v = evaluate(model, val, norm)[0]
    history.append({"epoch": ep, "loss": round(total / len(train), 4), "train": round(hits / len(train), 4),
                    "val": round(v, 4)})
    print(f"epoch {ep}  loss {total / len(train):.3f}  train {hits / len(train):.1%}  val {v:.1%}  "
          f"({time.time() - t0:.0f}s)", flush=True)
    if v > best_val:
        best_val, best_state = v, copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
new_test = evaluate(model, test, norm)
print(f"NEW model on field photos: {new_test[0]:.1%} (with crop chosen: {new_test[1]:.1%})", flush=True)

# Regression check on the PlantVillage sample photos used in the demo.
from PIL import Image  # noqa: E402
from training.common import centre, SIZE  # noqa: E402
from PIL import ImageOps  # noqa: E402
from diseases import DISEASES  # noqa: E402
samples = {}
for p in sorted((ROOT / "sample_leaves").glob("*.jpg")):
    img = centre(ImageOps.fit(Image.open(p).convert("RGB"), (SIZE, SIZE), Image.BILINEAR))
    with torch.no_grad():
        pr = model(pixel_values=norm(img)[None]).logits.softmax(-1)[0]
    samples[p.name] = f"{DISEASES[int(pr.argmax())]['name_en']} {pr.max():.0%}"
    print("  sample", p.name, "->", samples[p.name], flush=True)

FINETUNED.mkdir(parents=True, exist_ok=True)
model.save_pretrained(FINETUNED)
processor.save_pretrained(FINETUNED)
json.dump({
    "dataset": "PlantDoc (real field photos), github.com/pratikkayal/PlantDoc-Dataset, CC-BY-4.0",
    "train": len(train), "val": len(val), "test": len(test), "test_dupes_removed": dup,
    "base_test_acc": round(base_test[0], 4), "base_test_acc_crop_chosen": round(base_test[1], 4),
    "new_test_acc": round(new_test[0], 4), "new_test_acc_crop_chosen": round(new_test[1], 4),
    "best_val": round(best_val, 4), "history": history, "samples": samples,
    "minutes": round((time.time() - t0) / 60, 1),
}, open(FINETUNED / "results.json", "w"), indent=2)
print(f"Saved to {FINETUNED}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
