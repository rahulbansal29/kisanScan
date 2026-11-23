---
title: KisanScan
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: AI crop doctor for farmers, Hindi and English
---

# KisanScan – AI Crop Disease Detector

1M1B Green Internship (YIRP) project by Rahul Bansal · Domain: Agriculture

A farmer takes a photo of one leaf. KisanScan says which disease it is, how sure it is,
and what to do now, in Hindi or English, and can read the advice aloud.

## Features (prototype)

1. **AI disease detection** – recognises 38 classes (26 diseases + healthy) across 14 crops:
   tomato, potato, maize, grape, apple and more. The farmer first picks the crop, so the AI only
   chooses among that crop's diseases; photos that don't match the chosen crop, non-leaf photos and
   unsupported crops get a clear message instead of a wrong answer. Shows confidence and the next
   two possibilities, and asks for a retake when confidence is below 50%.
2. **Treatment advice in Hindi & English with voice** – cause, what to do today (with typical
   Indian dose per litre), and prevention, for every class. One tap switches language;
   the 🔊 button reads it aloud for farmers who prefer listening.
3. **Scan history & disease alerts** – every scan is saved; the dashboard shows total scans,
   diseased count and the most common disease, a base for village-level outbreak alerts.

## Run it

```bash
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python app.py
```

Open http://localhost:5000 and try a photo from `sample_leaves/`.

## Tech

Python · Flask · PyTorch · Hugging Face Transformers
(`linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`) · SQLite · HTML/JS (Web Speech API)

## Accuracy on real field photos

The base model was trained on PlantVillage lab photos (plain background). I measured it on
**225 real field photos** from the [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset)
test set (CC-BY-4.0; 11 photos that also appear in its train set removed), then fine-tuned a copy on
PlantDoc's 1,986 training photos (`training/finetune.py`, ~50 min on a laptop CPU). The app averages
the base and fine-tuned models.

| Model | Field photos | Field photos, crop chosen (how the app works) | 7 demo lab photos |
|---|---|---|---|
| Base (PlantVillage only) | 31.1% | 59.6% | 7/7 |
| Fine-tuned on PlantDoc | 59.1% | 77.3% | 4/7 |
| **Base + fine-tuned (used in app)** | 45.8% | **71.6%** | **7/7** |

225 test photos is a small set, so treat these as ±6% estimates. To reproduce: download PlantDoc
into `../datasets/plantdoc` (`training/extract_plantdoc.py`), then run `training/finetune.py` and
`training/compare.py`.

## Known limits / next steps

- About 7 in 10 field photos are right with the crop chosen: good for a prototype, not yet for
  advice without a human check. Next: collect and label photos from local farmers to fine-tune further.
- Add Indian staple crops not in the dataset: wheat, rice, cotton, mustard.
- Offline Android app (TensorFlow Lite) for areas with poor network.
- Treatment doses are typical label rates; to be reviewed by a KVK / agriculture expert.
