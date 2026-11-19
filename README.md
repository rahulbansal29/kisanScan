# KisanScan – AI Crop Disease Detector

1M1B Green Internship (YIRP) project by Rahul Bansal · Domain: Agriculture

A farmer takes a photo of one leaf. KisanScan says which disease it is, how sure it is,
and what to do now, in Hindi or English, and can read the advice aloud.

## Features (prototype)

1. **AI disease detection** – a MobileNetV2 model trained on the PlantVillage dataset recognises
   38 classes (26 diseases + healthy) across 14 crops: tomato, potato, maize, grape, apple and more.
   Shows confidence and the next two possibilities, and asks for a retake when confidence is below 50%.
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

## Known limits / next steps

- PlantVillage photos are lab images on plain backgrounds; accuracy on real field photos is lower.
  Next: fine-tune on Indian field photos (e.g. PlantDoc + photos collected from local farmers).
- Add Indian staple crops not in the dataset: wheat, rice, cotton, mustard.
- Offline Android app (TensorFlow Lite) for areas with poor network.
- Treatment doses are typical label rates; to be reviewed by a KVK / agriculture expert.
