# KisanScan on Hugging Face Spaces (Docker SDK, free CPU). Also runs locally:
#   docker build -t kisanscan . && docker run -p 7860:7860 kisanscan
FROM python:3.13-slim

# Hugging Face runs the container as user 1000; give it a home and a writable app folder.
RUN useradd -m -u 1000 user
ENV HOME=/home/user HF_HOME=/home/user/.cache/huggingface PORT=7860 PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY requirements.txt .
# CPU-only PyTorch (much smaller than the default CUDA build)
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user
# Download the base model at build time so the first visitor doesn't wait for it.
RUN python -c "from transformers import AutoModelForImageClassification as M, MobileNetV2ImageProcessor as P; \
n='linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification'; M.from_pretrained(n); P.from_pretrained(n)"

EXPOSE 7860
CMD ["python", "app.py"]
