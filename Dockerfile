@'
# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# System deps for torch/onnxruntime wheels and pdf parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the embedding model at build time so cold starts don't
# pay the HuggingFace download cost on every restart.
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')"

# data/chroma_db and artifacts/ should be mounted as a persistent volume
# on your host (Render disk, Fly volume, Railway volume, etc.) - otherwise
# ingestion and trained models are lost on every redeploy/restart.
VOLUME ["/app/data/chroma_db", "/app/artifacts"]

EXPOSE 8000

CMD ["python", "src/api.py"]
'@ | Set-Content -Path "Dockerfile" -Encoding utf8 -NoNewline