FROM python:3.10-slim

LABEL org.opencontainers.image.title="Agente Cultural Foliar" \
      org.opencontainers.image.description="Backend ético de IA para o Museu Virtual, com RAG local e respeito às comunidades Karipuna, Palikur e quilombolas." \
      maintainer="AAFCP <afolclorica@gmail.com>"

ENV HOST=0.0.0.0 \
    PORT=8000 \
    MODEL_PROVIDER=ollama \
    OLLAMA_BASE_URL=http://localhost:11434 \
    OLLAMA_MODEL=phi3:latest \
    DATA_DIR=/app/data \
    STORAGE_DIR=/app/storage \
    TAINACAN_JSON_PATH=/app/data/metadados-tainacan.json

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R 1001:1001 /app && chmod -R 755 /app
USER 1001

EXPOSE $PORT

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]