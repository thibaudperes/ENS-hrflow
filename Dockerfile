FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENV MODEL_PATH=/models/hybrid.joblib
ENV DEMO_PATH=/models/demo.json
ENV REFERENCE_PATH=/models/reference.json

EXPOSE 8080
CMD ["sh", "-c", "uvicorn hrflow_recsys.serve:app --host 0.0.0.0 --port ${PORT:-8080}"]
