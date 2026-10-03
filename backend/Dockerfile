FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml ./
COPY aqua_drift ./aqua_drift
RUN pip install --no-cache-dir .

CMD ["uvicorn", "aqua_drift.api:app", "--host", "0.0.0.0", "--port", "8000"]
