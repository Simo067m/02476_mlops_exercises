# Base image
FROM python:3.11.6-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY requirements.txt .
COPY pyproject.toml .
COPY src/mlops_02476_exercises/ .

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["uvicorn", "src/mlops_02476_exercises/api:app", "--host", "0.0.0.0", "--port", "8000"]
