FROM python:3.10-slim

WORKDIR /trainer

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl libsndfile1 libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

# create models mount point
RUN mkdir -p /models /trainer_logs

EXPOSE 8090

CMD ["python", "src/api.py"]
