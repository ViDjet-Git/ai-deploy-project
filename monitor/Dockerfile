FROM python:3.10-slim

# Встановлюємо системні залежності
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копіюємо файли залежностей і встановлюємо їх
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо код
COPY ./src /app/src

# Flask конфіг
ENV PYTHONPATH=/app/src
ENV FLASK_APP=/app/src/monitor.py

EXPOSE 8070

CMD ["python", "src/monitor.py"]
