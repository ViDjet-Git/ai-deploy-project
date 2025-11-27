# Базовий образ
FROM python:3.10-slim

# Робоча директорія
WORKDIR /app

#Системні утиліти для curl та pillow codecs
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Копіюємо залежності
COPY requirements.txt .

# Встановлюємо залежності
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо увесь код
COPY . .

#Створюємо папку логів
RUN mkdir -p /app/logs

# Відкриваємо порт
EXPOSE 8080

#Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://127.0.0.1:8080/health || exit 1

# Використовуємо gunicorn як production server; модуль: src.api:app
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "src.api:app"]
