FROM python:3.10-slim

WORKDIR /app

# Встановлюємо залежності
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо код
COPY ./src /app/src

ENV PYTHONPATH=/app/src

EXPOSE 8050

CMD ["python", "src/app.py"]
