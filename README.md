# AI Deploy Project

Проєкт для автоматизації розгортання ШІ-додатків у хмарі.

## Структура
- `app/src/` — вихідний код Flask API та логіка
- `app/model/` — файли нейронних мереж
- `app/requirements.txt` — Python-залежності
- `app/Dockerfile` — опис контейнера
- `docker-compose.yml` — конфігурація для запуску

## Запуск
```bash
docker compose up --build
