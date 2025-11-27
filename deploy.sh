#!/bin/bash
set -e

echo "=== [CI/CD] Starting deployment ==="

# Визначаємо активне оточення
if docker ps | grep -q "ai_api_blue"; then
    ACTIVE="blue"
    NEXT="green"
elif docker ps | grep -q "ai_api_green"; then
    ACTIVE="green"
    NEXT="blue"
else
    ACTIVE="none"
    NEXT="blue"
fi

echo "Active environment: $ACTIVE"
echo "Next environment:   $NEXT"

echo "=== Building next version ($NEXT) ==="
docker compose -f docker-compose.$NEXT.yml build

if [ "$ACTIVE" != "none" ]; then
  echo "Stopping OLD ($ACTIVE) stack..."
  docker compose -f docker-compose.$ACTIVE.yml down
fi

echo "=== Starting $NEXT stack ==="
docker compose -f docker-compose.$NEXT.yml up -d

echo "Waiting 10 seconds for services to warm up..."
sleep 10

echo "Checking health of new version..."
HEALTH=$(curl -s http://localhost:8080/health || echo "fail")

if [[ "$HEALTH" == "fail" ]]; then
    echo "New version is NOT healthy. Deployment aborted."
    exit 1
fi

echo "New version is healthy!"
echo "=== Deployment finished successfully! ==="
