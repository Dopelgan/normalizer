#!/bin/bash
set -euo pipefail

echo "Развёртывание Parser..."

if [ ! -f .env ]; then
    echo "Файл .env не найден. Скопируйте .env.example в .env и задайте пароли."
    exit 1
fi

mkdir -p models data/shared tests/fixtures

docker compose up -d --build

echo
echo "Сервисы запущены. Статус: docker compose ps"
echo "Документация API: http://127.0.0.1:${API_PORT:-8000}/docs"
echo "Проверка:         curl http://127.0.0.1:${API_PORT:-8000}/health"
