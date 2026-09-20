#!/bin/bash
set -euo pipefail

apt-get update && apt-get install -y --no-install-recommends ca-certificates
pip install --no-cache-dir "huggingface-hub[hf_xet]"

export HF_XET_HIGH_PERFORMANCE=1
# Официальный хост по умолчанию: зеркало подставляется только явно, своим
# решением. Веса — исполняемый код (.pt распаковывается pickle'ом), и
# источник у них должен быть выбран, а не унаследован из скрипта.
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

if [ -n "${HF_TOKEN:-}" ]; then
    echo "🔑 Авторизация Hugging Face включена."
else
    echo "⚠️ HF_TOKEN не найден, скачивание без авторизации."
fi

: "${MINERU_MODEL_CACHE:?Переменная MINERU_MODEL_CACHE не задана}"

echo "🚀 Запуск загрузки моделей..."
echo "   MINERU_MODEL_CACHE=$MINERU_MODEL_CACHE"

# ----------------------------------------------------------------------
# PDF-Extract-Kit (полный)
# ----------------------------------------------------------------------
PDF_DIR="$MINERU_MODEL_CACHE"
mkdir -p "$PDF_DIR"
# Признак готовности — макет PP-DocLayoutV2: именно его грузит pipeline
# MinerU 3.4.x. Прежняя проверка смотрела на models/Layout/YOLO — путь из
# MinerU 2.x, который к работающей ветке разбора отношения не имеет.
if [ -d "$PDF_DIR/models/Layout/PP-DocLayoutV2" ] && [ -f "$PDF_DIR/mineru.json" ]; then
    echo "✅ PDF-Extract-Kit уже скачан, пропуск."
else
    echo "📥 Скачивание PDF-Extract-Kit (~15 ГБ)..."
    # Путь передаётся через окружение, а не подстановкой в текст программы:
    # кавычка в значении переменной иначе выполняет произвольный Python.
    TARGET_DIR="$PDF_DIR" python3 -c '
import os
from huggingface_hub import snapshot_download
snapshot_download("opendatalab/PDF-Extract-Kit-1.0", local_dir=os.environ["TARGET_DIR"])
'
fi

# ----------------------------------------------------------------------
# Конфиг MinerU: при MINERU_MODEL_SOURCE=local сервис берёт корень моделей
# только отсюда. Без этого файла get_local_models_dir() возвращает None и
# первый же разбор падает с AttributeError, ещё до чтения документа.
# Источник моделей задаётся окружением, в JSON нужен только путь.
# ----------------------------------------------------------------------
TARGET_DIR="$PDF_DIR" python3 -c '
import json, os
root = os.environ["TARGET_DIR"]
path = os.path.join(root, "mineru.json")
config = {}
if os.path.exists(path):
    try:
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            config = loaded
    except ValueError:
        pass
models_dir = config.get("models-dir")
if not isinstance(models_dir, dict):
    models_dir = {}
models_dir["pipeline"] = root
config["models-dir"] = models_dir
with open(path, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
    f.write("\n")
'
echo "🧾 Конфиг моделей записан: $PDF_DIR/mineru.json"

echo "🎉 Модели скачаны."
echo
echo "ℹ️  Модель чертежей здесь НЕ скачивается: она живёт на отдельном хосте с"
echo "   GPU и поднимается там через vLLM, например:"
echo "     vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 --limit-mm-per-prompt image=2"
echo "   Проверка из воркера: curl \$QWEN_ENDPOINT/v1/models"
