from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from core.filetypes import PROBE_ORDER


class Settings(BaseSettings):
    """Единая точка конфигурации сервиса. Значения берутся из .env / окружения."""

    # ------------------------------------------------------------------ Общие
    TZ: str = "UTC"
    LOG_LEVEL: str = "INFO"

    # ------------------------------------------------------------ База данных
    POSTGRES_USER: str = "knowledge_user"
    POSTGRES_PASSWORD: str = "secure_password"
    POSTGRES_DB: str = "knowledge"
    POSTGRES_PORT: int = 5432
    DATABASE_URL: str = "postgresql://knowledge_user:secure_password@postgres:5432/knowledge"

    # --------------------------------------------------------- Redis / Celery
    REDIS_PORT: int = 6379
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"
    EVENT_BROKER_URL: str = "redis://redis:6379/1"

    # ------------------------------------------------------------- Хранилище
    STORAGE_TYPE: str = "local"                 # local | s3
    STORAGE_LOCAL_MOUNT: str = "/data/shared"
    S3_ENDPOINT: Optional[str] = None
    S3_ACCESS_KEY: Optional[str] = None
    S3_SECRET_KEY: Optional[str] = None
    S3_BUCKET: Optional[str] = None
    RAW_PARSE_S3_PREFIX: str = "raw_parse/"

    # Где Parser ищет входные файлы по s3_fileid (контракт RAG <-> Parser).
    SOURCE_PREFIX: str = "documents/"
    # Расширения, которые перебираются, если s3_fileid пришёл без расширения.
    # Значение по умолчанию собирается из core.filetypes — единого реестра
    # поддерживаемых форматов, чтобы список не расходился с ним.
    SOURCE_PROBE_EXTENSIONS: List[str] = ["." + ext for ext in PROBE_ORDER]
    # Куда складываются изображения, вырезанные из PDF.
    ASSETS_PREFIX: str = "assets/"

    # ----------------------------------------------------------------- MinerU
    MINERU_VERSION: str = "3.4.5"
    MINERU_PORT: int = 8000
    MINERU_ENDPOINT: str = "http://mineru-api:8000"
    MINERU_TIMEOUT: int = 300
    MINERU_MODEL_CACHE: str = "/models/pdf-extract-kit"
    MINERU_BACKEND: str = "pipeline"
    MINERU_DEFAULT_LANG: str = "east_slavic"
    # Собирать ли для растра текстовый слой построчным распознаванием и
    # сшивать его с блоками MinerU. Стоит одного прохода Tesseract на
    # документ; без него русская проза в ячейках таблиц остаётся LaTeX-кашей.
    OCR_LAYER_ENABLED: bool = True

    # ------------------------- Чертежи: разметка зрением + внешняя модель
    # Адрес хоста с моделью (vLLM, OpenAI-совместимый). По умолчанию — сосед
    # по compose: контейнер vllm на внутренней сети. Пустое значение выключает
    # ветку чертежей целиком, и это видно в логе: подменять разбор сплошным
    # распознаванием листа больше нечем и незачем.
    QWEN_ENDPOINT: Optional[str] = "http://vllm:8001/v1"
    QWEN_MODEL: str = "Qwen/Qwen3.8-27B-FP8"
    # vLLM внутри контура поднимается с --api-key local: ключ не секрет, но
    # без него OpenAI-клиенты отказываются слать запрос.
    QWEN_API_KEY: Optional[str] = "local"
    QWEN_TIMEOUT: int = 180
    # Повторы только на сетевых сбоях и 5xx: «модель ответила ерунду» не
    # повторяется.
    QWEN_RETRIES: int = 2
    QWEN_MAX_TOKENS: int = 4096
    QWEN_TEMPERATURE: float = 0.0
    # Длинная сторона листа и вырезанной области при отправке модели.
    QWEN_SHEET_MAX_SIDE: int = 2000
    QWEN_REGION_MAX_SIDE: int = 1600
    # Во сколько раз увеличивается вырезанная область: мелкий текст штампа
    # читается заметно лучше на увеличенном фрагменте.
    QWEN_REGION_UPSCALE: float = 2.0

    # Разметка листа зрением: выравнивание, рамка формата, сетки таблиц.
    # Выключение оставляет только чтение листа целиком.
    DRAWING_VISION_ENABLED: bool = True
    # Предел выправления наклона в градусах: скан кладут криво на градус-два,
    # а поворот на десять означает, что лист опознан неверно.
    DRAWING_DESKEW_LIMIT: float = 3.0
    # Сколько областей за раз уходит в модель отдельными запросами.
    DRAWING_MAX_REGIONS: int = 4
    # Потолок времени на один лист. Запросов к модели на лист несколько
    # (сам лист плюс области), и при неудачном сочетании таймаутов они
    # переживают предел задачи Celery: тогда результат теряется целиком, а
    # не деградирует. Лучше отдать лист с прочитанными областями и
    # пометкой, что на остальные не хватило времени.
    DRAWING_TIME_BUDGET: int = 420

    # -------------------------------------------------- Приём документов
    # Профиль отсева: какие данные эта установка считает знанием.
    GATEWAY_PROFILE_PATH: str = "config/profile.yaml"
    # Классификатор приёма (слои G-2, G-3, G-4). Пусто — решают правила.
    CLASSIFIER_ENDPOINT: Optional[str] = None
    CLASSIFIER_TIMEOUT: int = 30
    GATEWAY_PORT: int = 8010
    QUALITY_GATE_PORT: int = 8011
    # Адреса соседей в цепочке приёма: gateway -> quality gate -> нормализатор.
    QUALITY_GATE_ENDPOINT: str = "http://quality_gate:8000"
    PARSER_ENDPOINT: str = "http://ingest_api:8000"
    INTAKE_TIMEOUT: int = 120
    # Сколько файлов принимается за один запрос. Без потолка список из
    # десятков тысяч идентификаторов ставит столько же задач Celery.
    INTAKE_MAX_FILES: int = 500

    # ------------------------------------------------- Проверка качества
    # Семантическое сходство выше порога — флаг «обновление или дубль».
    QG_SIMILARITY_THRESHOLD: float = 0.85
    # Документ старше этого срока сопровождается предупреждением.
    QG_MAX_AGE_DAYS: int = 1825
    # Уверенность быстрого OCR: ниже первого порога — предупреждение,
    # ниже второго — блокировка.
    QG_OCR_WARN_CONFIDENCE: float = 0.65
    QG_OCR_BLOCK_CONFIDENCE: float = 0.35
    # Сколько первых страниц осматривает быстрый OCR.
    QG_OCR_PAGES: int = 2
    # Потолок размера файла. Больше — в карантин, не читая целиком в память.
    QG_MAX_SIZE_BYTES: int = 200 * 1024 * 1024

    # ------------------------------------------------- Лестница стратегий
    # Оценка, начиная с которой результат уровня считается достаточным и
    # подъём прекращается. Ниже — маршрутизатор пробует следующий уровень.
    LADDER_SCORE_THRESHOLD: float = 0.75
    # Какую долю текста лучшего фоллбэка структурный разбор обязан сохранить,
    # чтобы получить старшинство над ним. Ниже этой доли считается, что
    # структурный уровень потерял содержимое, и он соревнуется по баллу.
    LADDER_CONTENT_KEEP_RATIO: float = 0.5
    # Какие уровни включены в этой установке. Профиль может отключить
    # дорогие уровни, не трогая код: 1 исходник САПР, 2 текстовый слой,
    # 3 табличный разбор, 4 распознавание без структуры, 5 детекция
    # областей, 6 восстановление растра плюс 5, 7 мультимодальная модель.
    LADDER_ENABLED_LEVELS: List[int] = [1, 2, 3, 4, 5, 6, 7]

    # --------------------------------- Мультимодальная модель (уровень 7)
    # Пусто — уровень 7 неприменим, и лестница заканчивается на предыдущем.
    VLM_ENDPOINT: Optional[str] = None
    VLM_TIMEOUT: int = 180
    VLM_PROMPT: str = "Извлеки весь текст документа, сохраняя структуру."

    # ---------------------------------------------------------- Бизнес-логика
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_OVERLAP: int = 50
    MIN_CHUNK_SIZE: int = 30
    COMPLETENESS_THRESHOLD: float = 0.5
    CONFIDENCE_THRESHOLD: float = 0.6
    PROCESSING_TIMEOUT_SECONDS: int = 900
    DEFAULT_LANGUAGE: str = "ru"

    # ----------------------------------------------------------------- Celery
    CELERY_TASK_TIME_LIMIT: int = 600
    CELERY_TASK_SOFT_TIME_LIMIT: int = 540
    CELERY_ML_QUEUE: str = "ml_gpu"
    CELERY_TASK_MAX_RETRIES: int = 3

    # ------------------------------------------------------ Результат запроса
    # Сколько держать готовый результат для GET /internal/v1/parse/results/{id}
    RESULT_TTL_SECONDS: int = 86400
    # Опциональный push-канал поверх поллинга: none | redis_streams | callback | nats
    RESULT_DELIVERY_TYPE: str = "none"
    RESULT_CALLBACK_URL: Optional[str] = None
    RESULT_STREAM_NAME: str = "normalization_results"
    NATS_SERVERS: Optional[str] = None
    NATS_SUBJECT: str = "normalization.result"

    # ------------------------------------------------------------------ Прочее
    HF_TOKEN: Optional[str] = None
    API_PORT: int = 8000
    # Потолок числа пикселей при открытии растра. Лист A0 при 600 dpi —
    # около 280 Мпикс; всё, что заметно больше, это попытка положить
    # воркер по памяти одним файлом, а не чертёж.
    MAX_IMAGE_PIXELS: int = 300_000_000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()


def _guard_pillow() -> None:
    """
    Потолок распаковки растра — один на процесс. Pillow по умолчанию лишь
    предупреждает, а предупреждение никто не читает: один подложенный PNG
    на сотни мегапикселей кладёт воркер по памяти.
    """
    try:
        import warnings

        from PIL import Image
    except ImportError:  # pragma: no cover — Pillow может быть не установлен
        return
    Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS
    warnings.simplefilter("error", Image.DecompressionBombWarning)


_guard_pillow()
