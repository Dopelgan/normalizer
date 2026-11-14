# Карта кода Parser/Normalizer

Навигация по коду: какой **файл** и какая **функция** владеют каждым шагом и каждым
решением. Выведена из кода (базовый коммит `fd0c0e5`), не из документации.
Номера строк проверены на этом коммите; ветка чертежей за 65 коммитов от `5040065`
уехала на 60–200 строк, поэтому номер строки сверять, а не верить.

## Как пользоваться картой

1. Прочитать нужный раздел этой карты.
2. Открыть **два-три файла**, которые она назвала, и остановиться. Читать дерево целиком
   не нужно: если стадия имеет одного владельца, он здесь назван; если владельца нет — это
   тоже сказано прямо.
3. Подробности настроек и их **значения на проде** — `README.md` §«Флаги разбора»
   (там же, чего флаг стоит). Контракт с RAG и его изменения — `docs/PARSER-CONTRACT-CHANGES.md`.
   Карта их не дублирует: она даёт то, чего в них нет, — файл и функцию на стадию и
   указатель «кто читает настройку».
4. **Карта ошиблась — починить строку в той же правке.** Код прав, карта догоняет.

## 1. Конвейер по стадиям

### 1.1. Приём (отсев до разбора) — отдельный сервис

| Стадия | Файл | Вход |
|---|---|---|
| HTTP-приём пакета файлов | `gateway/api.py` | `intake` (строка 98, `POST /internal/v1/intake`), `intake_sync` (154), `intake_results` (139) |
| Карантин и отчёт приёма | `gateway/api.py` | `quarantine` (202), `resolve` (220), `report` (241), `suggestions` (251) |
| Постановка файла в Celery | `core/intake/tasks.py` | `dispatch` (89) → задача `process_intake_file` (39) |
| Оба слоя приёма в одном процессе | `core/intake/pipeline.py` | `IntakePipeline` (43), `evaluate_file` (271) |
| Состояние пакета приёма (Redis) | `core/intake/state.py` | — |
| Слои G-1…G-4, три исхода | `core/gateway/service.py` | `DataGateway` (112), `GatewayVerdict` (87) |
| Правила отсева одной установки | `core/gateway/profile.py`, `core/gateway/rules.py` | профиль `config/profile.yaml` (`GATEWAY_PROFILE_PATH`) |
| Текст документа инструментом его формата | `core/gateway/text_probe.py` | — |
| Гейт качества (отдельный сервис) | `quality/api.py` | `check` (79, `POST /internal/v1/quality`) → `core/quality/service.py` |

### 1.2. Запрос на разбор и поиск файла

| Стадия | Файл | Вход |
|---|---|---|
| Синхронный разбор | `ingest/api.py` | `parse_sync` (186, `POST /internal/v1/parse`) |
| Фоновый разбор | `ingest/api.py` | `parse_background` (239, `POST /internal/v1/parse/background`) |
| Выдача результата | `ingest/api.py` | `parse_results` (275, `GET /internal/v1/parse/results/{request_id}`) |
| Выбор очереди (ml_gpu / интерактивная) | `ingest/api.py` | `parse_queue` (72), `_dispatch` (86) |
| **Задача разбора — главный оркестратор** | `ingest/tasks.py` | `process_document` (358) |
| Поиск файла по `s3_fileid` | `core/providers/file_locator.py` | `FileLocator.locate` (64); перебор расширений `_candidates` (120), вложения чата `_locate_attachment` (127) |
| Хранилище (S3/локальное, запись только в свой том) | `core/providers/storage.py` | — |
| Срок задачи и его резерв | `core/deadline.py` | `scope` (26), `remaining` (35), `bounded` (41) |

### 1.3. Лестница уровней — где решается, какой уровень выбран

**Владелец решения — один файл:** `core/ladder/router.py`.

| Что | Файл | Функция |
|---|---|---|
| Набор уровней, порядок, отключение уровня | `core/ladder/router.py` | `default_strategies` (95); фильтр по `LADDER_ENABLED_LEVELS` |
| **Перебор уровней и выбор победителя** | `core/ladder/router.py` | `LadderRouter._parse` (175); `_applicable` (396), `_keeps_content` (389) |
| Оценка результата уровня (порог) | `core/ladder/scoring.py` | `score_result` (66); `_completeness` (136), `_structure_ok` (155) |
| Контекст документа и классификация входа | `core/ladder/context.py` | `DocumentContext` (73), `classify` (231), `_classify_pdf` (290), `assess_raster` (432) |
| Выправка растра перед распознаванием | `core/ladder/restoration.py` | `restore` (33), `restore_pages` (44) |
| Интерфейс уровня | `core/ladder/base.py` | `Strategy` (15), `text_chars` (68), `has_structure` (80) |

Уровни (значение `level` — в коде класса):

| Уровень | Класс | Файл |
|---|---|---|
| 1 | `CadSourceStrategy` | `core/ladder/strategies/cad_source.py:58` |
| 2 | `PdfTextLayerStrategy` (текстовый слой PDF) | `core/ladder/strategies/native_text.py:40` |
| 2 | `DocxStrategy` | `core/ladder/strategies/native_text.py:81` |
| 2 | `PlainTextStrategy` | `core/ladder/strategies/native_text.py:162` |
| 3 | `SpreadsheetStrategy` | `core/ladder/strategies/tabular.py:53` |
| 4 | `PlainOcrStrategy` (Tesseract целиком) | `core/ladder/strategies/recognition.py:35` |
| 5 | `LayoutStrategy` (MinerU) | `core/ladder/strategies/recognition.py:73` |
| 6 | `RestoreThenLayoutStrategy` (выправка → MinerU) | `core/ladder/strategies/recognition.py:121` |
| 7 | `VlmEscalationStrategy` | `core/ladder/strategies/recognition.py:253` |

### 1.4. Текстовый слой, OCR, MinerU

| Стадия | Файл | Вход |
|---|---|---|
| Текстовый слой PDF | `core/providers/text_layer.py` | — |
| OCR Tesseract (полный проход) | `core/providers/tesseract_fallback.py` | многопоточность страниц — строки 262–285 (см. ловушку про OMP) |
| Подготовка растра к OCR | `core/providers/ocr_prepare.py` | выправка наклона, снятие линий таблиц, удаление шума |
| Чистка слов распознавания | `core/providers/ocr_cleanup.py` | — |
| MinerU | `core/providers/mineru_parser.py` | сшивка слоя распознавания, защита от «таблица-рамка», таймаут на страницу |
| Таблицы из линий / повторный проход VLM | `core/providers/table_from_lines.py`, `core/providers/table_vlm.py` | — |
| Конвертация office → PDF | `core/providers/office_convert.py` | — |
| Проза страницы (чертёж или текст в рамке) | `core/providers/page_prose.py` | — |
| Фабрика парсеров | `core/providers/document_parser_factory.py`, `core/providers/document_parser.py` | — |

### 1.5. Ветка чертежей и проходы модели

| Стадия | Файл | Функция |
|---|---|---|
| **Вся ветка чертежей** | `core/drawing_processor.py` | `DrawingProcessor.process` (131), `enrich_fragments` (489) — вызывается из `ingest/tasks.py:598` |
| Разметка листа геометрией | `core/drawing_processor.py` | `_sheet_drawings` (645) |
| Проход «перепиши лист целиком» | `core/drawing_processor.py` | `_transcribe` (768) → `core/providers/qwen_client.py` |
| Проход по штампу | `core/drawing_processor.py` | `_read_title_block` (803), `_read_title_block_fields` (846); канонические метки — `core/providers/title_block.py` `LABELS` |
| Проход по боковым графам | `core/drawing_processor.py` | `_read_side_column` (911) |
| Проход по таблицам листа | `core/drawing_processor.py` | `_read_table` (928) |
| Поля из аннотаций векторного PDF | `core/drawing_processor.py` | `_fields_from_annotations` (948); категории — `core/providers/vector_drawing.classify` |
| Клиент Qwen-VL, разрезание листа на плитки | `core/providers/qwen_client.py` | `transcribe` / `transcribe_sheet` (397) |
||||||| fd0c0e5
| **Вся ветка чертежей** | `core/drawing_processor.py` | `DrawingProcessor.process` (130), `enrich_fragments` (358) — вызывается из `ingest/tasks.py:513` |
| Разметка листа геометрией | `core/drawing_processor.py` | `_sheet_drawings` (434) |
| Проход «перепиши лист целиком» | `core/drawing_processor.py` | `_transcribe` (557) → `core/providers/qwen_client.py` |
| Проход по штампу | `core/drawing_processor.py` | `_read_title_block` (592), `_read_title_block_fields` (635); канонические метки — `core/providers/title_block.py` `LABELS` |
| Проход по боковым графам | `core/drawing_processor.py` | `_read_side_column` (700) |
| Проход по таблицам листа | `core/drawing_processor.py` | `_read_table` (717) |
| Поля из аннотаций векторного PDF | `core/drawing_processor.py` | `_fields_from_annotations` (737); категории — `core/providers/vector_drawing.classify` |
| Клиент Qwen-VL, разрезание листа на плитки | `core/providers/qwen_client.py` | `transcribe` / `transcribe_sheet` (397) |
| **Вся ветка чертежей** | `core/drawing_processor.py` | `DrawingProcessor.process` (131), `enrich_fragments` (489) — вызывается из `ingest/tasks.py:598` |
| Разметка листа геометрией | `core/drawing_processor.py` | `_sheet_drawings` (645) |
| Проход «перепиши лист целиком» | `core/drawing_processor.py` | `_transcribe` (768) → `core/providers/qwen_client.py` |
| Проход по штампу | `core/drawing_processor.py` | `_read_title_block` (803), `_read_title_block_fields` (846); канонические метки — `core/providers/title_block.py` `LABELS` |
| Проход по боковым графам | `core/drawing_processor.py` | `_read_side_column` (911) |
| Проход по таблицам листа | `core/drawing_processor.py` | `_read_table` (928) |
| Поля из аннотаций векторного PDF | `core/drawing_processor.py` | `_fields_from_annotations` (948); категории — `core/providers/vector_drawing.classify` |
| Клиент Qwen-VL, разрезание листа на плитки | `core/providers/qwen_client.py` | `transcribe` (621) / `transcribe_sheet` (635); единственная точка вызова модели — `_ask_text` (781) |
| Геометрия и повороты листа | `core/providers/drawing_vision.py`, `core/providers/raster_drawing.py` | — |
| Выправка размеров построчно | `core/providers/dimension_fidelity.py` | `repair_lines` |
| Нормализация строки чертежа (⌀, ±, гомоглифы) | `core/providers/drawing_text.py` | `normalize` |
| **Сборка `content.text` фрагмента чертежа** | `core/drawing_processor.py` | `build_fragment` (432), строка 460: `text="\n".join(transcript) or None`; `transcript` берётся из `payload.pop("transcript_lines")` (443) |
| Качество фрагмента чертежа | `core/drawing_quality.py` | `apply_drawing_quality` (7), `document_quality` (71) |
| Правка абзацев OCR по строкам листа | `core/drawing_transcript.py` | `_page_transcripts` (88) |
| **Пересборка текста записи по частям** (`DRAWING_RECORD_COMPOSED`) | `core/drawing_record.py` | `apply` (236), `compose_fragment` (292), разметка частей для чанкера — `_part_index` (363); часть «остальные надписи» — `_other_marks`; вид при размере — `_views_of` |
| Разметка частей обычного документа | `core/document_parts.py` | — |
| Учёт разбора в сторону от индекса (`DRAWING_DIAGNOSTICS_NOT_INDEXED`) | `core/contract_finalize.py` | `_diagnostics_aside` |
| Снятие устаревших источников | `core/drawing_stale_sources.py` | — |
| **Решение «чертёж против OCR»** | `core/ladder/drawing_route.py` | `settle` (173) — вызывается из `ingest/tasks.py:601`; `deferred` (84), `volume` (143), `lost` (320), `thinner_kept` (325) |

**Сборщика и переупорядочивания строк листа нет.** Текст записи чертежа — строки
модели в порядке выдачи, правленные построчно (`dimension_fidelity`, `drawing_text`).
Структурные данные (`structured_drawing_fields`, `structured_payload["tables"]`) в текст
сознательно **не** попадают: в `payload` остаётся только `transcript_line_count`
(`drawing_processor.py:445`), чтобы потребитель не индексировал текст дважды.

### 1.6. Нормализация во фрагменты и контракт наружу

| Стадия | Файл | Функция |
|---|---|---|
| Блоки → фрагменты контракта | `core/normalizer/normalizer.py` | `TextNormalizer.normalize` (73), `normalize_with_flags` (80) |
| **Рамка бланка и переносы обычного документа** | `core/document_furniture.py` | `apply`, `is_furniture`, `without_splices`, `without_hyphen_breaks` — вызывается из `core/contract_finalize.finalize_fragments` |
| Ярлык/заголовок и его текст одним фрагментом | `core/document_parts.py` | `apply`, `is_label_line`, `is_heading_line` |
| Полнота текста/таблиц/формул | `core/normalizer/completeness.py` | `PageCoverage` (17), `TextLayerRecall` (59), `table_completeness` (103), `formula_completeness` (117) |
| Чанкование | `core/strategies/factory.py` + `fixed_size.py`, `paragraph.py`, `semantic.py` | — |
| Извлечение фактов | `core/facts/extractor.py` | `extract_facts` (426); `facts_from_text` (281), `facts_from_table` (328), `facts_from_drawing_fields` (375) |
| **Стабильные `fragment_id`, финал контракта** | `core/contract_finalize.py` | `finalize_fragments` (126); подпись — `_content_signature` (36), id — `stable_fragment_id` (61), хеш — `normalized_content_hash` (281) |
| Сборка `DocumentResult` | `ingest/tasks.py` | `build_document_result` (253), `finalize` (288), `page_report` (218), `build_warnings` (245) |
| Классификация ошибки, ретраи | `ingest/tasks.py` | `classify_error` (158), `build_error_document` (190) |
| Состояние пакетного запроса (Redis) | `core/result_aggregator.py` | `init_request_state` (70), `put_document_result` (152), `compute_status` (304), `_expire_lost_files` (234) |
| Публикация результата | `core/result_publisher.py` | `publish_result` (12) |
| Push наружу (streams/callback/NATS) | `core/result_delivery.py` | `get_delivery_provider` (150) |
| Фрагмент ↔ строка БД | `core/mappers.py`, `core/db/models.py`, `core/repositories/*` | — |

## 2. Где живут решения

Все настройки — один класс `Settings` в `core/config.py` (pydantic-settings, значения из
`.env`/окружения; `getenv` в коде нет). **Значения на проде для флагов разбора —
`README.md` §«Флаги разбора»** (колонка «На проде»); `docker-compose.yml` в репозитории
флаги не переопределяет, прод-`.env` в репозитории нет.

### 2.1. Флаги поведения разбора (bool) — файл-читатель

| Флаг | По умолчанию | Читает |
|---|---|---|
| `LADDER_DRAWING_OVER_OCR` | `false` | `core/ladder/drawing_route.py`, `core/ladder/router.py`, `core/models/parse_result.py`, `ingest/tasks.py` |
| `DRAWING_VISION_ENABLED` | `true` | `core/drawing_processor.py` |
| `DRAWING_TEXT_IN_FRAME_ENABLED` | `false` | `core/drawing_processor.py`, `core/providers/vector_drawing.py` |
| `DRAWING_PAGE_CLASSIFIER_ENABLED` | `false` | `core/drawing_processor.py`, `core/providers/vector_drawing.py` |
| `DRAWING_TITLE_BLOCK_PASS` | `false` | `core/drawing_processor.py`, `core/providers/qwen_client.py` |
| `DRAWING_SIDE_COLUMN_READ` | `true` | `core/drawing_processor.py` |
| `VLM_RESPONSE_CACHE` | `false` | `core/providers/qwen_client.py` (`cache_key`/`cache_get`/`cache_put`, вызов из `_ask_text` 781) |
| `DRAWING_IMAGE_FULL_READ` | `false` | `core/drawing_processor.py`, `core/drawing_quality.py`, `core/ladder/drawing_route.py` |
| `DRAWING_PDF_FULL_READ` | `false` | `core/drawing_processor.py`, `core/drawing_quality.py` |
| `DRAWING_TRANSCRIBE_TILES` | `false` | `core/providers/qwen_client.py`, `core/drawing_processor.py`, `core/drawing_quality.py` |
| `DRAWING_TRANSCRIPT_FIXES_OCR` | `false` | `core/drawing_transcript.py`, `core/drawing_processor.py`, `core/drawing_stale_sources.py` |
| `DRAWING_SUPPRESS_STALE_ON_FULL_READ` | `false` | `core/drawing_processor.py`, `core/drawing_stale_sources.py` |
| `DRAWING_UNMEASURED_NOT_PARTIAL` | `false` | `core/drawing_quality.py` |
| `DRAWING_DIMENSION_FIDELITY` | `false` | `core/providers/dimension_fidelity.py`, `core/providers/drawing_text.py`, `core/drawing_processor.py`, `core/providers/qwen_client.py`, `core/providers/mineru_parser.py`, `core/providers/vector_drawing.py` |
| `DRAWING_MATERIAL_NO_TOLERANCE` | `false` | `core/drawing_processor.py` |
| `DRAWING_SHEET_MISSING_PAGES` | `false` | `core/drawing_processor.py` |
| `DRAWING_RECORD_COMPOSED` | `false` | `core/drawing_record.py`, `core/contract_finalize.py`, `core/document_parts.py` |
| `DRAWING_VALUE_VIEW_LINKS` | `false` | `core/providers/drawing_views.py`, `core/drawing_processor.py` |
| `DRAWING_DIAGNOSTICS_NOT_INDEXED` | `false` | `core/contract_finalize.py` |
| `DRAWING_RECORD_WITHOUT_SHEET_TEXT` | `false` | `core/drawing_record.py` |
| `DRAWING_RECORD_USES_VALUE_LINKS` | `false` | `core/drawing_record.py` |
| `DRAWING_RECORD_STAMP_FROM_FIELDS` | `false` | `core/drawing_record.py` |
| `DRAWING_RECORD_ALL_FIELDS_SAID` | `false` | `core/drawing_record.py` |
| `SCAN_DROP_FRAME_FURNITURE` | `false` | `core/document_furniture.py`, `core/contract_finalize.py` |
| `SCAN_JOIN_HYPHENATED_WORDS` | `false` | `core/document_furniture.py`, `core/contract_finalize.py` |
| `DOCUMENT_HEADING_WITH_BODY` | `false` | `core/document_parts.py` |
| `DRAWING_FIELD_CATEGORY_STRICT` | `false` | `core/providers/vector_drawing.py` |
| `DRAWING_CATEGORY_PROSE_GUARD` | `false` | `core/providers/vector_drawing.py` |
| `DRAWING_SMALL_PRINT_CARE` | `false` | `core/providers/qwen_client.py` |
| `DRAWING_AUTO_ROTATE` | `true` | `core/providers/drawing_vision.py` |
| `RASTER_FRAME_THIN_LINES` | `false` | `core/providers/raster_drawing.py` |
| `FRAME_TABLE_GUARD_ENABLED` | `false` | `core/providers/mineru_parser.py` |
| `OCR_LAYER_ENABLED` | `true` | `core/providers/mineru_parser.py`, `core/providers/tesseract_fallback.py` |
| `OCR_LAYER_DROP_LOW_CONFIDENCE` | `false` | `core/providers/mineru_parser.py`, `core/providers/tesseract_fallback.py` |
| `OCR_LAYER_DROP_REPEATED` | `false` | `core/providers/mineru_parser.py` |
| `OCR_WORD_FIX_ENABLED` | `false` | `core/providers/ocr_cleanup.py`, `core/providers/mineru_parser.py`, `core/providers/tesseract_fallback.py` |
| `OCR_KEEP_STRUCTURED_TABLES` | `false` | `core/providers/mineru_parser.py` |
| `OCR_SPLIT_WIDE_GAPS` | `false` | `core/providers/mineru_parser.py`, `core/providers/tesseract_fallback.py` |
| `OCR_PREPROCESS_ENABLED` | `true` | `core/providers/tesseract_fallback.py` |
| `OCR_REMOVE_TABLE_LINES` | `true` | `core/providers/ocr_prepare.py` |
| `TABLE_VLM_RETRY_ENABLED` | `false` | `core/providers/table_vlm.py` |
| `DOC_CONVERT_ENABLED` | `true` | `core/providers/office_convert.py` |
| `INTAKE_FORWARD_TO_PARSER` | `false` | `core/intake/pipeline.py`, `gateway/api.py`, `quality/api.py` |
| `S3_READONLY` | `true` | `core/providers/storage.py` |

### 2.2. Пороги и лимиты — файл-читатель

| Настройка | По умолчанию | Читает | Что решает |
|---|---|---|---|
| `LADDER_SCORE_THRESHOLD` | `0.75` | `core/ladder/router.py` | достаточна ли оценка уровня, чтобы не идти дальше |
| `LADDER_CONTENT_KEEP_RATIO` | `0.5` | `core/ladder/router.py`, `core/ladder/drawing_route.py` | сколько текста дорогой уровень вправе потерять |
| `LADDER_ENABLED_LEVELS` | `[1..7]` | `core/ladder/router.py` | какие уровни вообще создаются |
| `LADDER_MIN_REMAINING` | `60` | `core/ladder/router.py` | минимум секунд, чтобы начать следующий уровень |
| `COMPLETENESS_THRESHOLD` | `0.5` | `core/normalizer/normalizer.py` | порог полноты фрагмента |
| `DRAWING_MIN_COMPLETENESS` = `0.90`, `DRAWING_MIN_FACT_CONFIDENCE` | константы кода | `core/processing_contract.py` → `core/drawing_quality.py` | когда фрагмент чертежа «потерян» |
| `DRAWING_TIME_BUDGET` | `420` | `core/drawing_processor.py` (`_out_of_time` 587, `_transcribe` 566), `core/providers/qwen_client.py` | бюджет **на лист** (не на документ) |
| `DRAWING_MAX_SHEETS` | `6` | `core/drawing_processor.py` | сколько листов читается моделью |
| `DRAWING_MAX_REGIONS` | `4` | `core/drawing_processor.py`, `core/providers/drawing_vision.py` | сколько областей на лист |
| `DRAWING_RENDER_DPI` | `200` | `core/drawing_processor.py` | — |
| `DRAWING_TRANSCRIBE_MAX_SIDE` / `_MAX_TOKENS` | `2800` / `8192` | `core/providers/qwen_client.py` | — |
| `DRAWING_TRANSCRIBE_TILE_GRID/_OVERLAP/_DEPTH/_PARALLEL` | `2`/`0.06`/`2`/`4` | `core/providers/qwen_client.py` | разрезание листа на плитки |
| `QWEN_SHEET_MAX_SIDE`, `QWEN_REGION_MAX_SIDE`, `QWEN_REGION_UPSCALE`, `QWEN_TITLE_BLOCK_MAX_SIDE`, `QWEN_TITLE_BLOCK_UPSCALE` | — | `core/providers/qwen_client.py`, `core/providers/drawing_vision.py`, `core/drawing_processor.py` | размеры картинок проходов |
| `QWEN_ENDPOINT/_MODEL/_TIMEOUT/_RETRIES/_MAX_TOKENS/_TEMPERATURE` | — | `core/providers/qwen_client.py` | — |
| `VLM_ENDPOINT`, `VLM_PROMPT` | — | `core/ladder/strategies/recognition.py` | уровень 7 |
| `VLM_RESPONSE_CACHE_DIR` | пусто → `{ARTIFACTS_MOUNT}/vlm-cache` | `core/providers/qwen_client.py` | где лежит кэш ответов модели |
| `VLM_TIMEOUT` | `180` | `core/providers/qwen_client.py`, `core/providers/table_vlm.py`, `core/drawing_processor.py`, `core/ladder/strategies/recognition.py` | один вызов модели |
| `OCR_TIME_BUDGET` | `1200` | `core/providers/tesseract_fallback.py`, `core/providers/mineru_parser.py`, `core/ladder/strategies/recognition.py` | — |
| `OCR_FULL_MAX_PAGES` | `300` | те же три файла | — |
| `OCR_DPI`, `OCR_MAX_SIDE`, `OCR_MIN_SIDE`, `OCR_BATCH_PAGES`, `TESSERACT_PAGE_WORKERS` | `200`/`3500`/`900`/`5`/`1` | `core/providers/tesseract_fallback.py` | — |
| `OCR_DESKEW_LIMIT`, `OCR_OSD_MIN_CONFIDENCE`, `OCR_DESPECKLE_MIN_SHARE`, `OCR_LINE_MIN_SHARE`, `OCR_LINE_MAX_THICKNESS` | — | `core/providers/ocr_prepare.py` | подготовка растра |
| `OCR_LAYER_MIN_LINE_CONFIDENCE`, `OCR_LAYER_REPEAT_SHARE`, `OCR_TABLE_LINE_MIN_CONFIDENCE` | `0.4`/`0.6`/`0.0` | `core/providers/mineru_parser.py` (+ `tesseract_fallback.py` для первого) | пороги чистки слоя |
| `MINERU_ENDPOINT/_TIMEOUT/_TIMEOUT_PER_PAGE/_BACKEND/_DEFAULT_LANG` | — | `core/providers/mineru_parser.py` | — |
| `TABLE_VLM_MAX_TABLES`, `TABLE_VLM_DPI` | `8`/`200` | `core/providers/table_vlm.py` | — |
| `SPREADSHEET_MAX_CELLS` | `5 000 000` | `core/ladder/strategies/tabular.py` | — |
| `DEFAULT_CHUNK_SIZE`, `DEFAULT_OVERLAP`, `MIN_CHUNK_SIZE` | `500`/`50`/`30` | `core/strategies/*` | — |
| `PARSE_TASK_SOFT_TIME_LIMIT` / `PARSE_TASK_TIME_LIMIT` | `3600` / `3900` | `ingest/tasks.py`, `core/celery_app.py`, `core/providers/mineru_parser.py`, `core/result_aggregator.py` | предел **задачи** |
| `PARSE_DEADLINE_RESERVE` | `120` | `ingest/tasks.py` | резерв на финал |
| `PARSE_STALE_MARGIN_SECONDS` | `300` | `core/result_aggregator.py` | когда файл признан потерянным |
| `CELERY_TASK_MAX_RETRIES` | `3` | `ingest/tasks.py` | — |
| `CELERY_ML_QUEUE`, `CELERY_INTERACTIVE_QUEUE`, `CELERY_INTAKE_QUEUE` | — | `core/celery_app.py`, `ingest/api.py`, `ingest/tasks.py` | маршрутизация по очередям |
| `RESULT_TTL_SECONDS`, `RESULT_DELIVERY_TYPE`, `RESULT_CALLBACK_URL` | — | `core/result_aggregator.py`, `core/result_delivery.py`, `ingest/api.py` | — |
| `QG_SIMILARITY_THRESHOLD`, `QG_MAX_AGE_DAYS`, `QG_OCR_WARN_CONFIDENCE`, `QG_OCR_BLOCK_CONFIDENCE`, `QG_OCR_PAGES`, `QG_OCR_UNREADABLE_SHARE`, `QG_MAX_SIZE_BYTES` | — | `core/quality/service.py` | гейт качества |
| `GATEWAY_PROFILE_PATH` | `config/profile.yaml` | `core/gateway/profile.py` | правила отсева |
| `GATEWAY_OCR_PROBE_PAGES`, `GATEWAY_OCR_PROBE_SECONDS` | `2`/`60.0` | `core/gateway/text_probe.py` | — |
| `INTAKE_MAX_FILES`, `INTAKE_TIMEOUT`, `INTAKE_TASK_(SOFT_)TIME_LIMIT`, `INTAKE_STATE_TTL_SECONDS` | — | `core/models/intake.py`, `gateway/api.py`, `core/intake/pipeline.py`, `core/intake/tasks.py`, `core/intake/state.py` | — |
| `SOURCE_PREFIX`, `SOURCE_PROBE_EXTENSIONS`, `S3_ATTACHMENTS_BUCKET` | `documents/` | `core/providers/file_locator.py` | где ищется файл |
| `MAX_IMAGE_PIXELS`, `CONFIDENCE_THRESHOLD`, `HF_TOKEN` | — | **никто** (объявлены в `core/config.py`, чтений в коде нет) | — |

## 3. Формы данных

| Форма | Файл | Что это | Что потребитель реально читает |
|---|---|---|---|
| `Fragment` | `core/models/contract.py:305` | единица контракта наружу | `fragment_id`, `type`, `content`, `position`, `confidence`, `completeness`, `provenance`, `warnings`, `review_required` |
| `Content` | `core/models/contract.py:230` | тело фрагмента | `text` (у чертежа — строки листа); `table_data`; `structured_drawing_fields`; `structured_payload` |
| `DrawingField` | `core/models/contract.py:208` | графа штампа / подписанная величина | `category`, `value`, `tolerance`, `unit`, `nature`, `confirmed` |
| `Position` / `SourceSpan` | `core/models/contract.py:111` / `:95` | где во входном файле | `page`, `sheet`, `bbox`, `source_spans` |
| `Provenance` | `core/models/contract.py:148` | как получено | `method`, `strategy_level` (1..7), `source` |
| `TableData` | `core/models/contract.py:182` | таблица | `headers`, `rows`, `total_row` |
| `ExtractedFact` / `FactValue` | `core/models/contract.py:280` / `:249` | числовой факт | `key`, `label`, `value.{kind,raw,number,min,max,unit,operator}` |
| **`DocumentResult`** | `core/models/contract.py:338` | **контракт на документ** | `status`, `warnings`, `fragments`, `pages`, `processed_pages`, `failed_pages`, `unknown_pages`, `content_hash`, `normalized_content_hash`, `review_required`, `error`; `schema_version="2.0"` |
| `ParserError` | `core/models/contract.py:62` | ошибка | `reason_code`, `retryable`, `attempt` |
| Ответы ручек | `core/models/contract.py:377–441` | `ParseResponse`, `SyncPendingResponse`, `BackgroundAcceptedResponse`, `ResultsResponse` | `request_id`, `status`, `documents`, `error` |
| Запросы ручек | `core/models/api.py` | `ParseRequest` (11), `BackgroundParseRequest` (21) | `request_id`, `event_id`, `operation`, `files` |
| `ParsedBlock` / `ParseResult` | `core/models/parse_result.py:10` / `:69` | **внутренний** результат уровня, наружу не уходит | `blocks`, `ladder`, `degraded`, `failed_pages`, `text_layer_stats`, `source_page_count`; `fallback_reserve` — `exclude=True` |
| Состояние пакета в Redis | `core/result_aggregator.py` | ключи `_meta_key`/`_docs_key`/`_start_key`/`_published_key`/`_since_key` | — |
| Строка БД фрагмента | `core/db/models.py` + `core/mappers.py` | — | — |

## 4. Ловушки

1. **`Super-RAG/` — не код.** Это корень правил и планов. Код RAG — в
   `_worktrees/Super-RAG-*`; код Parser — здесь. Перед работой в любой worktree
   сверить `git rev-list --count HEAD..origin/main`: несколько копий отстали
   (например `_worktrees/Super-RAG-fast-route-20261108` — на 164 коммита, и в ней есть
   незакоммиченные правки). Файл, прочитанный в устаревшей копии, может не совпадать с main.
2. **Известное падение теста, не регрессия:** `tests/unit/test_structural_contract.py:269`
   `test_rag_profile_matches_agreed_parameters` падает, потому что в репозитории нет
   `config/rag-consumer-profile.yaml` (есть только `config/profile.example.yaml`). Ожидаемый
   результат `python -m pytest tests/unit -q` — PASS + этот один FAIL.
3. **`DRAWING_SHEETS_SKIPPED:7,8` — список страниц, а не количество.** Собирается в
   `core/drawing_processor.py:716`. В RAG эта же строка разбирается по префиксу
   (`super_rag/parser/completeness.py`), поэтому формат менять нельзя.
4. **`DRAWING_TIME_BUDGET` — бюджет на лист** (`core/drawing_processor.py:798`,
   `:229`), по умолчанию 420 с. Предел задачи — другой: `PARSE_TASK_SOFT_TIME_LIMIT=3600` /
   `PARSE_TASK_TIME_LIMIT=3900`. Их не складывать и не путать.
||||||| fd0c0e5
4. **`DRAWING_TIME_BUDGET` — бюджет на лист** (`core/drawing_processor.py:587`,
   `:566`), по умолчанию 420 с. Предел задачи — другой: `PARSE_TASK_SOFT_TIME_LIMIT=3600` /
   `PARSE_TASK_TIME_LIMIT=3900`. Их не складывать и не путать.
4. **`DRAWING_TIME_BUDGET` — бюджет на лист** (`core/drawing_processor.py:798`,
   `:777`), по умолчанию 420 с. Предел задачи — другой: `PARSE_TASK_SOFT_TIME_LIMIT=3600` /
   `PARSE_TASK_TIME_LIMIT=3900`, и они объявлены на самой задаче
   (`ingest/tasks.py:440-441`), поэтому **именно они и действуют**: `--time-limit=600
   --soft-time-limit=540` в командной строке воркеров — лишь предел пула для задач,
   которые своего не объявили (celery 5.4.0, `worker/request.py:363`: `timeout=time_limit
   or task.time_limit`). Проверено на проде 14.11: 12 разборов завершились успешно
   дольше 600 с (до 2918 с), `SoftTimeLimitExceeded` за 36 ч — ни одного.
   Их не складывать и не путать.
5. **Имя предупреждения решает, попадёт ли документ к человеку.** RAG считает текст
   полным при `partial`, если **все** предупреждения — из списка
   `_ADVISORY_PARSER_WARNINGS` / `_ADVISORY_PARSER_WARNING_PREFIXES`
   (`super_rag/parser/completeness.py`): `DRAWING_COMPLETENESS_UNKNOWN`,
   `DRAWING_NOT_RECOGNIZED`, `DRAWING_SKIPPED_NO_TIME`, префиксы `DRAWING_SHEETS_SKIPPED:`
   и `parser_contract_dropped_invalid_extracted_facts:`. **Новое имя предупреждения
   автоматически отправляет документ на ручную проверку.** Добавляя предупреждение, решить
   осознанно и, если оно про графику, добавить его в тот список в репозитории RAG.
6. **OMP-потоки и замеры OCR.** `OMP_THREAD_LIMIT=1` ставится **только** в
   `core/providers/tesseract_fallback.py:272` — в ветке параллельного чтения страниц и
   только при `TESSERACT_PAGE_WORKERS > 1`. Любой замер OCR вне этой ветки (прямой вызов
   tesseract, скрипт-бенчмарк, `workers == 1`) идёт со всеми ядрами и борьбой за них —
   время получается примерно на порядок хуже реального. Перед замером выставлять
   `OMP_THREAD_LIMIT=1` вручную.
7. **Живые аргументы vLLM — не из репозитория.** `docker-compose.yml` здесь не содержит
   сервиса vLLM, а в RAG `deploy/compose/docker-compose.vllm.yml` — это поставочный
   вариант. Фактическая командная строка прод-vLLM берётся из неотслеживаемого файла в
   домашнем каталоге на сервере. Сверять только на сервере, из репозитория не выводить.
8. **Боковые графы рамки приходят строками записи.** «Инв. № подл.», «Подп. и дата»,
   «Взам. инв. №», «Перв. применение», «Справ. N» модель читает как обычный текст листа и
   они попадают в `content.text`. Те же строки — канонические ключи в
   `core/providers/title_block.py` `LABELS`. Поэтому метка может оказаться в записи без
   значения, дублируя графу, которую путь полей принёс бы правильно.
9. **`content.text` фрагмента чертежа — ещё и внутренний носитель.**
   `core/drawing_transcript.py::_page_transcripts` (88) режет его по `"\n"` и считает
   каждый кусок одной строкой модели, чтобы править абзацы OCR на той же странице. Кроме
   того `core/ladder/drawing_route.volume` (143) считает **символы** `content.text` для
   сравнения «чертёж против OCR» — всё добавленное в текст раздувает этот счёт. При этом
   `core/contract_finalize._content_signature` (36) для чертежа хеширует поля и таблицы,
   а не текст, поэтому `fragment_id` от правок текста не меняется.
10. **Место сборки текста чертежа — `build_fragment`** (строка 460), а при
    `DRAWING_RECORD_COMPOSED` текст пересобирается вторым и последним местом —
    `core/drawing_record.compose_fragment`, из `core/contract_finalize.py`.
||||||| fd0c0e5
10. **Единственное место сборки текста чертежа — `build_fragment`** (строка 329).
10. **Единственное место сборки текста чертежа — `build_fragment`** (строка 460).
    `core/providers/vector_drawing.py` своих `Content` не создаёт и `text` не ставит
    (проверено: конструкций `Content(` там нет). Если появится второй путь — эту строку
    и п. 9 править.
11. **Запись идёт не туда, откуда пришёл файл.** При `S3_READONLY=true` (по умолчанию)
    всё производное пишется в `ARTIFACTS_MOUNT`, а не в исходный бакет
    (`core/providers/storage.py`).
12. **`MAX_IMAGE_PIXELS`, `CONFIDENCE_THRESHOLD`, `HF_TOKEN` объявлены, но не читаются.**
    Менять их бессмысленно — поведение не изменится.
13. **Любой наш ключ `structured_payload` по умолчанию становится искомым
    текстом.** Адаптер RAG (`super_rag/parser/adapter.drawing_search_text`)
    перебирает ключи payload и выводит строкой «Метка: значение» каждый, которого
    нет в его собственном списке `_DRAWING_PAYLOAD_TECHNICAL_KEYS` (там 11 имён:
    `layout`, `model`, `coordinate_reference`, `regions_read`, `regions_skipped`,
    `detected_count`, `recognition_mode`, `sheet_type`, `tables`, `title_block`,
    `page_score`). Поэтому `value_links` ушёл в индекс целиком, с рамками надписей
    и причинами непривязки (42 чанка из 19 699 на проде). Делается это **только**
    у фрагмента с пустым `content.text` — у фрагмента с текстом индексируется
    текст, — но таких фрагментов большинство (на проде 4687 из 5110). Добавляя
    ключ в payload, либо вносить его в список на стороне RAG, либо держать учёт в
    `completeness_mechanisms` (`DRAWING_DIAGNOSTICS_NOT_INDEXED`): это поле RAG не
    читает вовсе (`extra="ignore"`), и в текст оно не попадает ниоткуда.
14. **Хвост «(требует сверки)» ставит RAG, а не Normalizer.**
    `DRAWING_UNCONFIRMED_MARK` в `super_rag/parser/adapter.py` дописывается к
    каждой строке поля, у которого `confirmed is not True`, а подтверждает
    Normalizer почти ничего (порог 0.95). На проде таких хвостов 8633 на 327
    документов, и они доходят до ответа на экране. Из Normalizer это убирается
    только тем, что фрагменту даётся собранный `content.text`: тогда адаптер
    индексирует текст и строк полей не строит вовсе.
15. **`title_block.fix_designation` — единственное место, где меняется группировка
    цифр в обозначении** (`_split_triples`, строка 245): слипшийся хвост делится по
    три цифры, `ТАДУ 687000.001001` → `ТАДУ 687000.001.001`. Это приведение к
    ГОСТ 2.201, и оно верное, но применяется **только** к графе штампа. Текст
    листа и распознавание остаются с прочтением модели, поэтому в одном документе
    оказываются два написания одного обозначения (на проде 207 документов из 327).
    Проверено: `drawing_text.normalize`, `dimension_fidelity.repair_lines` и
    `designation_repair` группировку цифр не меняют ни в одну сторону, а формы с
    лишней точкой («ГОСТ 14.771-76», «4.00007.002») не даёт ни один проход кода —
    они приходят с чтения.
16. **Короткий чанк — это почти всегда текстовый документ, а не чертёж.**
    Замер по выгрузке индекса 50 документов прода: 2901 чанк, медиана 158
    символов, короче 120 — 1195, то есть 41,2 %. Из этих 1195 **ни один** не
    приходит от чертёжного фрагмента, а чанки собранной записи чертежа (52 в
    выборке) имеют медиану 1338 и ни одного короче 120. Короткие — это рамка
    бланка текстового документа (270), заголовок в отрыве от своего текста
    (262), обрывки распознавания (60), строки полей адаптера RAG (63) и
    короткие фразы настоящих документов (516); девять сканированных РЭ/ПС/СТП
    держат 1003 из 1195. Искать причину в ветке чертежей — терять время.
17. **Долю коротких чанков снятием мусора почти не сдвинуть, и это
    арифметика, а не свойство правила.** Снятый короткий чанк уходит и из
    числителя, и из знаменателя: 240 снятых рамок дают 41,2 % → 36,6 %, а не
    41,2 % → 29,8 %, как выходит, если вычесть только из числителя (так было
    посчитано в первом докладе, и это было неверно). Долю двигает упаковка
    соседних обрывков — она переносит чанк из коротких в длинные, а не
    удаляет, — и она на стороне RAG (`super_rag/ingest/normalize/service.py`).
    Со стороны Normalizer честные метрики другие: сколько мусорных чанков
    снято, сколько подписей граф вырезано из читаемого текста и сколько слов
    снова находится поиском.
