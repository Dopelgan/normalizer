# Изменения контракта Parser

Это согласованное изменение Parser и RAG; локальная реализация не означает, что контракт уже развёрнут на сервере.

## Связь с пачкой lifecycle-событий Backend → RAG

`POST /internal/rag/lifecycle` принимает от Backend `events` из 1–256 элементов (см. `docs/BACKEND-ADMIN-CONTRACT-CHANGES.md`). **Это пачка только на входе RAG.** RAG обрабатывает её поэлементно: для каждого принятого события создаёт отдельную задачу с его `event_id` и `idempotency_key`. Повторное событие не создаёт повторной задачи Parser.

- `asset.created` → фоновый Parser `operation: "create"`;
- `asset.updated`, `asset.restored`, `asset.reprocess_requested` → фоновый Parser `operation: "update"`;
- `asset.deleted`, `asset.withdrawn` обрабатываются в RAG без вызова Parser.

Для одного события, требующего разбора, RAG отправляет **один** `POST /internal/v1/parse/background` с тем же `event_id` и одним значением в `s3_fileid`. `request_id` относится к отдельному запросу Parser. Если входящая пачка содержит несколько таких событий, RAG отправляет несколько независимых запросов Parser; общий batch ID в Parser не требуется. Результат или ответ Parser сопоставляется с исходным событием по `event_id` и с файлом по `s3_fileid`. Для `completed`/`partial` результат содержит ровно один документ с тем же `s3_fileid`. Ошибка или частичный результат одного файла не меняет статус другого элемента пачки.

`asset_id` и `storage_key` из lifecycle-события не переименовывают автоматически в `s3_fileid`: RAG должен передать Parser идентификатор, по которому Parser действительно может прочитать тот же файл, и сохранить соответствие с исходным `asset_id`. Этот маппинг необходимо закрепить интеграционным тестом перед реализацией нового endpoint.

## Главное правило по идентификаторам

- Синхронная обработка `POST /internal/v1/parse`: `dialog_id` оставляем.
- Фоновая обработка `POST /internal/v1/parse/background`: `dialog_id` убираем, используем только `event_id`.
- В фоновой обработке `event_id` связывает lifecycle-событие, запрос Parser и результат Parser.
- `s3_fileid` не переименовываем и не меняем его тип.
- `event_id`, пришедший от Backend, не заменяем новым UUID при постановке отдельной задачи Parser.

## 1. Синхронный контракт — без изменения

### `POST /internal/v1/parse`

```json
{
  "request_id": "parse-123",
  "dialog_id": "dialog-456",
  "s3_fileid": ["cluster-a/document.pdf"]
}
```

`dialog_id` здесь остаётся, потому что синхронный разбор вызывается из пользовательского диалога.

Внутри Parser синхронный разбор может идти в отдельную очередь Celery (`CELERY_INTERACTIVE_QUEUE`, см. README, «Очередь синхронного разбора»), чтобы вложения чата не ждали пакетную переобработку. Контракт ручки от этого не меняется.

Ответ также остаётся в текущем формате:

```json
{
  "request_id": "parse-123",
  "dialog_id": "dialog-456",
  "documents": []
}
```

## 2. Фоновый контракт — убрать `dialog_id`

### `POST /internal/v1/parse/background`

### Было

```json
{
  "request_id": "parser-request-123",
  "event_id": "event-123",
  "dialog_id": "event-123",
  "operation": "create",
  "s3_fileid": ["cluster-a/document.pdf"]
}
```

### Станет

```json
{
  "request_id": "parser-request-123",
  "event_id": "event-123",
  "operation": "create",
  "s3_fileid": ["cluster-a/document.pdf"]
}
```

Изменилось только это поле:

```diff
- "dialog_id": "event-123"
```

В фоне `dialog_id` больше не передаём и не проверяем.

Локальный RAG runtime уже отправляет фоновый `event_id` без технического `dialog_id=event_id`. Совместимость с развёрнутым Parser необходимо подтвердить интеграционным тестом перед включением нового пути; диалоговый Parser-контракт с `dialog_id` остаётся отдельным.

Для Backend-facing `GET /graph` остаётся уточнить семантику Parser `relations`: стабильные ID исходного и целевого объекта, тип направленной связи, fragment ID доказательства и калиброванные `weight`/`confidence`. Локальный Parser-контракт сейчас допускает произвольные словари `graph_nodes`/`relations`; RAG сохраняет их в provenance, но не публикует как доказанные document-document рёбра без этих полей.

### Ответ `202 Accepted`

```json
{
  "request_id": "parser-request-123",
  "event_id": "event-123",
  "operation": "create",
  "accepted": true,
  "status": "queued"
}
```

Из ответа также убирается только `dialog_id`.

## 3. Результат фонового Parser → RAG

### `POST /internal/v1/parser/results`

```json
{
  "request_id": "parser-request-123",
  "event_id": "event-123",
  "operation": "create",
  "status": "completed",
  "documents": [
    {
      "s3_fileid": "cluster-a/document.pdf",
      "status": "completed",
      "content_hash": "sha256:binary-file",
      "normalized_content_hash": "sha256:normalized-text",
      "mime_type": "application/pdf",
      "pages": 18,
      "document_metadata": {
        "doc_id": "ТАДУ 405220.002",
        "doc_version": "изм. 32",
        "language": "ru",
        "source_path": "cluster-a/document.pdf",
        "updated_at": "2026-09-21T12:00:00Z",
        "status": "completed"
      },
      "fragments": [],
      "blocks": [],
      "warnings": []
    }
  ],
  "error": null
}
```

В envelope больше нет `dialog_id`. Остальные поля результата сохраняются.

Polling `GET /internal/v1/parse/results/{request_id}` должен возвращать такой же envelope.

## 4. Общее извлечение фактов и значений

### Зачем

RAG должен видеть не только текстовый chunk, но и явно извлечённые значения из любого документа: числа, диапазоны, даты, сроки, единицы измерения, идентификаторы, перечисления и значения из таблиц. Это общий продуктовый контракт, а не набор полей под конкретные демонстрационные вопросы.

В `ParsedFragment` добавить поле `extracted_facts`.

### Пример

```json
{
  "fragment_id": "cluster-a/document.pdf:page-4:block-2",
  "type": "text",
  "content": {
    "text": "Максимальная температура — 80 °C. Допустимый диапазон давления — 0,4–0,6 МПа.",
    "table_data": null,
    "image_ref": null,
    "formula_mathml": null,
    "parsed_expression": null
  },
  "position": {
    "page": 4,
    "sheet": null,
    "bbox": [10, 20, 300, 400],
    "order": 12
  },
  "section_title": "Технические характеристики",
  "confidence": 0.96,
  "completeness": 1.0,
  "provenance": {
    "method": "pdf_text",
    "strategy_level": 1,
    "source": "cluster-a/document.pdf"
  },
  "extracted_facts": [
    {
      "fact_id": "cluster-a/document.pdf:page-4:block-2:fact-1",
      "fragment_id": "cluster-a/document.pdf:page-4:block-2",
      "key": "document.parameter",
      "label": "максимальная температура",
      "value": {
        "kind": "number",
        "raw": "80 °C",
        "number": 80,
        "unit": "°C",
        "operator": "<="
      },
      "confidence": 0.98
    },
    {
      "fact_id": "cluster-a/document.pdf:page-4:block-2:fact-2",
      "fragment_id": "cluster-a/document.pdf:page-4:block-2",
      "key": "document.parameter",
      "label": "допустимый диапазон давления",
      "value": {
        "kind": "range",
        "raw": "0,4–0,6 МПа",
        "min": 0.4,
        "max": 0.6,
        "unit": "MPa",
        "operator": "range"
      },
      "confidence": 0.94
    }
  ],
  "graph_nodes": [],
  "relations": [],
  "structured_drawing_fields": [],
  "structured_payload": null
}
```

Правила для `extracted_facts`:

- `fact_id` стабилен в рамках одного файла и фрагмента;
- `fragment_id` связывает факт с исходным текстом;
- `key` — расширяемый ключ типа факта, без фиксированного списка под один проект;
- `label` сохраняет название параметра так, как оно извлечено из документа;
- `value.kind`: `number`, `range`, `date`, `duration`, `text`, `boolean`, `identifier`, `enum`;
- `value.raw` сохраняет исходную запись;
- `value.number` используется для одного числа;
- `value.min` и `value.max` используются для диапазона;
- `value.unit` хранит единицу измерения, если она есть;
- `value.operator`: `=`, `<=`, `>=`, `<`, `>`, `range`;
- Parser не должен придумывать значение или единицу, которых нет в документе;
- Parser извлекает факт, но не решает, нужен ли он для конкретного пользовательского вопроса;
- для таблиц факты должны сохранять координаты строки и столбца через provenance/structured payload.

## 5. Provenance и стабильные фрагменты

`fragment_id` должен оставаться одинаковым при повторном разборе того же файла.

В provenance добавить:

```json
{
  "method": "pdf_text",
  "strategy_level": 1,
  "source": "cluster-a/document.pdf",
  "source_file_id": "cluster-a/document.pdf",
  "source_page": 4
}
```

Без этого RAG не сможет открыть страницу, показать доказательство числового ответа и обновить старый фрагмент без дубля.

## 6. Частичный результат

Для `status: partial` добавить список обработанных и необработанных страниц:

```json
{
  "request_id": "parser-request-123",
  "event_id": "event-123",
  "operation": "create",
  "status": "partial",
  "documents": [
    {
      "s3_fileid": "cluster-a/document.pdf",
      "status": "partial",
      "pages": 18,
      "processed_pages": [1, 2, 3, 4, 5, 6, 8, 9],
      "failed_pages": [7],
      "fragments": [],
      "warnings": ["OCR_LOW_CONFIDENCE: page 7"]
    }
  ],
  "error": null
}
```

RAG индексирует только обработанные фрагменты и переводит файл в `review_required`.

## 7. Структурированная ошибка

Вместо строки:

```json
{"error": "Parser failed"}
```

передавать объект:

```json
{
  "error": {
    "reason_code": "UNSUPPORTED_FORMAT",
    "message": "Parser does not support this MIME type",
    "retryable": false,
    "attempt": 1
  }
}
```

Минимальные `reason_code`:

```text
SOURCE_NOT_FOUND
SOURCE_READ_ERROR
UNSUPPORTED_FORMAT
OCR_FAILED
EXTRACTION_FAILED
TABLE_EXTRACTION_FAILED
PARSER_TIMEOUT
INTERNAL_ERROR
```

## 8. Что нужно изменить в Parser/RAG

1. В фоновых моделях после согласованного перехода убрать `dialog_id`; использовать только `event_id`.
2. В синхронных моделях `dialog_id` оставить.
3. Добавить `normalized_content_hash` в `ParsedDocument`.
4. Добавить `extracted_facts` в `ParsedFragment`.
5. Сделать `fragment_id` стабильным.
6. Добавить `source_file_id` и `source_page` в provenance.
7. Добавить `processed_pages` и `failed_pages` для `partial`.

### Покрытие страниц: неизвестность не означает успешный разбор

`processed_pages` содержит только страницы с наблюдёнными блоками либо с
явным подтверждением пустоты от producer. Отсутствие текста или блоков само
по себе не подтверждает пустую страницу. `failed_pages` имеет приоритет над
наблюдёнными результатами. Необязательный список `unknown_pages` перечисляет
ожидаемые страницы без результата и без подтверждённого отказа; он не
подменяется списком ошибок.

Неизвестное общее количество (`pages: null`), противоречивые номера или
непустой `unknown_pages` дают `status: partial` и машинное предупреждение
`PAGE_COUNT_UNKNOWN`, `PAGE_COUNT_MISMATCH` или `PAGE_COVERAGE_UNKNOWN: page N`.
Пустой `unknown_pages` при `pages: null` не означает полного охвата. Наличие
блока подтверждает результат на странице, но не полноту её содержимого.
Внутренний `source_page_count` подтверждается контейнером исходного документа;
прежний `page_count`, выведенный из количества блоков, не подтверждает границу.
Для Word/TXT единица логическая, а не заявленное число печатных страниц.

Свежая и повторная выдача сохраняют одинаковые страницы, статус и
предупреждения, включая `pages: null`. Записи кеша без актуального внутреннего
`page_coverage_version` повторно разбираются при штатном запросе; прежние
отчёты нельзя повысить до нового формата без разбора. Structural profile и
`schema_version: 2.0` сохраняются.
8. Заменить строковый `error` на объект ошибки.
9. В RAG развернуть входящий lifecycle batch в отдельные задачи Parser только для событий, требующих разбора; сохранять исходный `event_id` и связь `asset_id` ↔ `s3_fileid`.
10. Проверить пачку с несколькими файлами, повторную доставку, смешанные операции и независимые результаты Parser по `event_id`, а также переход от технического `dialog_id` к контракту без него.

## Актуальное дополнение от 19.10.2026

Ранее описанные изменения сохраняются. Структурный выход, координаты,
профиль обработки и качество чертежей дополнены согласованной схемой `2.0`:
[Parser → RAG](RAG-NORMALIZER-CONTRACT-V2.md). Финальная нарезка принадлежит
RAG; Parser передаёт исходные блоки и полные таблицы без токенового усечения.

## Вложения чата как `s3_fileid` (11.11.2026)

Контракт `/internal/v1/parse` не меняется: для вложения чата RAG передаёт в
`s3_fileid` uuid вложения (голый, в нижнем регистре, без расширения). Backend
хранит такие файлы не в основном бакете, а в `chat-attachments`, исходное имя
— в метаданных объекта `filename` (URL-encoded). Раньше Parser искал uuid
только в `S3_BUCKET` и отвечал `source_not_found`.

Теперь при заданном `S3_ATTACHMENTS_BUCKET` Parser, не найдя файл в
`S3_BUCKET`, проверяет ровно `s3://<S3_ATTACHMENTS_BUCKET>/<uuid>`; тип файла
определяется по расширению имени из метаданных, иначе по сигнатуре. Любой
другой вид идентификатора (путь, `..`, расширение, верхний регистр) в бакет
вложений не попадает. По умолчанию настройка пуста; на проде нужно
`S3_ATTACHMENTS_BUCKET=chat-attachments`. Подробности — README, раздел
«Где Parser ищет файлы».

## Повёрнутые листы чертежей (11.11.2026)

Схема `2.0` не меняется. В `structured_payload.layout` фрагмента `drawing`
(скан, ветка `vision_plus_vlm`) добавлено поле `turn` — поворот листа кратно
90° против часовой (0, 90, 180, 270), которым штамп приведён в правый нижний
угол, — и при повороте `turn_source` (`osd` или `geometry`), при отказе
`turn_rejected`. Рамки областей и надписей модели (`bbox`) задаются в
системе уже повёрнутого и выправленного листа (`coordinate_reference:
deskewed_page`); при `turn != 0` они не совпадают с исходным растром без
обратного поворота. Опознание векторного чертежа учитывает `/Rotate`
страницы при поиске сетки штампа.

Скан или векторная страница, текст которой в большинстве повёрнут кратно
90° (альбомная страница без `/Rotate`), разбирается по прямо стоящему
листу: фрагменты и строки внутри них идут в порядке чтения прямого листа,
абзацы не слипаются, штамп ищется в его правом нижнем углу (у полей
векторного чертежа — категории `title_block`/`material`). Поля и схема не
меняются; `position.bbox` по-прежнему в долях страницы, как её видно
(с учётом `/Rotate`), без поворота.

## Повтор `request_id` и пропавшая задача (12.11.2026)

- Повтор `POST /internal/v1/parse/background` или `POST /internal/v1/parse` с `request_id` **готового** запроса (`completed`/`partial`) и теми же `s3_fileid` с теми же операциями и `event_id`/`dialog_id` не запускает разбор заново: фоновая ручка отвечает тем же `202` (`status: "queued"`, как и раньше), синхронная — готовым результатом. Результат берётся `GET /internal/v1/parse/results/{request_id}`.
- Повтор с `request_id` запроса, закончившегося `error`, запускает разбор заново (ретрай с тем же `request_id`).
- Повтор незавершённого запроса — `409`, как и раньше; повтор готового запроса с другим набором файлов, операцией или `event_id`/`dialog_id` — тоже `409`.
- `partial` считается готовым: многофайловый синхронный запрос, где один файл получил ошибку, повторяется с новым `request_id`.
- Если задача файла стартовала дольше `PARSE_TASK_TIME_LIMIT + PARSE_STALE_MARGIN_SECONDS` назад и результата не оставила (жёсткий лимит, OOM, SIGKILL), файл получает документ со `status: "error"` и ошибкой `PARSER_TIMEOUT` (`retryable: true`), и запрос перестаёт быть `processing`. Ошибка выставляется при чтении состояния (`GET .../results/{request_id}`), поэтому гарантия относится к режиму опроса (`RESULT_DELIVERY_TYPE=none`, по умолчанию): push-канал о таком файле не сообщает. Если настоящий результат всё же придёт позже, он заменяет ошибку.

## Текст в рамке ЕСКД и слой распознавания (12.11.2026)

Схема `2.0`, поля и ручки не меняются. Под флагами (README, «Флаги
разбора»; на проде включены) меняется состав фрагментов:

- лист векторного PDF с прозой в рамке ЕСКД (ПОК, СТО, ТУ, ТЗ, письма) или
  из одной сетки таблиц приходит фрагментами `text`/`table`, а не `drawing`
  (`DRAWING_TEXT_IN_FRAME_ENABLED`);
- «таблица» MinerU на половину листа и больше с прозой внутри приходит
  обычным текстом: многостраничная книга в рамке больше не одна таблица
  (`FRAME_TABLE_GUARD_ENABLED`);
- у скана на страницах, где MinerU дал свой текст, нет фрагментов
  `ocr_layer` из строк с уверенностью ниже 0.4
  (`OCR_LAYER_DROP_LOW_CONFIDENCE`).
- классификатор «чертёж или текст» (`DRAWING_PAGE_CLASSIFIER_ENABLED`,
  поверх `DRAWING_TEXT_IN_FRAME_ENABLED`): лист ТЗ/ПМ/ТУ с рисунком-картинкой
  и штампом текстового документа приходит текстом, рисунок — отдельным
  фрагментом; в текстовом документе картинки меньше 6 % листа (логотип,
  печать, подпись) остаются `image` и не становятся `drawing`.

После включения флагов повторный разбор даёт на таких страницах другие
`fragment_id`: он считается от содержимого фрагмента (раздел 5).

## Шум распознавания сканов (13.11.2026)

Контракт не меняется: меняется текст фрагментов растровых документов (скан, картинка). Векторные PDF не затрагиваются.
Четыре флага, все по умолчанию выключены (README, «Флаги разбора»): `OCR_WORD_FIX_ENABLED` — латиница и курсив в
русских словах по словарю моделей Tesseract; `OCR_KEEP_STRUCTURED_TABLES` — спецификация ЕСКД во весь лист остаётся
таблицей; `OCR_TABLE_LINE_MIN_CONFIDENCE` — без повтора таблицы мусорными строками через сетку; `OCR_SPLIT_WIDE_GAPS` —
разворот в две колонки читается по колонкам. В `text_layer_stats` растра добавлены `low_confidence_table_lines` и
`word_fix` (`latin_words`, `confusion_words`, `junk_tokens`).

## Полное чтение листов-картинок (13.11.2026)

Схема и поля не меняются. Два флага, оба по умолчанию выключены (README, «Флаги разбора»):

- `DRAWING_UNMEASURED_NOT_PARTIAL` — фрагмент `drawing` без измеренной полноты остаётся `review_required`, но
  документ `partial` только при реальной потере (`missing_regions`, `DRAWING_NOT_RECOGNIZED`,
  `DRAWING_SKIPPED_NO_TIME`, `DRAWING_SHEETS_SKIPPED`);
- `DRAWING_IMAGE_FULL_READ` — у входа-картинки (JPG, PNG, TIFF) первым приходит фрагмент `drawing` всего листа, и
  когда MinerU нарезал из листа виды (они остаются своими фрагментами), и когда лестница назвала лист текстом при
  рамке чертежа. В его `content.text` — весь текст листа, переписанный Qwen-VL построчно (штамп, номер листа,
  изменения, примечания, технические требования); `structured_payload.transcript_line_count` — число строк. Вместе с
  первым флагом непрочитанная область на странице с переписанным текстом не делает документ `partial`.

После включения у таких листов появляется новый фрагмент, у прочих `fragment_id` не меняются.

## Примечания чертежей на растре (13.11.2026)

Контракт не меняется. `OCR_WORD_FIX_ENABLED` дополнительно читает курсив ГОСТ в примечаниях чертежей
(«Cozobopa» — «Договора», «Macca usdenua» — «Масса изделия»). Новый флаг `OCR_LAYER_DROP_REPEATED`
(по умолчанию выключен): на странице с текстом MinerU фрагменты `ocr_layer`, восстановленные из слоя
распознавания и повторяющие этот текст или собранные из графики без русских слов, не выдаются. В
`text_layer_stats` растра добавлен `dropped_repeated`. Блок без русских слов отбрасывается только на русской
странице: на английском скане он остаётся. Блок с обозначением, которого нет в тексте страницы у MinerU или
модели (код KKS «10JEB10CT001», «⌀14», «R1,5»), не отбрасывается ни по одному правилу.

## Перечитывание неразобранной таблицы скана (13.11.2026)

Схема не меняется. Флаг `TABLE_VLM_RETRY_ENABLED` (по умолчанию выключен, README, «Флаги разбора»): таблица
растра, которую раньше разбирали на обрывки строк OCR (`table_from_lines.degrade`), перечитывается Qwen-VL и
приходит фрагментом `table` с `provenance.method` = `vlm_escalation` (уровень 7) без `table_html`. Строки слоя
с новыми словами остаются текстом. В `text_layer_stats` добавлен `vlm_tables`. Не вышло — прежний исход.

## Трассировка десяти чертежей (13.11.2026)

Схема не меняется. Три флага, все по умолчанию выключены (README, «Флаги разбора»):

- `DRAWING_SHEET_MISSING_PAGES` — у скана чертежа в PDF лист без фрагмента `drawing` получает фрагмент `drawing`
  всего листа (как при чтении листов целиком), даже если на других листах фрагменты `drawing` уже есть.
  `fragment_id` нового фрагмента — `<doc_id>-frag-000-<лист>` (у листа 1 — `<doc_id>-frag-000`), он встаёт перед
  фрагментами своей страницы; у прочих фрагментов `fragment_id` не меняются;
- `RASTER_FRAME_THIN_LINES` — растровый детектор может назвать чертежом лист, который прежде шёл страницей текста;
  в `classification.signals` такого листа — `thin_line_frame`, `thin_line_rules`. Жанр документа и, значит,
  маршрут разбора и предлагаемый гриф у таких листов меняются;
- `DRAWING_MATERIAL_NO_TOLERANCE` — у поля `material` `tolerance` = `null`, `source_of_tolerance` = `unknown`.

