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
8. Заменить строковый `error` на объект ошибки.
9. В RAG развернуть входящий lifecycle batch в отдельные задачи Parser только для событий, требующих разбора; сохранять исходный `event_id` и связь `asset_id` ↔ `s3_fileid`.
10. Проверить пачку с несколькими файлами, повторную доставку, смешанные операции и независимые результаты Parser по `event_id`, а также переход от технического `dialog_id` к контракту без него.
