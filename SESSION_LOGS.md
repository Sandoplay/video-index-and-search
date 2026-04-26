# Multimodal Video Indexer & Search: Session Logs

Цей файл призначений для збереження контексту розробки між сесіями. Він допомагає швидко зрозуміти, на чому ми зупинилися, які проблеми вирішили та які наступні кроки запланували.

When trying to read the file always consider that it is written in ukrainian. Use according file codings.

---

## Останні зміни (26 квітня 2026 р.)

### 1. Оновлення Vision-Language моделі
- **Модель:** Перейшли на **Qwen 3.5 9B Q4** у файлі `models/Qwen3.5-9B-UD-Q4_K_XL.gguf`.
- **Конфігурація:** Актуальний шлях до моделі вказано в `main.py` як значення за замовчуванням для `--vlm_model`.

### 2. Structured JSON Output для VLM
- **Формат результатів:** Додано structured JSON-вивід із `summary` та `segments`, де сегменти мають `start_frame`, `end_frame`, `start_time`, `end_time` і `description`.
- **Оптимізація:** `main.py` більше не викликає окремий summary-запит у стандартному pipeline; підсумок і timeline-сегменти генеруються одним VLM-запитом.
- **JSON Schema:** Для structured-запитів у `src/vision_analyzer.py` додано `json_schema` в payload до `llama-server`, щоб constrained decoding стабілізував JSON-відповідь.
- **Сумісність:** Старе поле `analysis` лишається в `analysis_result.json` як legacy-представлення, але `src/indexer.py` тепер надає перевагу `segments` і індексує їх як `type = "segment"`.
- **Тести:** Додано `test_structured_analysis.py` для перевірки парсингу JSON, markdown-fenced JSON і clamp-логіки frame indices.

### 3. Performance Timings
- **Заміри часу:** У `main.py` додано wall-clock таймінги для media processing, transcription, VLM-аналізу кожного відео, загального Stage 3 та всього pipeline.
- **JSON Output:** `analysis_result.json` тепер містить поле `timings`, зокрема `vlm_client_wall_seconds`, `vlm_server_total_seconds`, `vlm_server_prompt_seconds`, `vlm_server_generation_seconds`, `vlm_client_overhead_or_queue_seconds`, `stage3_total_wall_seconds`, `stage3_executor_workers` і `stage3_server_parallel_slots`.
- **Уточнення метрик:** Per-video client wall time включає очікування HTTP/slot queue, а server timings беруться з `timings` у відповіді `llama-server` і краще відображають активну prompt/generation роботу моделі.
- **Паралельність:** `max_parallel_requests` у `main.py` встановлено в `2`, щоб відповідати `llama-server --parallel 2` і не створювати зайву серверну чергу з боку клієнта.
- **Планування VLM-задач:** Додано CLI-параметри `--parallel_requests` та `--server_parallel`; перед Stage 3 відео сортуються за `vlm_estimated_cost` від найважчих до найлегших.
- **Cost Metrics:** У `timings` додаються `vlm_frames_total`, `vlm_frames_sent`, `vlm_frames_bytes`, `vlm_transcription_chars`, `vlm_estimated_cost` і `vlm_schedule_rank` для аналізу порядку запуску та benchmark `parallel 1 vs 2`.

---

## Останні зміни (9 квітня 2026 р.)

### 1. Vector Search & Indexing (LanceDB)
- **База Даних:** Інтегровано `lancedb` як локальну векторну базу даних (зберігається в `data/lancedb`).
- **Embedding:** Використано легку модель `sentence-transformers/all-MiniLM-L6-v2` для генерації 384-вимірних векторів з текстових даних (розмірковувань та описів кадрів).
- **Скрипти:** 
  - Створено `src/indexer.py`, який парсить `analysis_result.json`, створює ембединги для загального підсумку відео (Summary) та кожного окремого кадру (Frame), та зберігає їх у LanceDB.
  - Створено `search.py` — CLI-інтерфейс для виконання семантичного пошуку (напр., `python search.py "neck stretching"`). Показує відстань (distance), відео та точний таймкод.
- **Автоматизація:** Оновлено `main.py`. Тепер після генерації JSON результатів автоматично запускається `VideoIndexer` (Stage 5), щоб додавати нові відео прямо в пошуковий індекс.

---

## Останні зміни (28 березня 2026 р.)

### 1. Оновлення Vision-Language моделі
- **Модель:** Перейшли з Qwen3-VL 8B на новішу **Qwen 3.5 9B (Instruct)**.
- **Квантування:** Використовуємо **Q8_0** (файл `models/Qwen3.5-9B-Q8_0.gguf`).
- **Продуктивність:** Завдяки тестам на RTX 5060 Ti 16GB збільшили ліміт контексту до **72,000 токенів**. Це дозволяє обробляти до ~50 кадрів за один батч.
- **Виправлення:** Налаштували парсинг `reasoning_content`, щоб модель не видавала порожні відповіді під час тривалого "обдумування". Збільшили `max_tokens` до 4096 для Summary.

### 2. Покращення стабільності Whisper
- **Кешування:** Тепер Whisper завантажує моделі локально в папку проекту `models/`, а не на диск C.
- **Точність:** Додали параметри `condition_on_previous_text=False` та `word_timestamps=True`. Це запобігає зацикленню (hallucination loops) на шумних відео або відео з музикою.
- **VLM Context:** Оновили промпти, щоб Qwen знав про можливі помилки в аудіо-транскрипції та більше спирався на візуальний ряд.

### 3. Оптимізація системи
- Оновили **PyTorch до версії 2.11.0+cu130** для кращої підтримки CUDA 13 та архітектури Blackwell.
- Очистили системний диск C від старих моделей Whisper (~6 ГБ).

---

## Наступні кроки (To-Do)
1. **[Виконано]** Vector Search / Indexing: Інтегровано LanceDB та sentence-transformers для пошуку за метаданими.
2. **[Виконано]** CLI Search: Створено `search.py` для семантичного пошуку.
3. **UI / Web Interface (опціонально):** Створення веб-інтерфейсу (наприклад, на FastAPI або Streamlit) для зручнішого пошуку та перегляду результатів (із картинками кадрів).
4. **Audio Separation (опціонально):** Розглянути інтеграцію Demucs для покращення якості транскрипції шляхом відділення голосу від фонової музики.

---

## Інструкція для AI-асистента
При продовженні роботи над проектом у новій сесії:
1. **Прочитай цей файл (`SESSION_LOGS.md`).**
2. **Додавай новий запис** у розділ "Останні зміни" на початку списку (після заголовка), вказуючи дату.
3. **Оновлюй розділ "Наступні кроки"**, видаляючи виконані завдання та додаючи нові.
4. **Формат запису:** Коротко описуй технічні рішення, змінені файли та результати тестів.
