### Документация к сервису FastAPI для анализа тональности

## Введение

Эта документация предоставляет информацию о сервисе FastAPI для анализа тональности текста. Сервис использует фреймворк FastAPI вместе с библиотекой transformers.

## Установка

### Зависимости

Убедитесь, что вы установили необходимые зависимости перед запуском сервиса. Вы можете установить их с помощью следующей команды:

```
pip install -r requirements.txt
```

## Запуск сервиса

Чтобы запустить сервис FastAPI для анализа тональности, выполните следующую команду:

```
uvicorn main:app --reload
```

По умолчанию сервис будет доступен по адресу http://127.0.0.1:8000

## Эндпоинты

## 1. Корневой эндпоинт

### GET /

-   Описание: Возвращает простое сообщение о том, что служба FastAPI запущена.
-   Пример ответа:
    ```json
    {"FastApi service started!"}
    ```

## 2. Эндпоинт анализа тональности

### GET /{text}

-   Описание: Выполняет анализ тональности предоставленного текста с использованием предварительно обученной модели анализа тональности.
-   Параметры:
-   text (str): Входной текст для анализа тональности.
-   Пример запроса:

```
curl -X GET "http://127.0.0.1:8000/Привет%20мир"
```

-   Пример ответа:

```json
[{ "label": "POSITIVE", "score": 0.9997 }]
```

## 3. Эндпоинт предсказания

### POST /predict/

-   Описание: Выполняет анализ тональности текста, предоставленного в теле запроса, с использованием предварительно обученной модели анализа тональности.
-   Тело запроса:

```json
{
    "text": "Входной текст для анализа тональности."
}
```

-   Пример запроса:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "Привет мир"}' "http://127.0.0.1:8000/predict/"
```

-   Пример ответа:

```json
{ "label": "POSITIVE", "score": 0.9997 }
```

## 4. Эндпоинт анализ массива текстов

### POST /analyze_texts/

-   Описание: Выполняет анализ тональности массива текстов, предоставленного в теле запроса, с использованием предварительно обученной модели анализа тональности.
-   Тело запроса:
    Ожидается JSON-массив, в котором каждый объект содержит поле "text", представляющее собой текст для анализа.

```json
[{ "text": "Привет" }, { "text": "Мир" }]
```

-   Пример запроса:

```bash
curl -X POST -H "Content-Type: application/json" -d '[{"text": "Привет"}, {"text": "Мир"}]' "http://127.0.0.1:8000/analyze_texts/"
```

-   Пример ответа:

```json
[
    { "text": "Привет", "sentiment": [{ "label": "POSITIVE", "score": 0.9999 }] },
    { "text": "Мир", "sentiment": [{ "label": "POSITIVE", "score": 0.9998 }] }
]
```

## 5. Запуск сервиса визуальной работы на Streamlit

Чтобы запустить сервис Streamlit для анализа тональности, выполните следующую команду в командной строке:

```
python streamlit_app.py
```

```
streamlit run streamlit_app.py
```
По умолчанию сервис будет доступен по адресу Local URL: http://localhost:8501


## Заключение

Сервис FastAPI для анализа тональности предоставляет простой и эффективный способ выполнения анализа тональности входного текста. Не стесняйтесь изучать и интегрировать этот сервис в ваши приложения.

Выполнил Сергеев С.В.
