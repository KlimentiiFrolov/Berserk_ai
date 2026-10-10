# Инструкция по запуску BerserkAI (RAG)

## Требования

- Python 3.12+
- Git
- Docker

## 1. Клонирование репозитория

```bash
git clone https://github.com/KlimentiiFrolov/Berserk_ai
cd Berserk_ai
```

## 2. Виртуальное окружение и зависимости

```bash
python -m venv .venv

# Linux / macOS / WSL
source .venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1

pip install -e .
```

## 3. Запуск Qdrant в Docker

Простой вариант (данные пропадут при удалении контейнера):

```bash
docker run -d --name berserkai-qdrant -p 6333:6333 qdrant/qdrant
```

Вариант с сохранением данных на диск (рекомендуется):

```bash
docker run -d --name berserkai-qdrant -p 6333:6333 \
  -v /путь/к/папке/хранения:/qdrant/storage \
  qdrant/qdrant
```

Например, в WSL: `-v /mnt/с/berserk-rag/qdrant_store:/qdrant/storage`.

Проверьте, что Qdrant работает: откройте `http://localhost:6333/dashboard` или выполните

```bash
curl http://localhost:6333/collections
```

## 4. Настройка конфигурации

Создайте файл `.env` (или заполните конфиг, который использует проект) на примере `.env.example`

## 5. Индексация данных

Перед **первым** запуском нужно загрузить документы в Qdrant, выполните:

```bash
python notebooks\build_rag.py
```

## 6. Запуск

Запускайте каждый компонент в отдельном терминале (с активированным окружением).

**Веб-интерфейс:**

```bash
python -m webui
```

**Telegram-бот:**

```bash
python -m bot
```