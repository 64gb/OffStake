# МФЦ RAG Система (Question-Answering)

Локальная RAG-система для вопрос-ответов по базе знаний МФЦ на основе документов (регламенты, инструкции).

## 🏗️ Архитектура

- **LLM**: Ollama с моделью `qwen2.5:7b-instruct`
- **Embeddings**: `mxbai-embed-large` через Ollama
- **Vector DB**: Qdrant (косинусное сходство)
- **Backend**: FastAPI + LangChain
- **Frontend**: Streamlit

## 📁 Структура проекта
/workspace
├── docker-compose.yml # Конфигурация всех сервисов
├── Dockerfile.backend # Dockerfile для FastAPI backend
├── Dockerfile.frontend # Dockerfile для Streamlit frontend
├── requirements.txt # Python зависимости
├── main.py # Backend логика (RAG, ingest, ask)
├── app.py # Frontend (Streamlit UI)
├── README.md # Этот файл
└── data/ # Папка для документов (создается автоматически)
