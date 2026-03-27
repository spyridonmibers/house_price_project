# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости для CatBoost
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем файлы зависимостей
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы приложения
COPY app.py .
COPY catboost_model.cbm .

# Создаём папку для данных (опционально)
RUN mkdir -p /app/data

# Открываем порт для Streamlit
EXPOSE 8501

# Переменные окружения для Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Запуск приложения
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]