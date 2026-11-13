FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
COPY . .
CMD ["python", "-m", "nowcast_data.cli", "backtest", "data/realtime.json", "--draws", "200"]
