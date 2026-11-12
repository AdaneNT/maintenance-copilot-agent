FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir ".[api]"

ENV MAINT_DATA_DIR=/app/data
EXPOSE 8000
USER nobody
CMD ["uvicorn", "maint_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
