# FastAPI LLM proxy (OpenAI / Gemini from Redis) - port 8080

FROM python:3.11-slim

WORKDIR /app

ARG APP_BUILD_ID=strip-bridge-fields-v2
ENV CS_AI_BRIDGE_API_BUILD_ID=${APP_BUILD_ID}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY cs_ai_bridge_api /app/cs_ai_bridge_api

EXPOSE 8080

CMD ["uvicorn", "cs_ai_bridge_api.app:app", "--host", "0.0.0.0", "--port", "8080"]
