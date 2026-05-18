# FastMCP (CS AI Bridge tools) - streamable HTTP on port 8000

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY cs_ai_bridge_mcp /app/cs_ai_bridge_mcp

EXPOSE 8000

CMD ["fastmcp", "run", "cs_ai_bridge_mcp/server.py:mcp", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8000"]
