FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
EXPOSE 8000
# Streamable HTTP MCP endpoint at http://<host>:8000/mcp
CMD ["drug-safety-mcp", "--http", "--host", "0.0.0.0", "--port", "8000"]
