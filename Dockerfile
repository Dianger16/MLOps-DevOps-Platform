FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY api/     ./api/
COPY model/   ./model/

# Non-root user
RUN addgroup --system appgroup && adduser --system --group appuser
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8080/health').raise_for_status()"

CMD ["uvicorn", "api.serve:app", "--host", "0.0.0.0", "--port", "8080"]
