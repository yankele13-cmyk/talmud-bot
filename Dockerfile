FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY rag_factory/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn

# Copy app code
COPY app/ /app/app/
COPY rag_factory/ /app/rag_factory/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
