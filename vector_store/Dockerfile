FROM python:3.11-slim

WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir fastapi uvicorn chromadb sentence-transformers

EXPOSE 8000

CMD ["uvicorn", "vector_store.main:app", "--host", "0.0.0.0", "--port", "8000"]
