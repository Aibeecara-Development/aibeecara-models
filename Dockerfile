# Dockerfile.grammar
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install fastapi happytransformer transfoermers pydantic

# Copy grammar server
COPY api.py .

# Expose port
EXPOSE 8001

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
