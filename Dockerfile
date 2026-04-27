FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Cloud Run sets PORT env var, default 8080
ENV PORT=8080
EXPOSE $PORT

# Run the app
CMD sh -c "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"
