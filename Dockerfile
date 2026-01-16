FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (needed for some python packages like psycopg2)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create cache directory for Pathway
RUN mkdir -p /app/cache && chmod 777 /app/cache

# Expose the API port (Render uses PORT env var)
EXPOSE 8000

# Health check for Render
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Command to run the application
# Use exec form for proper signal handling
CMD ["python", "app.py"]
