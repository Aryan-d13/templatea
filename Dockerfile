# Templatea Backend - Production Docker Image
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (ffmpeg for video processing)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1-mesa-glx \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY templates/ ./templates/
COPY template_engine.py ./
COPY orchestrator.py ./
COPY video_detector.py ./
COPY downloader_mapper.py ./
COPY instagram_downloader.py ./

# Create necessary directories
RUN mkdir -p /app/workspace /app/db /app/__video_assets

# Make entrypoint executable
COPY docker/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint and default command
ENTRYPOINT ["./entrypoint.sh"]
CMD ["uvicorn", "api.app:APP", "--host", "0.0.0.0", "--port", "8000"]
