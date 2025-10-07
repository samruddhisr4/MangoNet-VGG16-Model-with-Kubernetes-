# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Install system dependencies required by TensorFlow (small set)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# NOTE: Place your `mango_classification_model.h5` in the project root (next to app.py)
# before building the image, or mount it at runtime. The default MODEL_PATH is
# 'mango_classification_model.h5'. If you want to use a different path, set the
# MODEL_PATH env var when running the container.

EXPOSE 5000

# Use a non-root user in production images (optional)
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Default environment variables
ENV PORT=5000
ENV MODEL_PATH=mango_classification_model.h5

# Start the app
CMD ["python", "app.py"]