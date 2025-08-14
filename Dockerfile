# Autonomous Drone Navigation System Docker Container
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libfontconfig1-dev \
    libcairo2-dev \
    libgdk-pixbuf2.0-dev \
    libpango1.0-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libqtgui4 \
    libqtwebkit4 \
    libqt4-test \
    python3-pyqt5 \
    python3-opencv \
    alsa-utils \
    portaudio19-dev \
    espeak \
    espeak-data \
    libespeak1 \
    libespeak-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 droneuser && \
    chown -R droneuser:droneuser /app
USER droneuser

# Copy requirements first for better caching
COPY --chown=droneuser:droneuser requirements.txt .

# Install Python dependencies
RUN python3 -m pip install --user --upgrade pip && \
    python3 -m pip install --user --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=droneuser:droneuser . .

# Create necessary directories
RUN mkdir -p logs config models missions data && \
    chmod +x scripts/*.py

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH
ENV PATH=/home/droneuser/.local/bin:$PATH

# Expose ports for web interface and communication
EXPOSE 8080 14550 14551

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import cv2, numpy, torch; print('Dependencies OK')" || exit 1

# Default command
CMD ["python3", "main.py"]

# Multi-stage build for production
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as production

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-opencv \
    libportaudio2 \
    espeak \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create user and working directory
RUN useradd -m -u 1000 droneuser
WORKDIR /app
USER droneuser

# Copy from builder stage
COPY --from=0 --chown=droneuser:droneuser /home/droneuser/.local /home/droneuser/.local
COPY --chown=droneuser:droneuser . .

ENV PATH=/home/droneuser/.local/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH

EXPOSE 8080 14550 14551
CMD ["python3", "main.py"]
