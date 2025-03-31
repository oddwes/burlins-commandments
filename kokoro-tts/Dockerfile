FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libsndfile1 \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install Python dependencies in a specific order to handle dependencies better
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    torch --index-url https://download.pytorch.org/whl/cpu \
    torchaudio --index-url https://download.pytorch.org/whl/cpu \
    soundfile \
    huggingface_hub \
    python-dotenv \
    gradio==3.50.2 \
    matplotlib \
    pydub

# Install kokoro last, as it has complex dependencies
RUN pip install --no-cache-dir kokoro>=0.9.2

# Copy your local files
COPY . .

# Download model weights - this will use the token from .env
RUN python download_weights.py

# Expose the port Gradio will run on
EXPOSE 7860

# Start the application
CMD ["python", "app.py"]