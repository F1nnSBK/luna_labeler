# Use a lightweight python image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with UID 1000
RUN useradd -m -u 1000 user

# Set env variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR $HOME/app

# Copy requirements file first to leverage Docker cache
COPY --chown=user:user requirements.txt .

# Install dependencies as the non-root user
USER user
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application files
COPY --chown=user:user . .

# Expose Hugging Face Space port
EXPOSE 7860

# Run uvicorn server on port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]