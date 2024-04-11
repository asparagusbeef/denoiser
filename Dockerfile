# Stage 1: Build Stage
FROM nvidia/cuda:11.8.0-base-ubuntu22.04 as build-stage

# Set environment variables (optional)
ENV PYTHONUNBUFFERED=1

# Install Python, Pip, Development Headers, and Virtual Environment
RUN apt-get update -y && \
    apt-get install -y python3.11 python3-pip python3.11-dev python3.11-venv build-essential python3-distutils --no-install-recommends && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# Set up a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the requirements.txt into the container
COPY requirements.txt .

# Install PyTorch compatible with CUDA 11.8
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime Stage
FROM nvidia/cuda:11.8.0-base-ubuntu22.04 as runtime-stage

# Set environment variables (optional)
ENV PYTHONUNBUFFERED=1
ENV LOGGING_LEVEL=INFO

# Install Python
RUN apt-get update -y && \
    apt-get install -y python3.11 python3-distutils --no-install-recommends && \
    apt-get install -y ffmpeg git && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \ 
    git init

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the virtual environment from the build stage
COPY --from=build-stage /opt/venv /opt/venv

# Activate virtual environment
ENV PATH="/opt/venv/bin:$PATH"

COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Run your application
CMD ["uvicorn", "main:app", "--host", "::", "--port", "8888", "--log-level", "debug"]