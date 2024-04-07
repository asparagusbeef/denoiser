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
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime Stage
FROM nvidia/cuda:11.8.0-base-ubuntu22.04 as runtime-stage

# Set environment variables (optional)
ENV PYTHONUNBUFFERED=1
ENV LOGGING_LEVEL=INFO

# Install Python
RUN apt-get update -y && \
    apt-get install -y python3.11 python3-distutils --no-install-recommends && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the virtual environment from the build stage
COPY --from=build-stage /opt/venv /opt/venv

# Activate virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy the entire project directory contents into the container at /app
COPY . .

# Run your application
CMD ["sh", "-c", "gunicorn -w 2 -k uvicorn.workers.UvicornWorker --log-level debug --access-logfile - --error-logfile - -b '[::]:8888' --timeout 600 main:app"]