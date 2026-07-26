# Base image: NVIDIA PyTorch container optimized for DGX / CUDA acceleration
FROM nvcr.io/nvidia/pytorch:25.10-py3

# Set working directory inside the container
WORKDIR /workspace

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install debugpy for remote debugging support
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir debugpy

# Default command to launch an interactive bash shell
CMD ["/bin/bash"]
