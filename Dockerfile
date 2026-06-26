# CUDA 12.8 base (Blackwell / RTX 50-series). The torch cu128 wheels bundle
# their own CUDA runtime; the host needs a driver that supports CUDA 12.8.
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# --- Combined Installation Layer ---
# Install system dependencies and Miniforge in a single RUN command to reduce layers.
# This also cleans up the apt cache in the same layer to reduce image size.
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

# Add conda to the system's PATH and set up the shell
ENV PATH="/opt/conda/bin:${PATH}"

# Copy your environment file
COPY environment.yml .

# Update the base conda environment with your packages and clean up
RUN conda env update -n base --file environment.yml && \
    conda clean -y --all

# Permissions for cache/config
RUN mkdir -p /.cache /.config && \
    chmod -R 777 /.cache /.config