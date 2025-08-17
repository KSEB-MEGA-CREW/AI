# Stage 1: Base Image - Using NVIDIA CUDA for GPU support
# This ensures the container has the necessary drivers and libraries for PyTorch with GPU.
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Set environment variables to prevent interactive prompts during installations
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Install system dependencies required for Miniconda and other packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl bzip2 && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda in the /opt/conda directory
ENV CONDA_DIR /opt/conda
RUN curl -sL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -o miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    $CONDA_DIR/bin/conda init bash

# Add Conda to the system's PATH environment variable
ENV PATH=$CONDA_DIR/bin:$PATH

# Set the working directory for the application
WORKDIR /app

# Copy the Conda environment file into the container
COPY environment.yml .

# Create the Conda environment from the environment.yml file.
# This command sets up the 'KSEB' environment with all specified conda and pip packages.
# This step can take a significant amount of time due to the large number of dependencies.
RUN conda env create -f environment.yml

# Copy the rest of the application source code into the container
COPY . .

# Expose the port the Flask application runs on
EXPOSE 1958

# Define the default command to run when the container starts.
# It activates the 'KSEB' conda environment and then executes the Flask app.
# Using --no-capture-output to ensure logs are streamed correctly.
CMD ["/bin/bash", "-c", "conda run --no-capture-output -n KSEB python app.py"]
