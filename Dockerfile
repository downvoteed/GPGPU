# Use the NVIDIA CUDA devel image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set the timezone in the environment
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

# Make sure we have the latest package lists
RUN apt-get update

# Install required packages
RUN apt-get install -y build-essential libboost-all-dev libopencv-dev cmake

# Copy the current directory contents into the Docker image
COPY . /app

# Set the working directory to /app
WORKDIR /app