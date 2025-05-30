
# Use an official Python runtime as a parent image
FROM python:3.10-bullseye AS builder

# Upgrade pip to avoid issues with older versions of pip
RUN pip install --upgrade pip

# Set working directory (Use SageMaker's convention for code location)
WORKDIR /app

# Install system-level dependencies for building packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file (create one with all necessary Python packages)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

#ENV CUDA_VISIBLE_DEVICES=-1  
# Disables GPU to prevent memory issues

#RUN python -c "from insightface.app import FaceAnalysis; \
 #              app = FaceAnalysis(providers=['CPUExecutionProvider']); \
#		 app.prepare(ctx_id=0, det_size=(640, 640))"


# Copy your inference code into the container
COPY . /app


# Command to run the inference script (ensure correct path)
ENTRYPOINT ["python", "/app/main.py"]

