# Use Python base image
FROM python:3.10-slim

# Install system dependencies for Tkinter GUI and plotting
RUN apt-get update && \
    apt-get install -y python3-tk libx11-6 libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy everything into the container
COPY . /app

# Install Python packages
RUN pip install --upgrade pip
RUN pip install -r tkinter_requirements.txt

# Set up display environment for GUI apps (optional for headless runs)
ENV DISPLAY=:0

# Use module mode to ensure relative imports work
CMD ["python", "-m", "test.ResearchProject"]
