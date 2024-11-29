# Use Python 3.10 as the base image
FROM python:3.10-slim

# Set environment variables to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary system packages for OpenCV and other dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    xvfb

# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     libqt5gui5 \
#     libqt5core5a \
#     libqt5dbus5 \
#     qtbase5-dev \
#     qtchooser \
#     qt5-qmake \
#     qtbase5-dev-tools \
#     && rm -rf /var/lib/apt/lists/*

# Create a working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose any port if needed (optional, adjust if your code listens on a port)
EXPOSE 5000

# Run the Python script when the container launches
CMD ["python", "Live_feed.py"]



