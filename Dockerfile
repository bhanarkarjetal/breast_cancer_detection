FROM python:3.11-slim

WORKDIR /app

# System dependencies 
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Environment variable
ENV PYTHONUNBUFFERED=1

# Python dependencies
RUN pip install --no-cache-dir --upgrade pip 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose port for Jupyter Notebook
EXPOSE 8000

# Default command
CMD ["/bin/bash"]