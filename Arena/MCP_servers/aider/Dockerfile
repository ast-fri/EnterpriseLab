FROM python:3.11-slim

WORKDIR /app

# Install git (required by aider)
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY aider_mcp_server.py .
COPY .env .

# Expose port
EXPOSE 12011

# Run the server
CMD ["python", "aider_mcp_server.py"]