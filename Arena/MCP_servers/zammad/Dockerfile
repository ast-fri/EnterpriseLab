FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY zammad_mcp_server.py .

# Expose port
EXPOSE 12010

# Run the server
CMD ["python", "zammad_mcp_server.py"]