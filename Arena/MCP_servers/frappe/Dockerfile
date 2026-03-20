FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY frappe_mcp_server.py .

# Expose port
EXPOSE 12013

# Run the server
CMD ["python", "frappe_mcp_server.py"]