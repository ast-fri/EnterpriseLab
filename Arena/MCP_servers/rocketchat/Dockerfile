FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rocketchat_mcp_server.py .

EXPOSE 12004

CMD ["python", "rocketchat_mcp_server.py"]