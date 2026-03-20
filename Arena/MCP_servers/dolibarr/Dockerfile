FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dolibarr_mcp_server.py .

EXPOSE 12000

CMD ["python", "dolibarr_mcp_server.py"]