FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

COPY calendar_server.py .
COPY calendar_storage.py .

EXPOSE 12006

CMD ["python", "calendar_server.py"]