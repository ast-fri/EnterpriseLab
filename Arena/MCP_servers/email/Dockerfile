FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

COPY email_server.py .
COPY email_storage.py .


EXPOSE 12005

CMD ["python", "email_server.py"]