FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY owncloud_mcp_server.py .

ENV PORT=12001
ENV OWNCLOUD_URL=http://host.docker.internal:8081
ENV OWNCLOUD_USERNAME=admin
ENV OWNCLOUD_PASSWORD=admin

EXPOSE 12001

CMD ["python", "owncloud_mcp_server.py"]
