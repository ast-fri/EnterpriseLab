#!/bin/sh
set -e

echo "Waiting for MongoDB to be ready..."

# Wait for MongoDB to be ready
max_attempts=60
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if docker exec rocket-chat-mongodb-1 mongosh --eval "db.adminCommand('ping')" >/dev/null 2>&1; then
        echo "MongoDB is ready!"
        break
    fi
    attempt=$((attempt + 1))
    echo "Attempt $attempt/$max_attempts - MongoDB not ready yet..."
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "ERROR: MongoDB did not become ready in time"
    exit 1
fi

echo "Restoring RocketChat database..."

# Restore the database dump
docker exec -i rocket-chat-mongodb-1 mongorestore --archive < /data/rocketchat.dump

echo "Database restored successfully!"
