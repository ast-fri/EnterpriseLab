FROM owncloud/server:latest

# Set environment variables
ENV OWNCLOUD_DOMAIN="localhost:8081"
ENV OWNCLOUD_TRUSTED_DOMAINS="localhost,localhost:8081,136.111.0.30"
ENV ADMIN_USERNAME="admin"
ENV ADMIN_PASSWORD="admin"
ENV OWNCLOUD_DB_TYPE="mysql"
ENV OWNCLOUD_DB_NAME="owncloud"
ENV OWNCLOUD_DB_USERNAME="owncloud"
ENV OWNCLOUD_DB_PASSWORD="secret"
ENV OWNCLOUD_DB_HOST="db"

# Copy pre-filled data
COPY owncloud_data/files /mnt/data/prefill_files

# Copy initialization script
COPY init.sh /usr/local/bin/init.sh
RUN chmod +x /usr/local/bin/init.sh

# Run initialization during build
RUN sh /usr/local/bin/init.sh
