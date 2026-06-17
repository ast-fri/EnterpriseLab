FROM gitlab/gitlab-ce:latest

# Install necessary tools
RUN apt-get update && apt-get install -y jq curl && rm -rf /var/lib/apt/lists/*

# Set GitLab configuration
ENV GITLAB_SKIP_TAIL_LOGS=true
ENV GITLAB_ROOT_EMAIL="admin@enterpriselab.local"
ENV GITLAB_ROOT_PASSWORD="admin"

# Copy initialization script and data to the image
COPY init.sh /assets/init.sh
COPY exports /assets/exports
COPY wikis /assets/wikis

# Make script executable and run initialization
RUN chmod +x /assets/init.sh
RUN bash /assets/init.sh

# Default CMD from gitlab-ce image
CMD ["/assets/wrapper"]
