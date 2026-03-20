# 🛠️ Self-Hosting GitLab with Docker

This guide explains how to deploy a self-hosted GitLab instance using Docker. It ensures persistent storage and supports access through a web browser via SSH tunneling.

---

## 🚀 Run GitLab Container

Run the following command on your server to start the GitLab instance:

docker compose up -d




📁 Replace /path/to/gitlab/config, /logs, and /data with appropriate absolute paths on your server to enable data persistence.

Accessing GitLab in a Web Browser (Optional)
If you're working on a remote machine, you can access the GitLab web interface using SSH port forwarding:
ssh -L 8080:localhost:8080 your_user@your_server_ip
Then open your browser and navigate to:
http://localhost:8080

ADMINISTRATOR setup option comes with new installation only.