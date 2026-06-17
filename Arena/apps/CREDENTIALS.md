# 🔐 Service Credentials

Default login credentials for all services in EnterpriseLab.

---

## Services with Pre-configured Credentials

### GitLab
- **URL:** http://localhost:8080
- **Username:** `root`
- **Password:** Check your GitLab configuration (default from backup)
- **Note:** First-time login may require password setup

### OwnCloud
- **URL:** http://localhost:8081
- **Username:** `admin`
- **Password:** `admin`
- **Database:** SQLite (no separate DB credentials)

### Dolibarr
- **URL:** http://localhost:8082
- **Username:** `admin`
- **Password:** `admin`
- **Database:** MariaDB (managed by docker-compose)

---

## Services with Backup-Restored Credentials

These services restore data from backups, so they use whatever credentials were in the original data:

### RocketChat
- **URL:** http://localhost:3000
- **Admin Username:** `suraj.nagaje`
- **Admin Email:** `meetsurajn12@gmail.com`
- **Password:** *Your configured password*

**Other Users:**
- `raj.patel` (raj.patel@inazuma.com)
- `rahul.khanna` (rahul.khanna@inazuma.com)
- `username` (generic test user)

**If you forgot the password:**

Reset via database:
```bash
# Reset admin password to "admin123"
docker exec rocket-chat-mongodb-1 mongosh rocketchat --quiet --eval '
db.users.updateOne(
  {username: "suraj.nagaje"},
  {$set: {
    "services.password.bcrypt": "$2b$10$n9CM8OgInDlwpvjLKLPML.eizXIzLlRtgCh3GRLafOdR9ldAUh/KG"
  }}
);
print("Password reset to: admin123");
'

# Restart RocketChat
docker restart rocket-chat-rocketchat-1
```

### Plane
- **URL:** http://localhost:3001
- **Credentials:** Restored from backup
- **Database:** PostgreSQL (managed by docker-compose)
- **Note:** Create account on first visit or use restored accounts

---

## For Maintainers: Updating Credentials

When you update the pre-filled data:

### Option 1: Keep Same Credentials
- Just use `./backup_volumes.sh`
- Credentials are preserved in backups

### Option 2: Create Test Credentials
Before backing up, set known credentials:

**RocketChat:**
```bash
# Create a test admin user
docker exec -it rocket-chat-rocketchat-1 mongo rocketchat --eval '
db.users.insert({
  username: "admin",
  emails: [{address: "admin@example.com", verified: true}],
  name: "Admin User",
  roles: ["admin"],
  services: {
    password: {
      bcrypt: "$2b$10$n9CM8OgInDlwpvjLKLPML.eizXIzLlRtgCh3GRLafOdR9ldAUh/KG"
    }
  }
});
'
# Password: admin123
```

**OwnCloud:**
Already has admin/admin

**GitLab:**
```bash
docker exec -it gitlab gitlab-rake "gitlab:password:reset[root]"
# Follow prompts to set password
```

**Dolibarr:**
Already has admin/admin

---

## Database Credentials (Docker Internal)

These are used between containers, not for user login:

### OwnCloud Database (MariaDB)
- Host: `db`
- Database: `owncloud`
- User: `owncloud`
- Password: `secret`

### Dolibarr Database (MariaDB)
- Host: `mariadb`
- Database: `dolibarr`
- User: `dolibarr`
- Password: `dolibarr`

### Plane Database (PostgreSQL)
- Host: `plane-db`
- Database: `plane`
- User: `plane`
- Password: `plane`

### RocketChat Database (MongoDB)
- Host: `mongodb`
- Database: `rocketchat`
- User: (none - no authentication)
- ReplicaSet: `rs0`

---

## Security Notes

### For Development/Testing
These default credentials are fine for local development and testing environments.

### For Production
**IMPORTANT:** If deploying to production:

1. ❗ **Change ALL default passwords**
2. ❗ **Use strong, unique passwords**
3. ❗ **Enable HTTPS/TLS**
4. ❗ **Configure proper firewalls**
5. ❗ **Enable database authentication**
6. ❗ **Use secrets management (Docker secrets, Vault, etc.)**
7. ❗ **Regular security updates**

### Changing Passwords in Production

**OwnCloud:**
```bash
docker exec owncloud-owncloud-1 owncloud user:resetpassword admin
```

**Dolibarr:**
Via web interface: Admin → Users → Edit user → Change password

**GitLab:**
```bash
docker exec gitlab gitlab-rake "gitlab:password:reset[root]"
```

**RocketChat:**
Via web interface: Administration → Users → Edit user → Change password

**Plane:**
Via web interface: Settings → Account → Change password

---

## Quick Reference Table

| Service | URL | Username | Password | Notes |
|---------|-----|----------|----------|-------|
| **GitLab** | :8080 | root | *varies* | Check config |
| **OwnCloud** | :8081 | admin | admin | Fixed |
| **RocketChat** | :3000 | suraj.nagaje | *varies* | From backup |
| **Plane** | :3001 | *varies* | *varies* | From backup |
| **Dolibarr** | :8082 | admin | admin | Fixed |

---

## Troubleshooting Login Issues

### Can't login to RocketChat
1. Check if service is running: `docker ps | grep rocket`
2. Check logs: `docker logs rocket-chat-rocketchat-1`
3. Reset password using the command above
4. Restart: `docker restart rocket-chat-rocketchat-1`

### Can't login to GitLab
1. Wait 2-3 minutes after start (GitLab takes time)
2. Check health: `docker ps | grep gitlab`
3. Reset password: `docker exec gitlab gitlab-rake "gitlab:password:reset[root]"`

### Can't login to OwnCloud
1. Verify URL: http://localhost:8081 (not 3001 or 8080)
2. Use exactly: `admin` / `admin`
3. Check logs: `docker logs owncloud-owncloud-1`

### Can't login to Plane
1. Check if all Plane containers are running: `docker ps | grep plane`
2. Try creating a new account
3. Check PostgreSQL is running: `docker ps | grep plane-db`

### Can't login to Dolibarr
1. Verify URL: http://localhost:8082
2. Use exactly: `admin` / `admin`
3. Check logs: `docker logs dolibarr-web-1`

---

## Need Help?

If you're testing and need to reset everything:

```bash
# Stop all services
cd Arena/apps
./start_all_servers.sh stop

# Remove volumes (⚠️ deletes data!)
docker volume prune -f

# Remove restore marker
rm .volumes_restored

# Start fresh with backups
./start_all_servers.sh
```

This will restore all data from backups with original credentials!

---

**Last Updated:** June 17, 2026  
**Maintained By:** EnterpriseLab Team
