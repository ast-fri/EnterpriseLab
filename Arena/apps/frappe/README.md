# Seeded Frappe HRMS

This stack creates a fresh Frappe/ERPNext/HRMS v15 site, imports validated
records from `Extracted_data/frappe_from_db.json` through Frappe's document API,
and creates reusable project/task fixtures.

From the repository root:

```bash
./frappe/seed/seed.sh \
  Extracted_data/frappe_from_db.json \
  frappe/docker-compose.yml
```

The UI is at <http://localhost:8084>. All credentials are stored in
`user-credentials.json`. The 12 non-admin accounts have HR,
Projects, Accounts, Sales, Purchase and Stock user access; `admin` is the only
administrator.

The baseline keeps the compatible exported company, employees, attendance,
customers, suppliers, items and warehouses. It adds 3 projects, 24 tasks and 24
assignments so workflows can create, update, complete and reassign tasks.

The first reset downloads and builds Frappe, ERPNext and HRMS v15 and may take
several minutes. Later ordinary starts reuse the installed apps and seeded data.

Ordinary starts preserve the seeded site:

```bash
docker compose -p frappe-seeded \
  -f frappe/docker-compose.yml up -d
```

Run `seed.sh` again to delete only the `frappe-seeded` volumes and reconstruct
the baseline.
