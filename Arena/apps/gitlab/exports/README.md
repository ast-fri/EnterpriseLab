# GitLab Project Exports

Place your GitLab project export files (`.tar.gz`) in this directory.

## How to Export Projects

### Option 1: GitLab UI

1. Go to your GitLab instance (http://localhost:8080)
2. Login as admin/root
3. Navigate to your project
4. Go to **Settings** → **General** → **Advanced**
5. Click **Export project**
6. Wait for export to complete
7. Download the `.tar.gz` file
8. Place it in this directory

### Option 2: GitLab API

```bash
# Get access token from GitLab UI first
# Then:

# List all projects
curl --header "PRIVATE-TOKEN: your-token" \
     http://localhost:8080/api/v4/projects

# Export a project (replace PROJECT_ID)
curl --request POST \
     --header "PRIVATE-TOKEN: your-token" \
     http://localhost:8080/api/v4/projects/PROJECT_ID/export

# Check export status
curl --header "PRIVATE-TOKEN: your-token" \
     http://localhost:8080/api/v4/projects/PROJECT_ID/export

# Download when ready
curl --header "PRIVATE-TOKEN: your-token" \
     --output "project-export.tar.gz" \
     http://localhost:8080/api/v4/projects/PROJECT_ID/export/download
```

### Option 3: Automated Script

```bash
# From Arena/apps/ directory
./export_data_from_running_containers.sh
```

## File Naming

Files can have any name, but descriptive names help:
- `my-awesome-project.tar.gz`
- `company-website.tar.gz`
- `backend-api.tar.gz`

## During Build

The `init.sh` script will:
1. Find all `.tar.gz` files in this directory
2. Import them into GitLab
3. Assign sequential project IDs (starting from 2)

## Example Structure

```
exports/
├── README.md              ← This file
├── project1.tar.gz        ← Will be imported as project ID 2
├── project2.tar.gz        ← Will be imported as project ID 3
└── my-cool-app.tar.gz     ← Will be imported as project ID 4
```

## Notes

- Project ID 1 is reserved for the Documentation project
- Imports happen during `docker build`, not `docker run`
- Each import can take 1-5 minutes depending on project size
- The build process will show import progress
