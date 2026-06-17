#!/bin/bash
set -e

# Run the original wrapper script, but remove the last 2 lines
# This prevents the script from waiting indefinitely
head -n -2 /assets/wrapper > /tmp/modified_wrapper
source /tmp/modified_wrapper

echo "GitLab is up and running. Performing post-launch actions..."

# Function to check GitLab import status
check_import_status() {
    local id=$1
    local status=$(curl --silent --header "PRIVATE-TOKEN: admin-token" \
                        "http://0.0.0.0/api/v4/projects/${id}/import" |
                        jq -r '.import_status')
    echo $status
}

# Create admin token with necessary permissions
gitlab-rails runner "token = User.find_by_username('root').personal_access_tokens\
    .create(scopes: ['api', 'read_user', 'read_api', 'read_repository', 'write_repository', 'sudo', 'admin_mode'], name: 'admin-token', expires_at: 365.days.from_now); \
    token.set_token('admin-token'); \
    token.save!"

# Enable import from GitLab exports and project export
curl --request PUT --header "PRIVATE-TOKEN: admin-token" \
    "http://0.0.0.0/api/v4/application/settings?import_sources=gitlab_project&project_export_enabled=true"

# Create a documentation project (ID will be 1)
curl --request POST --header "PRIVATE-TOKEN: admin-token" \
     --header "Content-Type: application/json" --data '{
        "name": "Documentation", "description": "Company-wide documentation", "path": "docs",
        "wiki_access_level": "enabled", "with_issues_enabled": "false",
        "with_merge_requests_enabled": "false",
        "visibility": "public"}' \
     --url "http://0.0.0.0/api/v4/projects/"

# Add README to documentation project
curl --request POST --header "PRIVATE-TOKEN: admin-token" \
     --header "Content-Type: application/json" --data '{
        "branch": "main", "author_email": "admin@enterpriselab.local", "author_name": "Administrator",
        "content": "# Enterprise Lab Documentation\n\nWelcome to the Enterprise Lab documentation hub. Please navigate to [wiki](../../wikis) for all documentation.",
        "commit_message": "Add README"}' \
     --url "http://0.0.0.0/api/v4/projects/1/repository/files/README.md"

# Import projects from exports directory
if ls /assets/exports/*.tar.gz 1> /dev/null 2>&1; then
    project_id=2
    for file in $(ls /assets/exports/*.tar.gz); do
        filename=$(basename "$file" .tar.gz)

        echo "Importing $filename, project_id=$project_id"
        curl --request POST \
             --header "PRIVATE-TOKEN: admin-token" \
             --form "path=$filename" \
             --form "file=@$file" \
             "http://0.0.0.0/api/v4/projects/import"

        echo "Waiting for import to complete..."
        while true; do
            status=$(check_import_status $project_id)
            if [ "$status" == "started" ] || [ "$status" == "scheduled" ]; then
                echo "Project $project_id import status: $status"
            elif [ "$status" != "finished" ]; then
                echo "Error: Unexpected status for project $project_id: $status"
                exit 1
            else
                echo "Project $project_id import succeeded: $status"
                sleep 10
                break
            fi
            sleep 30
        done

        ((project_id++))
    done
else
    echo "No .tar.gz files found in /assets/exports/. Nothing to import."
fi

echo "Finished importing all repos"

# Import wiki pages
if ls /assets/wikis/*.md 1> /dev/null 2>&1; then
    for file in $(ls /assets/wikis/*.md); do
        title=$(basename "$file" .md)
        content=$(cat "$file" | jq -sRr @uri)

        curl --data "title=$title&content=$content" \
             --header "PRIVATE-TOKEN: admin-token" \
             "http://0.0.0.0/api/v4/projects/1/wikis"

        echo "Uploaded wiki: $title"
    done
else
    echo "No .md files found in /assets/wikis/. Nothing to import."
fi

echo "GitLab initialization complete!"
