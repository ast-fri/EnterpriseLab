# Database Tables Analysis - Current vs Available

This document compares what we're currently extracting vs all available tables in each application.

## 1. RocketChat (MongoDB)

### Currently Extracting:
- `users` (username, name, emails, roles, status)
- `rocketchat_room` (name, type, usernames, msgs)
- `rocketchat_message` (msg, ts, u.username, rid)

### Available Collections (~85 total):
Key collections we're **MISSING**:
- `rocketchat_subscription` - User subscriptions to rooms
- `rocketchat_settings` - System settings
- `rocketchat_permissions` - Permission definitions
- `rocketchat_roles` - Role definitions
- `rocketchat_custom_user_status` - Custom statuses
- `rocketchat_uploads` - File uploads metadata
- `rocketchat_integrations` - Webhooks & integrations
- `rocketchat_livechat_department` - Omnichannel departments
- `rocketchat_livechat_visitor` - Omnichannel visitors
- `rocketchat_oauth_apps` - OAuth applications
- `rocketchat_custom_emoji` - Custom emojis
- `rocketchat_import` - Import history

**Recommendation**: Add subscriptions, settings, roles, and uploads

---

## 2. Plane (PostgreSQL)

### Currently Extracting:
- `workspaces` (id, name, slug, created_at)
- `projects` (id, name, description, workspace_id)
- `issues` (id, name, state_id, project_id)

### Available Tables (~120+ total):
Key tables we're **MISSING**:
- `accounts` - User accounts
- `cycles` - Sprint/cycle management
- `cycle_issues` - Issues in cycles
- `modules` - Project modules
- `issue_assignees` - Issue assignments
- `issue_activities` - Issue activity log
- `comment_reactions` - Comments & reactions
- `estimates` / `estimate_points` - Story point estimates
- `integrations` - Third-party integrations
- `github_repositories` / `github_issue_syncs` - GitHub sync
- `file_assets` - Uploaded files
- `api_tokens` - API access tokens
- `analytic_views` - Analytics data
- `exporters` / `importers` - Import/export history

**Recommendation**: Add cycles, modules, assignees, activities, and estimates

---

## 3. Dolibarr (MariaDB)

### Currently Extracting:
- `llx_user` (login, lastname, firstname, email, statut)
- `llx_product` (ref, label, price, fk_product_type)
- `llx_socpeople` (lastname, firstname, email, fk_soc)

### Available Tables (~500+ total):
Key tables we're **MISSING**:
- `llx_societe` - Companies/customers
- `llx_facture` - Invoices
- `llx_facture_fourn` - Supplier invoices
- `llx_commande` - Customer orders
- `llx_commande_fournisseur` - Supplier orders
- `llx_propal` - Commercial proposals/quotes
- `llx_contrat` - Contracts
- `llx_projet` - Projects
- `llx_projet_task` - Project tasks
- `llx_actioncomm` - Events/actions/calendar
- `llx_bank_account` - Bank accounts
- `llx_bank` - Bank transactions
- `llx_stock_mouvement` - Stock movements
- `llx_entrepot` - Warehouses
- `llx_adherent` - Members/subscriptions
- `llx_ticket` - Support tickets
- `llx_emailcollector_emailcollector` - Email collectors

**Recommendation**: Add companies, invoices, orders, proposals, projects, and bank accounts

---

## 4. Frappe HRMS (MariaDB)

### Currently Extracting:
- `tabUser` (name, full_name, email, enabled)
- `tabEmployee` (name, employee_name, designation, status)
- `tabCompany` (name, abbr, country, default_currency)
- `tabDepartment` (name, parent_department, company)
- `tabAttendance` (employee, attendance_date, status)
- `tabLeave Application` (employee, from_date, to_date, status)

### Available Tables (~1,500+ DocTypes):
Key tables we're **MISSING**:
- `tabSalary Structure` - Salary structures
- `tabSalary Slip` - Generated salary slips
- `tabPayroll Entry` - Payroll processing
- `tabEmployee Checkin` - Attendance check-ins
- `tabShift Type` / `tabShift Assignment` - Shift management
- `tabHoliday List` / `tabHoliday` - Holiday calendars
- `tabLeave Type` / `tabLeave Policy` - Leave types & policies
- `tabEmployee Grade` - Employee grades/levels
- `tabDesignation` - Job designations
- `tabBranch` - Office branches
- `tabAppraisal` / `tabAppraisal Template` - Performance appraisals
- `tabTraining Event` / `tabTraining Program` - Training management
- `tabExpense Claim` - Employee expense claims
- `tabEmployee Onboarding` / `tabEmployee Separation` - HR workflows
- `tabEmployee Skill Map` - Skills & competencies

**Recommendation**: Add salary structures, shifts, holidays, leave policies, and appraisals

---

## 5. OwnCloud (MariaDB)

### Currently Extracting:
- `oc_users` (uid, displayname)
- `oc_filecache` (name, path, size, mtime)

### Available Tables (~50 total):
Key tables we're **MISSING**:
- `oc_share` - File shares (internal & external)
- `oc_share_external` - Federated shares
- `oc_accounts` - Extended user account info
- `oc_group_user` / `oc_groups` - Groups & memberships
- `oc_activity` - Activity feed/logs
- `oc_comments` - File comments
- `oc_files_trash` - Deleted files (trash bin)
- `oc_storages` - Storage backends
- `oc_mounts` - External mounts
- `oc_external_mounts` / `oc_external_config` - External storage configs
- `oc_dav_shares` - WebDAV shares
- `oc_systemtag` / `oc_systemtag_object_mapping` - Tags on files
- `oc_notifications` - User notifications
- `oc_jobs` - Background jobs
- `oc_file_locks` - File locking

**Recommendation**: Add shares, groups, activity, comments, and trash

---

## 6. Zammad (PostgreSQL)

### Currently Extracting:
- `tickets` (id, title, state_id, priority_id, group_id, customer_id)
- `users` (id, login, firstname, lastname, email, active)
- `organizations` (id, name, active)

### Available Tables (~100+ total):
Key tables we're **MISSING**:
- `ticket_articles` - Ticket messages/replies
- `ticket_time_accountings` - Time tracking
- `groups` - Agent groups
- `groups_users` - Group memberships
- `roles` / `roles_users` - User roles & permissions
- `overviews` - Ticket views/filters
- `channels` - Communication channels (email, chat, etc.)
- `email_addresses` - System email addresses
- `calendars` - SLA calendars
- `tags` / `tag_items` / `tag_objects` - Tagging system
- `knowledge_bases` / `knowledge_base_answers` - KB articles
- `chat_sessions` / `chat_messages` - Chat transcripts
- `cti_logs` / `cti_caller_ids` - Phone integration
- `activity_streams` - Activity feed
- `macros` - Ticket macros/templates
- `text_modules` - Text templates
- `triggers` / `schedulers` / `jobs` - Automation

**Recommendation**: Add ticket_articles, groups, roles, tags, and KB articles

---

## 7. GitLab (PostgreSQL)

### Currently Extracting:
- **NONE** (container was restarting, now fixed to v18.11.5)

### Expected Tables (~500+ total):
Key tables to extract:
- `users` - User accounts
- `projects` - Git repositories/projects
- `namespaces` - Groups & user namespaces
- `members` - Project/group memberships
- `issues` - Issue tracker
- `merge_requests` - Merge/pull requests
- `ci_pipelines` / `ci_builds` - CI/CD pipelines
- `notes` - Comments on issues/MRs
- `labels` / `label_links` - Labels & tagging
- `milestones` - Project milestones
- `snippets` - Code snippets
- `ci_runners` - CI runners
- `deployments` - Deployment history
- `environments` - Deployment environments
- `protected_branches` - Branch protection rules
- `project_features` - Feature flags per project

**Recommendation**: Build comprehensive GitLab extraction once container is stable

---

## Summary: Priority Enhancements

### High Priority (Core functionality):
1. **RocketChat**: Add subscriptions, settings, roles
2. **Plane**: Add cycles, modules, assignees, activities
3. **Dolibarr**: Add companies, invoices, orders, proposals
4. **Frappe**: Add salary structures, shifts, holidays
5. **OwnCloud**: Add shares, groups, activity, comments
6. **Zammad**: Add ticket_articles, groups, roles, tags
7. **GitLab**: Build complete extraction (once stable)

### Medium Priority (Extended features):
- File uploads/attachments for all apps
- Integration configurations
- API tokens/webhooks
- Activity logs/audit trails

### Low Priority (System internals):
- Background jobs
- Migration history
- System settings
- Internal metrics

---

**Next Steps:**
1. Update `fetch_database_data.sh` to include high-priority tables
2. Test extraction with sample data
3. Update `push_data.sh` to handle new tables
4. Document field mappings in detail
