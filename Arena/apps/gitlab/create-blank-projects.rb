# frozen_string_literal: true

admin = User.find_by!(username: 'aarav.mittal')
organization_id = admin.organization_id || 1

projects = [
  {
    group: 'platform-engineering', path: 'billing-gateway', name: 'billing-gateway',
    description: 'Blank seeded repository for enterprise billing gateway work.',
    members: { 'aarav.mittal' => 40, 'surya.reddy' => 40, 'raj.patel' => 30,
               'rahul.khanna' => 30, 'karan.sharma' => 30 }
  },
  {
    group: 'platform-engineering', path: 'document-collaboration-hub', name: 'document-collaboration-hub',
    description: 'Blank seeded repository for document collaboration work.',
    members: { 'aarav.mittal' => 40, 'raj.patel' => 40, 'priya.arora' => 30 }
  },
  {
    group: 'data-finance', path: 'revenue-intelligence-dashboard', name: 'revenue-intelligence-dashboard',
    description: 'Blank seeded repository for revenue intelligence work.',
    members: { 'raj.patel' => 40, 'sameer.malhotra' => 30, 'ethan.reynolds' => 30 }
  },
  {
    group: 'customer-operations', path: 'customer-onboarding-workspace', name: 'customer-onboarding-workspace',
    description: 'Blank seeded repository for customer onboarding work.',
    members: { 'surya.reddy' => 40, 'sameer.malhotra' => 30, 'ethan.reynolds' => 30 }
  },
  {
    group: 'field-service', path: 'field-service-control-tower', name: 'field-service-control-tower',
    description: 'Blank seeded repository for field service control work.',
    members: { 'ethan.reynolds' => 40, 'surya.reddy' => 30, 'sameer.malhotra' => 30 }
  },
  {
    group: 'support-engineering', path: 'support-operations-portal', name: 'support-operations-portal',
    description: 'Blank seeded repository for support operations work.',
    members: { 'surya.reddy' => 40, 'sameer.malhotra' => 40, 'rahul.khanna' => 30 }
  }
].freeze

subgroups = {
  'platform-engineering' => 'Platform Engineering',
  'data-finance' => 'Data and Finance',
  'customer-operations' => 'Customer Operations',
  'field-service' => 'Field Service',
  'support-engineering' => 'Support Engineering',
  'design-systems' => 'Design Systems'
}.freeze

def routed_namespace(path)
  Route.find_by(path: path, source_type: 'Namespace')&.source
end

root_group = routed_namespace('inazuma')

unless root_group
  # Preserve the database-only import for diagnostics, but move its invalid
  # route-less records out of the user-facing namespace before creating clean
  # GitLab-managed groups and repositories.
  Project.where(id: 301..306).find_each do |project|
    ProjectMember.where(source_type: 'Project', source_id: project.id).delete_all
    project.project_namespace&.update_columns(
      path: "legacy-#{project.path}-#{project.id}",
      name: "Legacy imported #{project.name}"
    )
    project.update_columns(
      path: "legacy-#{project.path}-#{project.id}",
      name: "Legacy imported #{project.name}",
      archived: true,
      hidden: true
    )
  end

  Namespace.where(id: 101..107).order(id: :desc).each do |namespace|
    namespace.update_columns(
      path: "legacy-#{namespace.path}-#{namespace.id}",
      name: "Legacy imported #{namespace.name}"
    )
  end

  response = Groups::CreateService.new(
    admin,
    name: 'Inazuma',
    path: 'inazuma',
    visibility_level: Gitlab::VisibilityLevel::PUBLIC,
    organization_id: organization_id
  ).execute
  raise "Unable to create Inazuma group: #{response.message}" unless response.success?

  root_group = response.payload.fetch(:group)
end

groups = {}
subgroups.each do |path, name|
  full_path = "inazuma/#{path}"
  group = routed_namespace(full_path)
  unless group
    response = Groups::CreateService.new(
      admin,
      name: name,
      path: path,
      parent_id: root_group.id,
      visibility_level: Gitlab::VisibilityLevel::PRIVATE,
      organization_id: organization_id
    ).execute
    raise "Unable to create #{full_path}: #{response.message}" unless response.success?

    group = response.payload.fetch(:group)
  end
  groups[path] = group
end

created_projects = []
projects.each do |definition|
  group = groups.fetch(definition.fetch(:group))
  full_path = "#{group.full_path}/#{definition.fetch(:path)}"
  project = Project.find_by_full_path(full_path)

  unless project&.route && project.repository_exists?
    project = Projects::CreateService.new(
      admin,
      name: definition.fetch(:name),
      path: definition.fetch(:path),
      description: definition.fetch(:description),
      namespace_id: group.id,
      organization_id: organization_id,
      visibility_level: Gitlab::VisibilityLevel::PRIVATE,
      initialize_with_readme: true
    ).execute
    unless project.persisted? && project.errors.empty?
      raise "Unable to create #{full_path}: #{project.errors.full_messages.join(', ')}"
    end
  end

  definition.fetch(:members).each do |username, access_level|
    user = User.find_by!(username: username)
    next if user.admin?

    member = ProjectMember.find_or_initialize_by(source: project, user: user)
    member.access_level = access_level
    member.save!
  end

  created_projects << project
end

usernames = projects.flat_map { |item| item.fetch(:members).keys }.uniq
usernames.each do |username|
  AuthorizedProjectsWorker.new.perform(User.find_by!(username: username).id)
end

projects.each do |definition|
  full_path = "inazuma/#{definition.fetch(:group)}/#{definition.fetch(:path)}"
  project = Project.find_by_full_path(full_path)
  raise "Project route verification failed: #{full_path}" unless project&.route
  raise "Repository verification failed: #{full_path}" unless project.repository_exists?
  raise "README verification failed: #{full_path}" unless project.repository.blob_at('HEAD', 'README.md')

  definition.fetch(:members).each_key do |username|
    user = User.find_by!(username: username)
    raise "Authorization verification failed: #{username} cannot read #{full_path}" unless Ability.allowed?(user, :read_project, project)
  end
end

puts "Verified #{created_projects.length} clean GitLab projects with routes, repositories, README files, and memberships."
