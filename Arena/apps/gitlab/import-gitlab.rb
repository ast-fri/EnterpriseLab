# frozen_string_literal: true

require 'json'

seed_path = ENV.fetch('SEED_JSON')
credentials_path = ENV.fetch('CREDENTIALS_JSON')
skip_import = ENV['SKIP_DATA_IMPORT'] == '1'
connection = ActiveRecord::Base.connection

unless skip_import
  export = JSON.parse(File.read(seed_path))
  raise "Expected a GitLab PostgreSQL export" unless export['service'] == 'gitlab' && export['database'] == 'postgresql'

  # GitLab owns these rows and migrations. Importing them from another instance
  # can prevent the current image from booting or running its own migrations.
  excluded_tables = %w[
    ar_internal_metadata
    schema_migrations
    application_settings
    plans
    shards
    postgres_async_foreign_key_validations
    postgres_async_indexes
    batched_background_migrations
    batched_background_migration_jobs
    batched_background_migration_job_transition_logs
  ].freeze

  tables = export.fetch('tables')
  project_namespace_ids = {}
  group_namespaces = tables.fetch('namespaces').to_h { |row| [row.fetch('id'), row] }
  tables.fetch('projects').each do |project|
    synthetic_id = 10_000 + project.fetch('id')
    project_namespace_ids[project.fetch('id')] = synthetic_id
    parent = group_namespaces.fetch(project.fetch('namespace_id'))
    project_namespace = parent.dup.merge(
      'id' => synthetic_id,
      'name' => project.fetch('name'),
      'path' => project.fetch('path'),
      'owner_id' => nil,
      'type' => 'Project',
      'parent_id' => project.fetch('namespace_id'),
      'visibility_level' => project.fetch('visibility_level'),
      'traversal_ids' => Array(parent['traversal_ids']) + [synthetic_id]
    )
    tables.fetch('namespaces') << project_namespace unless group_namespaces.key?(synthetic_id)
    project['project_namespace_id'] = synthetic_id
  end
  tables.fetch('issues').each do |issue|
    synthetic_id = project_namespace_ids.fetch(issue.fetch('project_id'))
    issue['namespace_id'] = synthetic_id
    issue['namespace_traversal_ids'] = [synthetic_id]
  end

  imported = {}
  connection.execute("SET session_replication_role = 'replica'")

  begin
    tables.each do |table, rows|
      next if rows.empty? || excluded_tables.include?(table)
      next unless connection.data_source_exists?(table)

      db_columns = connection.columns(table).to_h { |column| [column.name, column] }
      generated_columns = connection.select_values(<<~SQL)
        SELECT attname
        FROM pg_attribute
        WHERE attrelid = #{connection.quote(table)}::regclass
          AND attgenerated <> ''
      SQL
      columns = rows.flat_map(&:keys).uniq.select do |name|
        db_columns.key?(name) && !generated_columns.include?(name)
      end
      next if columns.empty?

      # Values encrypted with the source installation secrets cannot be decrypted
      # by this fresh installation. Drop nullable encrypted attributes; passwords
      # are replaced below through GitLab's model so they use the current secrets.
      rows.each do |row|
        if table == 'system_note_metadata' && row['namespace_id'].nil? && row['organization_id'].nil?
          row['organization_id'] = 1
        end
        columns.each do |name|
          column = db_columns.fetch(name)
          encrypted = (name.start_with?('encrypted_') && name != 'encrypted_password') ||
                      name.end_with?('_encrypted') || name.include?('ciphertext')
          row[name] = nil if encrypted && column.null
          # The extraction represented SQL NULL as the literal string "NULL"
          # in a handful of nullable milestone fields.
          row[name] = nil if row[name] == 'NULL' && column.null
        end
      end

      quoted_table = connection.quote_table_name(table)
      quoted_columns = columns.map { |name| connection.quote_column_name(name) }

      rows.each_slice(200) do |slice|
        json = connection.quote(JSON.generate(slice))
        column_list = quoted_columns.join(', ')
        sql = <<~SQL
          INSERT INTO #{quoted_table} (#{column_list})
          SELECT #{column_list}
          FROM json_populate_recordset(NULL::#{quoted_table}, #{json}::json)
          ON CONFLICT DO NOTHING
        SQL
        connection.execute(sql)
      end

      imported[table] = rows.length
    rescue StandardError => error
      raise "Failed while importing #{table}: #{error.message}"
    end
  ensure
    connection.execute("SET session_replication_role = 'origin'")
  end

  project_namespace_ids.each do |project_id, namespace_id|
    connection.execute(
      "UPDATE projects SET project_namespace_id = #{namespace_id} WHERE id = #{project_id}"
    )
    connection.execute(
      "UPDATE issues SET namespace_id = #{namespace_id}, " \
      "namespace_traversal_ids = ARRAY[#{namespace_id}] WHERE project_id = #{project_id}"
    )
  end

  # Move owned sequences beyond restored IDs so subsequent UI-created records do
  # not collide with the imported primary keys.
  imported.each_key do |table|
    next unless connection.column_exists?(table, 'id')

    sequence = connection.select_value(
      "SELECT pg_get_serial_sequence(#{connection.quote(table)}, 'id')"
    )
    next if sequence.nil?

    quoted_table = connection.quote_table_name(table)
    maximum = connection.select_value("SELECT MAX(id) FROM #{quoted_table}")
    next if maximum.nil?

    quoted_sequence = connection.quote_table_name(sequence)
    current = connection.select_value("SELECT last_value FROM #{quoted_sequence}").to_i
    target = [maximum.to_i, current].max
    connection.execute(
      "SELECT setval(#{connection.quote(sequence)}, #{target}, true)"
    )
  end

  puts "Imported #{imported.length} populated tables (conflicting bootstrap rows were preserved)."
else
  puts 'Seed data already imported; refreshing deterministic credentials.'
end

# The JSON contains routes for source namespaces that were not included in the
# export (including the original built-in bot namespaces). Leaving those rows
# prevents GitLab from recreating required internal users after a seeded persona
# is converted from a bot to a human account.
connection.execute(<<~SQL)
  DELETE FROM routes
  WHERE source_type = 'Namespace'
    AND NOT EXISTS (
      SELECT 1 FROM namespaces WHERE namespaces.id = routes.source_id
    )
SQL

# Align denormalized membership partition keys with the project namespaces that
# this importer creates for GitLab 18.
connection.execute(<<~SQL)
  UPDATE members
  SET member_namespace_id = projects.project_namespace_id
  FROM projects
  WHERE members.source_type = 'Project'
    AND members.source_id = projects.id
    AND projects.id BETWEEN 301 AND 306
SQL

credentials = JSON.parse(File.read(credentials_path)).fetch('users')
credentials.each do |entry|
  user = User.find_by(username: entry.fetch('username'))
  raise "Credential user not found: #{entry.fetch('username')}" unless user

  unless user.namespace
    namespace = Namespaces::UserNamespace.new(
      owner: user,
      name: user.name,
      path: user.username,
      organization_id: user.organization_id || 1
    )
    namespace.save!
    user.reload
  end

  # The source reused several built-in bot rows as named personas. Converting the
  # exported users to human accounts makes every listed credential UI-loginable.
  user.user_type = 0 if user.respond_to?(:user_type=)
  user.password = entry.fetch('password')
  user.password_confirmation = entry.fetch('password')
  user.password_automatically_set = false
  user.password_expires_at = nil if user.respond_to?(:password_expires_at=)
  user.confirmed_at ||= Time.current if user.respond_to?(:confirmed_at=)
  user.state = 'active'
  user.save!(validate: false)
end

missing = credentials.reject do |entry|
  user = User.find_by(username: entry.fetch('username'))
  user && user.namespace && user.valid_password?(entry.fetch('password'))
end
raise "Password verification failed for: #{missing.map { |item| item['username'] }.join(', ')}" unless missing.empty?

# Membership rows are imported directly, so run GitLab's synchronous refresh
# worker to materialize project_authorizations before declaring seeding done.
credentials.each do |entry|
  AuthorizedProjectsWorker.new.perform(User.find_by!(username: entry.fetch('username')).id)
end

expected = { 'users' => 8, 'projects' => 6, 'issues' => 18, 'merge_requests' => 9 }
actual = {
  'users' => User.where(id: 21..28).count,
  'projects' => Project.where(id: 301..306).count,
  'issues' => Issue.where(id: 1..1_000_000).where(project_id: 301..306).count,
  'merge_requests' => MergeRequest.where(target_project_id: 301..306).count
}
expected.each do |name, minimum|
  raise "Verification failed: expected at least #{minimum} #{name}, found #{actual.fetch(name)}" if actual.fetch(name) < minimum
end

puts "Verified #{actual.map { |name, count| "#{count} #{name}" }.join(', ')} and " \
     "#{credentials.length} UI passwords."
