# frozen_string_literal: true

require 'json'
require 'set'

json_path = ENV.fetch('ZAMMAD_SEED_JSON', '/seed/zammad.json')
credentials_path = ENV.fetch('ZAMMAD_SEED_CREDENTIALS', '/seed/user-credentials.json')
marker_path = ENV.fetch('ZAMMAD_SEED_MARKER', '/opt/zammad/storage/.enterprise-arena-json-imported')

if File.exist?(marker_path)
  puts "zammad-seed: marker exists; preserving the current database"
  exit 0
end

export = JSON.parse(File.read(json_path))
abort 'zammad-seed: this is not a Zammad PostgreSQL export' unless export['service'] == 'zammad' && export['database'] == 'postgresql'
tables = export.fetch('tables')
credentials = JSON.parse(File.read(credentials_path)).fetch('users')
connection = ActiveRecord::Base.connection

allowed_export_logins = credentials.flat_map do |credential|
  username = credential.fetch('username').downcase
  [username, "#{username}@inazuma.com"]
end.to_set
unexpected_export_users = tables.fetch('users', []).reject do |row|
  login = (row['login'] || row['email']).to_s.downcase
  allowed_export_logins.include?(login)
end
unless unexpected_export_users.empty?
  logins = unexpected_export_users.map { |row| row['login'] || row['email'] }
  abort "zammad-seed: export contains users outside user-credentials.json; run sanitize_seed_exports.py --write first: #{logins.join(', ')}"
end

connection.transaction do
  connection.execute('SET LOCAL session_replication_role = replica')

  exported_ticket_ids = tables.fetch('tickets', []).map { |row| Integer(row.fetch('id')) }
  unless exported_ticket_ids.empty?
    ticket_id_list = exported_ticket_ids.join(',')
    connection.tables.each do |table_name|
      next if table_name == 'tickets'
      next unless connection.columns(table_name).any? { |column| column.name == 'ticket_id' }
      connection.execute("DELETE FROM #{connection.quote_table_name(table_name)} WHERE ticket_id NOT IN (#{ticket_id_list})")
    end
    exported_article_ids = tables.fetch('ticket_articles', []).map { |row| Integer(row.fetch('id')) }
    unless exported_article_ids.empty?
      connection.execute("DELETE FROM ticket_articles WHERE id NOT IN (#{exported_article_ids.join(',')})")
    end
    connection.execute("DELETE FROM tickets WHERE id NOT IN (#{ticket_id_list})")
  end

  tables.each do |table_name, source_rows|
    next if source_rows.empty?
    abort "zammad-seed: unsafe table name #{table_name.inspect}" unless table_name.match?(/\A[a-z][a-z0-9_]*\z/)
    abort "zammad-seed: target table #{table_name} does not exist" unless connection.data_source_exists?(table_name)

    target_columns = connection.columns(table_name).map(&:name)
    source_columns = source_rows.first.keys
    missing = source_columns - target_columns
    non_null_missing = missing.select { |column| source_rows.any? { |row| !row[column].nil? } }
    abort "zammad-seed: #{table_name} is missing non-null columns: #{non_null_missing.join(', ')}" unless non_null_missing.empty?

    rows = source_rows.map { |row| row.slice(*(source_columns & target_columns)) }
    model = Class.new(ActiveRecord::Base) do
      self.table_name = table_name
      self.inheritance_column = :_disabled_sti
    end
    primary_key = connection.primary_key(table_name)

    rows.each_slice(500) do |batch|
      if primary_key && batch.all? { |row| row.key?(primary_key) }
        model.upsert_all(batch, unique_by: primary_key, record_timestamps: false)
      else
        model.insert_all(batch, record_timestamps: false)
      end
    end
    puts "zammad-seed: imported #{table_name}=#{rows.length}"
  end

  connection.execute('SET LOCAL session_replication_role = origin')
end

tables.each_key do |table_name|
  next unless connection.data_source_exists?(table_name)
  primary_key = connection.primary_key(table_name)
  next unless primary_key
  sequence = connection.select_value("SELECT pg_get_serial_sequence(#{connection.quote(table_name)}, #{connection.quote(primary_key)})")
  next unless sequence
  maximum = connection.select_value("SELECT MAX(#{connection.quote_column_name(primary_key)}) FROM #{connection.quote_table_name(table_name)}").to_i
  connection.execute("SELECT setval(#{connection.quote(sequence)}, #{[maximum, 1].max}, #{maximum.positive?})")
end

application_secret_row = tables.fetch('settings').find { |row| row['name'] == 'application_secret' }
abort 'zammad-seed: export has no application_secret setting' unless application_secret_row
application_secret = YAML.safe_load(application_secret_row.fetch('state_current')).fetch('value')
Setting.set('application_secret', application_secret)
PasswordHash.remove_instance_variable(:@secret) if PasswordHash.instance_variable_defined?(:@secret)

names = {
  'admin' => %w[System Administrator],
  'abigail.mitchell' => %w[Abigail Mitchell],
  'aarav.mittal' => %w[Aarav Mittal],
  'surya.reddy' => %w[Surya Reddy],
  'raj.patel' => %w[Raj Patel],
  'rahul.khanna' => %w[Rahul Khanna],
  'karan.sharma' => %w[Karan Sharma],
  'priya.arora' => %w[Priya Arora],
  'sameer.malhotra' => %w[Sameer Malhotra],
  'ethan.reynolds' => %w[Ethan Reynolds],
  'anjali.mathew' => %w[Anjali Mathew],
  'vandana.reddy' => %w[Vandana Reddy],
  'neeraj.sharma' => %w[Neeraj Sharma]
}.freeze

user_sequence = connection.select_value("SELECT pg_get_serial_sequence('users', 'id')")
user_maximum = connection.select_value('SELECT MAX(id) FROM users').to_i
connection.execute("SELECT setval(#{connection.quote(user_sequence)}, #{[user_maximum, 1].max}, #{user_maximum.positive?})")

admin_role = Role.find_by!(name: 'Admin')
agent_role = Role.find_by!(name: 'Agent')
seed_users = []

credentials.each do |credential|
  login = credential.fetch('username')
  first_name, last_name = names.fetch(login)
  email = login == 'admin' ? 'admin@localhost.invalid' : "#{login}@inazuma.com"
  user = if login == 'admin'
           User.find_by(id: 3) || User.find_by(login: login)
         else
           User.find_by(login: [login, email]) || User.find_by(email: email)
         end
  user ||= User.new
  user.id ||= if login == 'admin' && !User.exists?(id: 3)
                3
              else
                User.maximum(:id).to_i + 1
              end
  user.assign_attributes(
    login: login,
    firstname: first_name,
    lastname: last_name,
    email: email,
    active: true,
    verified: true,
    login_failed: 0,
    created_by_id: user.created_by_id || 1,
    updated_by_id: 1,
    password: Argon2::Password.new(secret: application_secret).create(credential.fetch('password'))
  )
  user.roles = [login == 'admin' ? admin_role : agent_role]
  user.save!
  seed_users << user
  puts "zammad-seed: credential user #{login}=#{user.id}"
end

seed_ids = seed_users.map(&:id)

# Zammad bootstraps a loginable Nicole Braun sample at id=2. It is not a
# required system principal, so remap its few audit references to our fixed
# administrator and remove it. User id=1 (login "-") is Zammad's internal
# system principal and must remain.
sample_user = User.find_by(id: 2, login: 'nicole.braun@zammad.org')
if sample_user
  admin_id = seed_users.find { |user| user.login == 'admin' }.id
  connection.transaction do
    connection.execute("UPDATE users SET updated_by_id = #{admin_id} WHERE updated_by_id = 2")
    connection.execute("UPDATE histories SET created_by_id = #{admin_id} WHERE created_by_id = 2")
    connection.execute('DELETE FROM roles_users WHERE user_id = 2')
    user_lookup_id = connection.select_value("SELECT id FROM object_lookups WHERE name = 'User'")
    connection.execute("DELETE FROM avatars WHERE object_lookup_id = #{Integer(user_lookup_id)} AND o_id = 2") if user_lookup_id
    connection.execute('DELETE FROM users WHERE id = 2')
  end
  puts 'zammad-seed: removed stock Nicole Braun sample account'
end

unexpected_users = User.where.not(id: [1] + seed_ids)
unless unexpected_users.empty?
  raise "Unexpected non-seed users remain after import: #{unexpected_users.pluck(:login).join(', ')}"
end

agent_ids = seed_users.reject { |user| user.login == 'admin' }.map(&:id)
group_ids = Group.where(active: true).pluck(:id)
connection.transaction do
  connection.execute("DELETE FROM groups_users WHERE user_id IN (#{agent_ids.join(',')})")
  rows = agent_ids.product(group_ids).map { |user_id, group_id| { user_id: user_id, group_id: group_id, access: 'full' } }
  Class.new(ActiveRecord::Base) { self.table_name = 'groups_users' }.insert_all(rows) unless rows.empty?
end

Setting.set('fqdn', 'localhost:8050')
Setting.set('http_type', 'http')
Setting.set('two_factor_authentication_enforce_role_ids', [])

tables.each_key do |table_name|
  next unless connection.data_source_exists?(table_name)
  primary_key = connection.primary_key(table_name)
  next unless primary_key
  sequence = connection.select_value("SELECT pg_get_serial_sequence(#{connection.quote(table_name)}, #{connection.quote(primary_key)})")
  next unless sequence
  maximum = connection.select_value("SELECT MAX(#{connection.quote_column_name(primary_key)}) FROM #{connection.quote_table_name(table_name)}").to_i
  connection.execute("SELECT setval(#{connection.quote(sequence)}, #{[maximum, 1].max}, #{maximum.positive?})")
end

File.write(marker_path, "seeded_at=#{Time.now.utc.iso8601}\nsource=#{File.basename(json_path)}\n")
File.write('/tmp/zammad-seed-imported', "yes\n")
puts "zammad-seed: complete; active credential users=#{User.where(id: seed_ids, active: true).count}"
