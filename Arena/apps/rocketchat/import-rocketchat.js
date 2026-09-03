/* global db, ObjectId, BinData, print */

// Executed by mongosh inside the MongoDB container.
const fs = require('fs');

const jsonPath = process.env.SEED_JSON_PATH;
if (!jsonPath) {
  throw new Error('SEED_JSON_PATH is required');
}

const dump = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
if (
  dump.service !== 'rocketchat' ||
  dump.database !== 'mongodb' ||
  dump.source !== 'database_complete_dump' ||
  !dump.collections ||
  typeof dump.collections !== 'object' ||
  Array.isArray(dump.collections)
) {
  throw new Error('Input is not a supported Rocket.Chat database_complete_dump');
}

const excludedCollections = new Set([
  'rocketchat_analytics',
  'rocketchat_cron',
  'rocketchat_cron_history',
  'rocketchat_federation_keys',
  'rocketchat_import',
  'rocketchat_import_data',
  'rocketchat_nps',
  'rocketchat_server_events',
  'rocketchat_sessions',
  'rocketchat_statistics',
  'rocketchat_workspace_credentials',
  'usersSessions',
]);

const dateKeys = new Set([
  '_computedAt',
  '_createdAt',
  '_updatedAt',
  'buildAt',
  'closedAt',
  'createdAt',
  'dismissedAt',
  'expirationDate',
  'expireAt',
  'finishedAt',
  'installedAt',
  'intendedAt',
  'lastActivityAt',
  'lastFinishedAt',
  'lastLoginAt',
  'lastMessageSentAt',
  'lastRunAt',
  'lm',
  'lockedAt',
  'loginAt',
  'logoutAt',
  'ls',
  'nextRunAt',
  'startAt',
  'startedAt',
  'ts',
  'updatedAt',
  'uploadedAt',
  'uploadDate',
]);

const isoDate = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$/;

function convertDates(value, key) {
  if (Array.isArray(value)) {
    return value.map((item) => convertDates(item));
  }
  if (value && typeof value === 'object') {
    const converted = {};
    for (const [childKey, childValue] of Object.entries(value)) {
      converted[childKey] = convertDates(childValue, childKey);
    }
    return converted;
  }
  if (dateKeys.has(key) && typeof value === 'string' && isoDate.test(value)) {
    const date = new Date(value);
    if (!Number.isNaN(date.getTime())) {
      return date;
    }
  }
  return value;
}

function isCloudCredentialSetting(document) {
  return (
    document &&
    typeof document._id === 'string' &&
    (/^Cloud_/.test(document._id) || document._id === 'Register_Server')
  );
}

function convertDocument(collectionName, source) {
  const document = convertDates(source);

  if (collectionName === 'rocketchat_avatars.chunks') {
    document._id = new ObjectId(document._id);
    document.data = new BinData(0, document.data);
  }

  return document;
}

const target = db.getSiblingDB('rocketchat');
const seedMode = process.env.SEED_MODE || 'if-needed';
// bcrypt(SHA-256("Admin123!")), as expected by Meteor Accounts.
const adminPasswordHash =
  process.env.SEED_ADMIN_PASSWORD_HASH ||
  '$2b$10$GtJVaC8IlUCF7aI8FhdiRuv7KOyoDNqvzfF/omc20N8xqh/.s7lSG';
const seedMarker = target.getCollection('rocketchat_seed_metadata').findOne({
  _id: 'json-baseline',
});
const existingSeedAdmin = target.getCollection('users').findOne({
  _id: 'seed-admin-user',
});

if (seedMode === 'if-needed' && seedMarker) {
  print('seed: baseline marker exists; keeping the current database');
  quit(0);
}

// Recognize databases created by the earlier seeder, before the marker existed.
if (
  seedMode === 'if-needed' &&
  existingSeedAdmin &&
  target.getCollection('rocketchat_message').countDocuments({}) > 0
) {
  target.getCollection('rocketchat_seed_metadata').updateOne(
    { _id: 'json-baseline' },
    {
      $set: {
        sourceTimestamp: dump.timestamp,
        adoptedAt: new Date(),
      },
    },
    { upsert: true },
  );
  print('seed: adopted the existing seeded database; no data was replaced');
  quit(0);
}

if (seedMode === 'if-needed' && target.getCollection('users').countDocuments({}) > 0) {
  throw new Error(
    'MongoDB contains an unseeded Rocket.Chat database. Run seed.sh once to reset it safely.',
  );
}

if (!['if-needed', 'reset'].includes(seedMode)) {
  throw new Error(`Unsupported SEED_MODE: ${seedMode}`);
}
if (!adminPasswordHash || !/^\$2[aby]\$/.test(adminPasswordHash)) {
  throw new Error('SEED_ADMIN_PASSWORD_HASH must contain a valid bcrypt hash');
}

const dropResult = target.dropDatabase();
if (!dropResult.ok) {
  throw new Error(`Could not reset rocketchat database: ${JSON.stringify(dropResult)}`);
}

const batchSize = 500;
const summary = {
  importedCollections: 0,
  importedDocuments: 0,
  skippedCollections: [],
  skippedDocuments: 0,
  filteredCloudSettings: 0,
  normalizedMessages: 0,
  normalizedDirectRooms: 0,
};

for (const [collectionName, sourceDocuments] of Object.entries(dump.collections)) {
  if (!Array.isArray(sourceDocuments)) {
    throw new Error(`Collection ${collectionName} is not an array`);
  }

  if (excludedCollections.has(collectionName)) {
    summary.skippedCollections.push(collectionName);
    summary.skippedDocuments += sourceDocuments.length;
    continue;
  }

  let documents = sourceDocuments;
  if (collectionName === 'rocketchat_settings') {
    documents = documents.filter((document) => {
      const filtered = isCloudCredentialSetting(document);
      if (filtered) summary.filteredCloudSettings += 1;
      return !filtered;
    });
  }

  if (documents.length === 0) continue;

  const collection = target.getCollection(collectionName);
  let inserted = 0;
  for (let offset = 0; offset < documents.length; offset += batchSize) {
    const batch = documents
      .slice(offset, offset + batchSize)
      .map((document) => convertDocument(collectionName, document));
    const result = collection.insertMany(batch, { ordered: true });
    inserted += Object.keys(result.insertedIds).length;
  }

  if (inserted !== documents.length) {
    throw new Error(
      `Collection ${collectionName}: expected ${documents.length} inserts, got ${inserted}`,
    );
  }
  summary.importedCollections += 1;
  summary.importedDocuments += inserted;
  print(`seed: imported ${collectionName}: ${inserted}`);
}

summary.skippedCollections.sort();
const normalizedMessages = target
  .getCollection('rocketchat_message')
  .updateMany({ t: 'uj', msg: { $type: 'string' } }, { $unset: { t: '' } });
summary.normalizedMessages = normalizedMessages.modifiedCount;

for (const room of target.getCollection('rocketchat_room').find({ t: 'd' }).toArray()) {
  const subscriptions = target
    .getCollection('rocketchat_subscription')
    .find({ rid: room._id, t: 'd' })
    .toArray();
  if (subscriptions.length !== 2) {
    throw new Error(`Direct room ${room._id} must have exactly two subscriptions`);
  }

  const participants = subscriptions.map((subscription) => subscription.u);
  target.getCollection('rocketchat_room').updateOne(
    { _id: room._id },
    {
      $set: {
        uids: participants.map((participant) => participant._id),
        usernames: participants.map((participant) => participant.username),
        _USERNAMES: participants.map((participant) => participant.username).sort(),
        usersCount: 2,
      },
    },
  );

  for (const subscription of subscriptions) {
    const other = participants.find(
      (participant) => participant._id !== subscription.u._id,
    );
    target.getCollection('rocketchat_subscription').updateOne(
      { _id: subscription._id },
      { $set: { name: other.username, fname: other.name } },
    );
  }
  summary.normalizedDirectRooms += 1;
}

const now = new Date();
target.getCollection('users').insertOne({
  _id: 'seed-admin-user',
  username: 'admin',
  name: 'Seed Administrator',
  emails: [{ address: 'admin@rocketchat.local', verified: true }],
  type: 'user',
  status: 'offline',
  statusConnection: 'offline',
  active: true,
  roles: ['user', 'admin'],
  requirePasswordChange: false,
  services: { password: { bcrypt: adminPasswordHash } },
  settings: {},
  createdAt: now,
  _updatedAt: now,
});
target.getCollection('rocketchat_seed_metadata').insertOne({
  _id: 'json-baseline',
  sourceTimestamp: dump.timestamp,
  seededAt: now,
  importSummary: summary,
});
summary.importedDocuments += 1;
print(`SEED_IMPORT_SUMMARY=${JSON.stringify(summary)}`);
