const crypto = require('crypto');
const fs = require('fs');
const bcrypt = require('/app/bundle/programs/server/npm/node_modules/bcrypt');
const { MongoClient } = require('/app/bundle/programs/server/npm/node_modules/mongodb');

const mongoUrl = process.env.MONGO_URL;
const credentialsPath = process.env.USER_CREDENTIALS_PATH;

if (!mongoUrl || !credentialsPath) {
  throw new Error('MONGO_URL and USER_CREDENTIALS_PATH are required');
}

const credentials = JSON.parse(fs.readFileSync(credentialsPath, 'utf8'));
if (!Array.isArray(credentials.users) || credentials.users.length === 0) {
  throw new Error('Credentials JSON must contain a non-empty users array');
}

const usernames = new Set();
for (const credential of credentials.users) {
  if (
    !credential ||
    typeof credential.username !== 'string' ||
    !credential.username ||
    typeof credential.password !== 'string' ||
    credential.password.length < 8
  ) {
    throw new Error('Every credential requires a username and password of at least 8 characters');
  }
  if (usernames.has(credential.username)) {
    throw new Error(`Duplicate credential for ${credential.username}`);
  }
  usernames.add(credential.username);
}

async function main() {
  const client = new MongoClient(mongoUrl);
  await client.connect();

  try {
    const users = client.db('rocketchat').collection('users');
    for (const credential of credentials.users) {
      const digest = crypto
        .createHash('sha256')
        .update(credential.password)
        .digest('hex');
      const passwordHash = await bcrypt.hash(digest, 10);
      const result = await users.updateOne(
        { username: credential.username },
        {
          $set: {
            'services.password.bcrypt': passwordHash,
            requirePasswordChange: false,
            _updatedAt: new Date(),
          },
          $unset: {
            'services.password.reset': '',
          },
        },
      );
      if (result.matchedCount !== 1) {
        throw new Error(`Credential user not found in seeded data: ${credential.username}`);
      }
    }
  } finally {
    await client.close();
  }

  console.log(`credentials: applied preset passwords to ${credentials.users.length} users`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
