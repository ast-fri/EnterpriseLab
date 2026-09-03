<?php

declare(strict_types=1);

set_time_limit(0);
error_reporting(E_ALL);

function fail(string $message): never
{
    fwrite(STDERR, "dolibarr-seed: error: {$message}\n");
    exit(1);
}

function requiredEnvironment(string $name): string
{
    $value = getenv($name);
    if ($value === false || $value === '') {
        fail("missing environment variable {$name}");
    }
    return $value;
}

function quotedIdentifier(string $identifier): string
{
    if (!preg_match('/^[A-Za-z0-9_]+$/', $identifier)) {
        fail("unsafe SQL identifier in export: {$identifier}");
    }
    return '`'.$identifier.'`';
}

function normalizeValue(mixed $value): mixed
{
    // The database extractor serializes SQL NULL as the literal string "NULL".
    return $value === 'NULL' ? null : $value;
}

$seedPath = requiredEnvironment('DOLIBARR_SEED_JSON');
$credentialsPath = requiredEnvironment('DOLIBARR_CREDENTIALS_JSON');
$markerPath = requiredEnvironment('DOLIBARR_SEED_MARKER');

if (!is_readable($seedPath)) {
    fail("seed JSON is not readable: {$seedPath}");
}
if (!is_readable($credentialsPath)) {
    fail("credentials JSON is not readable: {$credentialsPath}");
}

$seedHash = hash_file('sha256', $seedPath);
$existingHash = is_file($markerPath) ? trim((string) file_get_contents($markerPath)) : '';
if ($existingHash !== '') {
    if (hash_equals($existingHash, $seedHash)) {
        fwrite(STDOUT, "dolibarr-seed: matching JSON is already loaded; skipping import\n");
        exit(0);
    }
    fail('seed JSON changed after initialization; run dolibarr/seed.sh to rebuild the isolated volumes');
}

$export = json_decode((string) file_get_contents($seedPath), true, 512, JSON_THROW_ON_ERROR);
$credentials = json_decode((string) file_get_contents($credentialsPath), true, 512, JSON_THROW_ON_ERROR);

// Repair the one known multiline-value artifact produced by the source extractor.
$constantRows =& $export["tables"]["llx_const"];
$artifactHead = $constantRows[58] ?? null;
$artifactTail = $constantRows[63] ?? null;
if (is_array($artifactHead) && is_array($artifactTail) &&
    ($artifactHead["rowid"] ?? null) === "82" &&
    ($artifactHead["name"] ?? null) === "ADHERENT_CARD_TEXT" &&
    array_keys($artifactTail) === ["rowid", "name", "entity", "value", "type"]) {
    $continuation = "";
    for ($artifactIndex = 59; $artifactIndex <= 63; $artifactIndex++) {
        if (!isset($constantRows[$artifactIndex]["rowid"])) {
            fail("unexpected ADHERENT_CARD_TEXT extraction artifact shape");
        }
        $continuation .= $constantRows[$artifactIndex]["rowid"];
    }
    $repairedConstant = [
        "rowid" => $artifactHead["rowid"],
        "name" => $artifactHead["name"],
        "entity" => $artifactHead["entity"],
        "value" => $artifactHead["value"].$continuation,
        "type" => $artifactTail["name"],
        "visible" => $artifactTail["entity"],
        "note" => $artifactTail["value"],
        "tms" => $artifactTail["type"],
    ];
    array_splice($constantRows, 58, 6, [$repairedConstant]);
    fwrite(STDOUT, "dolibarr-seed: repaired multiline ADHERENT_CARD_TEXT export artifact\n");
}

if (($export['service'] ?? null) !== 'dolibarr' || !is_array($export['tables'] ?? null)) {
    fail('JSON is not a Dolibarr database export');
}
if (!is_array($credentials['users'] ?? null) || $credentials['users'] === []) {
    fail('credentials JSON must contain a non-empty users array');
}

$publishedUsernames = [];
foreach ($credentials['users'] as $credential) {
    if (is_string($credential['username'] ?? null)) {
        $publishedUsernames[$credential['username']] = true;
    }
}
$unexpectedExportUsers = array_values(array_filter(
    $export['tables']['llx_user'] ?? [],
    static fn(array $row): bool => !isset($publishedUsernames[$row['login'] ?? ''])
));
if ($unexpectedExportUsers !== []) {
    $unexpectedLogins = array_map(
        static fn(array $row): string => (string) ($row['login'] ?? '<missing>'),
        $unexpectedExportUsers
    );
    fail(
        'export contains users outside user-credentials.json; run '.
        'sanitize_seed_exports.py --write first: '.implode(', ', array_slice($unexpectedLogins, 0, 20))
    );
}

$versionRows = $export['tables']['llx_const'] ?? [];
$exportVersion = null;
foreach ($versionRows as $row) {
    if (($row['name'] ?? null) === 'MAIN_VERSION_LAST_UPGRADE') {
        $exportVersion = $row['value'] ?? null;
        break;
    }
}
if ($exportVersion !== '23.0.2') {
    fail('expected a Dolibarr 23.0.2 export, got '.var_export($exportVersion, true));
}

$database = requiredEnvironment('DOLI_DB_NAME');
$dsn = sprintf(
    'mysql:host=%s;port=%s;dbname=%s;charset=utf8mb4',
    requiredEnvironment('DOLI_DB_HOST'),
    getenv('DOLI_DB_HOST_PORT') ?: '3306',
    $database
);

$pdo = new PDO(
    $dsn,
    requiredEnvironment('DOLI_DB_USER'),
    requiredEnvironment('DOLI_DB_PASSWORD'),
    [
        PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
        PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
        PDO::ATTR_EMULATE_PREPARES => true,
    ]
);

$existingTablesStatement = $pdo->prepare(
    'SELECT TABLE_NAME FROM information_schema.TABLES WHERE TABLE_SCHEMA = ? AND TABLE_TYPE = ?'
);
$existingTablesStatement->execute([$database, 'BASE TABLE']);
$existingTables = array_fill_keys($existingTablesStatement->fetchAll(PDO::FETCH_COLUMN), true);

$compatibilityTable = "llx_project_task_timesheet";
if (!isset($existingTables[$compatibilityTable]) &&
    ($export["tables"][$compatibilityTable] ?? []) !== []) {
    // This populated table belongs to an optional source timesheet module.
    $pdo->exec(
        "CREATE TABLE `llx_project_task_timesheet` (".
        "`rowid` integer NOT NULL AUTO_INCREMENT, ".
        "`date_start` date DEFAULT NULL, `date_end` date DEFAULT NULL, ".
        "`status` integer NOT NULL DEFAULT 0, `note` text DEFAULT NULL, ".
        "`date_creation` datetime DEFAULT NULL, `date_modification` datetime DEFAULT NULL, ".
        "`fk_userid` integer DEFAULT NULL, `fk_user_modification` integer DEFAULT NULL, ".
        "PRIMARY KEY (`rowid`), KEY `idx_project_task_timesheet_user` (`fk_userid`)".
        ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci"
    );
    $existingTables[$compatibilityTable] = true;
    fwrite(STDOUT, "dolibarr-seed: created compatibility table {$compatibilityTable}\n");
}

$missingPopulatedTables = array_values(array_filter(
    array_diff(array_keys($export["tables"]), array_keys($existingTables)),
    static fn(string $table): bool => ($export["tables"][$table] ?? []) !== []
));
if ($missingPopulatedTables !== []) {
    fail("target schema is missing populated exported tables: ".implode(", ", $missingPopulatedTables));
}

$skippedEmptyTables = array_values(array_diff(
    array_keys($export["tables"]),
    array_keys($existingTables)
));
if ($skippedEmptyTables !== []) {
    fwrite(
        STDOUT,
        "dolibarr-seed: skipping ".count($skippedEmptyTables).
        " absent optional tables because their exports are empty\n"
    );
}
$totalRows = 0;
$populatedTables = 0;
$pdo->exec("SET SESSION sql_mode = ''");
$pdo->exec('SET FOREIGN_KEY_CHECKS = 0');

try {
    foreach ($export['tables'] as $tableName => $rows) {
        if (!is_array($rows)) {
            fail("table {$tableName} does not contain an array");
        }

        if (!isset($existingTables[$tableName])) {
            continue;
        }

        $quotedTable = quotedIdentifier($tableName);
        $pdo->exec("TRUNCATE TABLE {$quotedTable}");

        if ($rows === []) {
            continue;
        }

        $columns = array_keys($rows[0]);
        $columnStatement = $pdo->query("SHOW COLUMNS FROM {$quotedTable}");
        $targetColumns = array_fill_keys($columnStatement->fetchAll(PDO::FETCH_COLUMN), true);
        $missingColumns = array_values(array_diff($columns, array_keys($targetColumns)));
        if ($missingColumns !== []) {
            foreach ($missingColumns as $missingColumn) {
                foreach ($rows as $row) {
                    if (array_key_exists($missingColumn, $row) &&
                        normalizeValue($row[$missingColumn]) !== null) {
                        fail(
                            "target table {$tableName} is missing non-null column: ".
                            $missingColumn
                        );
                    }
                }
            }
            $columns = array_values(array_diff($columns, $missingColumns));
            fwrite(
                STDOUT,
                "dolibarr-seed: ignored null-only absent columns in {$tableName}: ".
                implode(", ", $missingColumns)."\n"
            );
        }
        $quotedColumns = implode(', ', array_map('quotedIdentifier', $columns));
        $columnCount = count($columns);
        $chunkSize = max(1, min(100, intdiv(60000, $columnCount)));

        $pdo->beginTransaction();
        try {
            foreach (array_chunk($rows, $chunkSize) as $chunk) {
                $rowPlaceholder = '('.implode(', ', array_fill(0, $columnCount, '?')).')';
                $sql = "INSERT INTO {$quotedTable} ({$quotedColumns}) VALUES ".
                    implode(', ', array_fill(0, count($chunk), $rowPlaceholder));
                $parameters = [];

                foreach ($chunk as $row) {
                    if (array_keys($row) !== array_merge($columns, $missingColumns)) {
                        fail("inconsistent columns in exported table {$tableName}");
                    }
                    foreach ($columns as $column) {
                        $parameters[] = normalizeValue($row[$column]);
                    }
                }

                $pdo->prepare($sql)->execute($parameters);
            }
            $pdo->commit();
        } catch (Throwable $error) {
            if ($pdo->inTransaction()) {
                $pdo->rollBack();
            }
            throw $error;
        }

        $totalRows += count($rows);
        $populatedTables++;
        fwrite(STDOUT, "dolibarr-seed: imported {$tableName} (".count($rows)." rows)\n");
    }

    $seenUsernames = [];
    $credentialUpdate = $pdo->prepare(
        'UPDATE llx_user
         SET pass = NULL,
             pass_crypted = ?,
             pass_temp = NULL,
             pass_encoding = NULL,
             statut = 1,
             flagdelsessionsbefore = NULL,
             datelastpassvalidation = NOW()
         WHERE login = ?'
    );

    foreach ($credentials['users'] as $credential) {
        $username = $credential['username'] ?? null;
        $password = $credential['password'] ?? null;
        if (!is_string($username) || $username === '' ||
            !is_string($password) || strlen($password) < 8) {
            fail('every credential requires a username and a password of at least 8 characters');
        }
        if (isset($seenUsernames[$username])) {
            fail("duplicate credential for {$username}");
        }
        $seenUsernames[$username] = true;

        // The source database uses Dolibarr's legacy MD5 password representation.
        // Dolibarr accepts it and transparently upgrades it after a successful login.
        $credentialUpdate->execute([md5($password), $username]);
        if ($credentialUpdate->rowCount() !== 1) {
            fail("credential user was not found exactly once in imported data: {$username}");
        }
    }


    $credentialUsernames = array_keys($seenUsernames);
    $credentialPlaceholders = implode(', ', array_fill(0, count($credentialUsernames), '?'));

    // Preserve non-seed users as hidden references so imported author IDs stay valid.
    $hideNonSeedUsers = $pdo->prepare(
        "UPDATE llx_user SET entity = 999, statut = 0, admin = 0 ".
        "WHERE login NOT IN ({$credentialPlaceholders})"
    );
    $hideNonSeedUsers->execute($credentialUsernames);

    // The export contains these definitions but omitted their physical columns.
    $pdo->exec(
        'ALTER TABLE llx_user_extrafields '.
        'ADD COLUMN IF NOT EXISTS fk_service integer DEFAULT NULL'
    );
    $pdo->exec(
        'ALTER TABLE llx_projet_task_extrafields '.
        'ADD COLUMN IF NOT EXISTS fk_service integer DEFAULT NULL, '.
        'ADD COLUMN IF NOT EXISTS invoiceable tinyint(1) DEFAULT NULL'
    );
    $sellistParameter = serialize([
        'options' => [
            'product:ref|label:rowid::(tosell:=:1) AND (fk_product_type:=:1)' => 'N',
        ],
    ]);
    $repairSellist = $pdo->prepare(
        'UPDATE llx_extrafields SET param = ? '.
        'WHERE name = ? AND elementtype IN (?, ?)'
    );
    $repairSellist->execute([
        $sellistParameter, 'fk_service', 'user', 'projet_task',
    ]);

    // Seed users are broad business users, but cannot administer users or groups.
    $nonAdminUsernames = array_values(array_filter(
        $credentialUsernames,
        static fn(string $username): bool => $username !== 'admin'
    ));
    $nonAdminPlaceholders = implode(
        ', ', array_fill(0, count($nonAdminUsernames), '?')
    );
    $deleteSeedRights = $pdo->prepare(
        "DELETE ur FROM llx_user_rights ur ".
        "INNER JOIN llx_user u ON u.rowid = ur.fk_user ".
        "WHERE u.login IN ({$nonAdminPlaceholders})"
    );
    $deleteSeedRights->execute($nonAdminUsernames);
    $grantBusinessRights = $pdo->prepare(
        "INSERT INTO llx_user_rights (entity, fk_user, fk_id) ".
        "SELECT 1, u.rowid, r.id FROM llx_user u CROSS JOIN llx_rights_def r ".
        "WHERE u.login IN ({$nonAdminPlaceholders}) ".
        "AND r.entity = 1 AND r.enabled = 1 AND r.module <> 'user'"
    );
    $grantBusinessRights->execute($nonAdminUsernames);

    $visibleUserCount = (int) $pdo->query(
        'SELECT COUNT(*) FROM llx_user WHERE entity IN (0, 1)'
    )->fetchColumn();
    if ($visibleUserCount !== count($credentialUsernames)) {
        fail("visible user cleanup failed: {$visibleUserCount}");
    }
    $businessRightCount = (int) $pdo->query(
        "SELECT COUNT(*) FROM llx_rights_def ".
        "WHERE entity = 1 AND enabled = 1 AND module <> 'user'"
    )->fetchColumn();
    $permissionCheck = $pdo->prepare(
        "SELECT MIN(right_count) FROM (".
        "SELECT u.rowid, COUNT(ur.rowid) AS right_count FROM llx_user u ".
        "LEFT JOIN llx_user_rights ur ON ur.fk_user = u.rowid AND ur.entity = 1 ".
        "WHERE u.login IN ({$nonAdminPlaceholders}) GROUP BY u.rowid".
        ") seed_rights"
    );
    $permissionCheck->execute($nonAdminUsernames);
    if ((int) $permissionCheck->fetchColumn() !== $businessRightCount) {
        fail('business permission grant verification failed');
    }
    fwrite(
        STDOUT,
        "dolibarr-seed: retained 13 visible seed users and granted ".
        "{$businessRightCount} business rights to each non-admin user\n"
    );
    $placeholders = implode(', ', array_fill(0, count($seenUsernames), '?'));
    $verification = $pdo->prepare(
        "SELECT login, pass_crypted, statut FROM llx_user WHERE login IN ({$placeholders})"
    );
    $verification->execute(array_keys($seenUsernames));
    $verifiedUsers = $verification->fetchAll();
    if (count($verifiedUsers) !== count($seenUsernames)) {
        fail('not all credential users were present after import');
    }
    foreach ($verifiedUsers as $user) {
        $expectedPassword = null;
        foreach ($credentials['users'] as $credential) {
            if ($credential['username'] === $user['login']) {
                $expectedPassword = $credential['password'];
                break;
            }
        }
        if ((int) $user['statut'] !== 1 ||
            !hash_equals(md5((string) $expectedPassword), (string) $user['pass_crypted'])) {
            fail("credential verification failed for {$user['login']}");
        }
    }
} finally {
    $pdo->exec('SET FOREIGN_KEY_CHECKS = 1');
}

if (file_put_contents($markerPath, $seedHash."\n", LOCK_EX) === false) {
    fail("could not write seed marker: {$markerPath}");
}

fwrite(
    STDOUT,
    "dolibarr-seed: completed {$populatedTables} populated tables, ".
    "{$totalRows} rows, and ".count($seenUsernames)." preset UI credentials\n"
);
