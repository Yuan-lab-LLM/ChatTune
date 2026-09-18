import fs from 'fs';
import { DataSource } from 'typeorm';
import { AuthDao } from '../dao/Auth';
import {
    AuthLoginFailureTable,
    AuthSessionTable,
    AuthUserTable,
} from '../models/Auth';

async function main() {
    if (process.argv.length !== 4 || process.argv[2] !== '--username') {
        throw new Error('Usage: reset-admin-password --username NAME');
    }

    const username = process.argv[3].trim();
    const password = process.env.MEDFLOW_ADMIN_PASSWORD;
    const databasePath = process.env.MEDFLOW_DATABASE_PATH;
    if (!username) {
        throw new Error('Administrator username is required.');
    }
    if (!password) {
        throw new Error('MEDFLOW_ADMIN_PASSWORD is empty.');
    }
    if (!databasePath || !fs.existsSync(databasePath)) {
        throw new Error(`Studio database not found: ${databasePath || '(unset)'}`);
    }

    const database = new DataSource({
        type: 'better-sqlite3',
        database: databasePath,
        entities: [AuthUserTable, AuthSessionTable, AuthLoginFailureTable],
        synchronize: false,
    });

    await database.initialize();
    try {
        await AuthDao.resetAdminPasswordByUsername(username, password);
        console.log(`Administrator password reset: ${username}. Existing sessions revoked.`);
    } finally {
        await database.destroy();
    }
}

main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
});
