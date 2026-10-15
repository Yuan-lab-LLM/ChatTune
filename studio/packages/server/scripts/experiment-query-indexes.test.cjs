// Node 24+: runs the actual index migration against an in-memory SQLite database.
// TypeORM's TableIndex/QueryRunner boundary is adapted; the full integration suite remains separate.
const assert = require('node:assert/strict');
const { test, after } = require('node:test');
const { DatabaseSync } = require('node:sqlite');
const { registerHooks } = require('node:module');
const { pathToFileURL, fileURLToPath } = require('node:url');
const path = require('node:path');
const migrationUrl = pathToFileURL(path.resolve(__dirname, '../src/migrations/1960000000000-OptimizeExperimentQueries.ts')).href;
const hooks = registerHooks({
    resolve(specifier, context, next) {
        return context.parentURL === migrationUrl && specifier === 'typeorm'
            ? { url: 'experiment-index-mock:typeorm', shortCircuit: true } : next(specifier, context);
    },
    load(url, context, next) {
        return url === 'experiment-index-mock:typeorm' ? { format: 'module', shortCircuit: true,
            source: 'export class TableIndex { constructor(options) { Object.assign(this, options); } }' } : next(url, context);
    },
});
const { OptimizeExperimentQueries1960000000000 } = require(fileURLToPath(migrationUrl));
after(() => hooks.deregister());

test('indexes accelerate actual read predicates and survive rollback/reapply without changing records', async () => {
    const db = new DatabaseSync(':memory:');
    try {
        db.exec(`CREATE TABLE asset_version(id TEXT, ownerUserId TEXT, nodeId TEXT);
            CREATE TABLE task_asset(id TEXT, taskId TEXT);
            CREATE TABLE evaluation_result(id TEXT, modelVersionId TEXT, attemptId TEXT);
            INSERT INTO asset_version VALUES ('asset', 'owner', 'node');
            INSERT INTO task_asset VALUES ('link', 'task');
            INSERT INTO evaluation_result VALUES ('result', 'model', 'attempt');`);
        const queryRunner = {
            createIndex: async (table, index) => db.exec(`CREATE INDEX "${index.name}" ON "${table}" (${index.columnNames.map(column => `"${column}"`).join(', ')})`),
            dropIndex: async (_table, name) => db.exec(`DROP INDEX "${name}"`),
        };
        const migration = new OptimizeExperimentQueries1960000000000();
        const checks = [
            ['asset_version', "ownerUserId = 'owner' AND nodeId = 'node'", 'IDX_asset_owner_node'],
            ['task_asset', "taskId = 'task'", 'IDX_task_asset_task'],
            ['evaluation_result', "modelVersionId = 'model'", 'IDX_evaluation_model'],
            ['evaluation_result', "attemptId = 'attempt'", 'IDX_evaluation_attempt'],
        ];
        const before = checks.map(([table, where]) => db.prepare(`SELECT * FROM ${table} WHERE ${where}`).all());
        await migration.up(queryRunner);
        for (const [table, where, index] of checks) {
            assert.match(db.prepare(`EXPLAIN QUERY PLAN SELECT * FROM ${table} WHERE ${where}`).get().detail, new RegExp(index));
        }
        await migration.down(queryRunner);
        assert.equal(db.prepare("SELECT count(*) AS n FROM sqlite_master WHERE type = 'index'").get().n, 0);
        await migration.up(queryRunner);
        assert.deepEqual(checks.map(([table, where]) => db.prepare(`SELECT * FROM ${table} WHERE ${where}`).all()), before);
    } finally { db.close(); }
});
