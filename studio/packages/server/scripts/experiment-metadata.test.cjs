const assert = require('node:assert/strict');
const { test, after } = require('node:test');
const { registerHooks } = require('node:module');
const { pathToFileURL, fileURLToPath } = require('node:url');
const path = require('node:path');
const { ListQuery, createListDatabase } = require('./experiment-list-query.test.cjs');
const { applyExperimentListFilters } = require('../src/services/experimentListQuery.ts');
const url = pathToFileURL(path.resolve(__dirname, '../src/migrations/1970000000000-AddExperimentMetadata.ts')).href;
const hooks = registerHooks({
    resolve(specifier, context, next) {
        return context.parentURL === url && specifier === 'typeorm' ? { url: 'metadata-mock:typeorm', shortCircuit: true } : next(specifier, context);
    },
    load(specifier, context, next) {
        return specifier === 'metadata-mock:typeorm' ? { format: 'module', shortCircuit: true,
            source: 'export class TableColumn { constructor(values) { Object.assign(this, values); } }' } : next(specifier, context);
    },
});
const { AddExperimentMetadata1970000000000 } = require(fileURLToPath(url));
after(() => hooks.deregister());

test('metadata migration defaults old rows, filters exact bound tags, and rolls back without losing experiments', async () => {
    const db = createListDatabase();
    const migration = new AddExperimentMetadata1970000000000();
    const runner = {
        addColumn: async (table, column) => db.exec(`ALTER TABLE "${table}" ADD COLUMN "${column.name}" ${column.type} NOT NULL DEFAULT ${column.default}`),
        dropColumn: async (table, column) => db.exec(`ALTER TABLE "${table}" DROP COLUMN "${column}"`),
    };
    try {
        const before = db.prepare('SELECT id, spec FROM experiment ORDER BY id').all();
        await migration.up(runner);
        assert.equal(db.prepare('SELECT notes FROM experiment LIMIT 1').get().notes, '');
        assert.equal(db.prepare('SELECT tags FROM experiment LIMIT 1').get().tags, '[]');
        const tags = ['医疗', '100%_!', "x' OR 1=1 --"];
        db.prepare('UPDATE experiment SET notes = ?, tags = ? WHERE id IN (?, ?)').run('备注', JSON.stringify(tags), 'late', 'foreign-late');
        for (const tag of tags) {
            const query = applyExperimentListFilters(new ListQuery(db), 'owner', { tag });
            const [rows, total] = await query.orderBy('experiment.id', 'ASC').getManyAndCount();
            assert.equal(total, 1); assert.deepEqual(rows.map(row => row.id), ['late']);
        }
        const [rows, total] = await applyExperimentListFilters(new ListQuery(db), 'owner', { tag: '医' }).orderBy('experiment.id', 'ASC').getManyAndCount();
        assert.deepEqual(rows, []); assert.equal(total, 0);
        await migration.down(runner);
        assert.deepEqual(db.prepare('SELECT id, spec FROM experiment ORDER BY id').all(), before);
        await migration.up(runner);
        assert.equal(db.prepare("SELECT tags FROM experiment WHERE id = 'late'").get().tags, '[]');
    } finally { db.close(); }
});
