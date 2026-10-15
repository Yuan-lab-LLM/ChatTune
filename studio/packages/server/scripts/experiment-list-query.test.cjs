// Node 24+, runs production filter predicates against an in-memory SQLite database.
// QueryBuilder's execution boundary is adapted; full TypeORM validation remains separate.
const assert = require('node:assert/strict');
const { test } = require('node:test');
const { DatabaseSync } = require('node:sqlite');
const { applyExperimentListFilters } = require('../src/services/experimentListQuery.ts');

class ListQuery {
    constructor(db) { this.db = db; this.conditions = []; this.parameters = {}; this.orders = []; this.offset = 0; this.limit = 30; }
    andWhere(condition, parameters) { this.conditions.push(condition); Object.assign(this.parameters, parameters); return this; }
    orderBy(field, direction) { this.orders = [`${field} ${direction}`]; return this; }
    addOrderBy(field, direction) { this.orders.push(`${field} ${direction}`); return this; }
    skip(value) { this.offset = value; return this; }
    take(value) { this.limit = value; return this; }
    async getManyAndCount() {
        const from = `FROM experiment experiment WHERE ${this.conditions.join(' AND ')}`;
        const total = this.db.prepare(`SELECT COUNT(*) AS total ${from}`).get(this.parameters).total;
        const rows = this.db.prepare(`SELECT * ${from} ORDER BY ${this.orders.join(', ')} LIMIT :pageLimit OFFSET :pageOffset`)
            .all({ ...this.parameters, pageLimit: this.limit, pageOffset: this.offset });
        return [rows.map(row => ({ ...row, spec: JSON.parse(row.spec) })), total];
    }
}

function createListDatabase() {
    const db = new DatabaseSync(':memory:');
    db.exec(`CREATE TABLE experiment(id TEXT PRIMARY KEY, ownerUserId TEXT, name TEXT, status TEXT,
        project TEXT, conversationRunId TEXT, spec TEXT, createdAt TEXT);
        CREATE TABLE experiment_task(id TEXT PRIMARY KEY, experimentId TEXT, kind TEXT);`);
    const insert = db.prepare('INSERT INTO experiment VALUES (?, ?, ?, ?, ?, ?, ?, ?)');
    const task = db.prepare('INSERT INTO experiment_task VALUES (?, ?, ?)');
    const add = (id, name, kinds = ['train'], extra = {}) => {
        insert.run(id, extra.ownerUserId || 'owner', name, extra.status || 'succeeded', extra.project || 'alpha',
            extra.conversationRunId || 'run-a', JSON.stringify(extra.spec || {}), extra.createdAt || '2026-10-14T08:00:00.000Z');
        kinds.forEach((kind, index) => task.run(`${id}-task-${index}`, id, kind));
    };
    for (let index = 0; index < 40; index++) add(`exp-${String(index).padStart(2, '0')}`, `training ${index}`);
    add('late', 'Needle experiment', ['train'], { createdAt: '2026-10-13T08:00:00.000Z' });
    add('literal', 'Accuracy 100%_!'); add('similar', 'Accuracy 100AA'); add('injection', "x' OR 1=1 --");
    add('chinese', '医疗评测', ['evaluate'], { status: 'failed', project: 'beta', conversationRunId: 'run-b' });
    add('draft', 'evaluation draft', [], { status: 'draft', spec: { kind: 'evaluate' } });
    add('multi', 'many tasks', ['train', 'train', 'evaluate']); add('legacy', 'historical record', []);
    add('boundary', 'midnight', ['train'], { createdAt: '2026-10-15T00:00:00.000Z' });
    add('foreign-late', 'NEEDLE experiment', ['train'], { ownerUserId: 'other' });
    add('foreign-literal', 'Accuracy 100%_!', ['train'], { ownerUserId: 'other' });
    add('foreign-chinese', '医疗评测', ['evaluate'], { ownerUserId: 'other', status: 'failed', project: 'beta', conversationRunId: 'run-b' });
    return db;
}

const list = (db, filters = {}, offset = 0, limit = 30) => applyExperimentListFilters(new ListQuery(db), 'owner', filters)
    .orderBy('experiment.createdAt', 'DESC').addOrderBy('experiment.id', 'DESC').skip(offset).take(limit).getManyAndCount();
const check = fn => async () => { const db = createListDatabase(); try { await fn(db); } finally { db.close(); } };

test('search finds records beyond the first page and keeps all other users private', check(async db => {
    const [first, total] = await list(db);
    assert.equal(total, 49); assert.equal(first.length, 30); assert.ok(!first.some(row => row.id === 'late'));
    for (const search of ['  NEEDLE ', 'LATE']) {
        const [items, count] = await list(db, { search });
        assert.deepEqual(items.map(row => row.id), ['late']); assert.equal(count, 1);
    }
    const [second] = await list(db, {}, 30);
    assert.equal(second.length, 19);
    assert.equal(new Set([...first, ...second].map(row => row.id)).size, 49);
}));

test('wildcards and SQL-looking text are literal search content', check(async db => {
    for (const [search, expected] of [['100%_!', 'literal'], ["x' OR 1=1 --", 'injection'], ['医疗', 'chinese']]) {
        const [items, total] = await list(db, { search });
        assert.deepEqual(items.map(row => row.id), [expected]); assert.equal(total, 1);
    }
}));

test('matching multiple tasks returns an experiment once and includes unlaunched draft types', check(async db => {
    const [items, total] = await list(db, { taskKind: 'evaluate' });
    assert.deepEqual(items.map(row => row.id).sort(), ['chinese', 'draft', 'multi']); assert.equal(total, 3);
    const [filtered, count] = await list(db, { taskKind: 'evaluate', status: 'failed', project: ' beta ', conversationRunId: 'run-b', search: '医疗' });
    assert.deepEqual(filtered.map(row => row.id), ['chinese']); assert.equal(count, 1);
    assert.equal((await list(db, { taskKind: 'train', status: 'failed' }))[1], 0);
}));

test('date filters include the start and exclude the next midnight after the end day', check(async db => {
    const [items, total] = await list(db, { createdFrom: '2026-10-14T00:00:00Z', createdBefore: '2026-10-15T00:00:00Z' }, 0, 100);
    assert.equal(total, 47); assert.ok(items.every(row => row.id !== 'late' && row.id !== 'boundary'));
    assert.deepEqual((await list(db, { createdFrom: '2026-10-15T00:00:00Z' }))[0].map(row => row.id), ['boundary']);
    assert.deepEqual((await list(db, { createdBefore: '2026-10-14T00:00:00Z' }))[0].map(row => row.id), ['late']);
    const [empty, count] = await list(db, { createdBefore: '2026-10-13T00:00:00Z' });
    assert.deepEqual(empty, []); assert.equal(count, 0);
}));

test('filtered count and page contents use identical predicates and stable ordering', check(async db => {
    const all = (await list(db, { taskKind: 'train', project: 'alpha' }, 0, 100))[0];
    const [first, total] = await list(db, { taskKind: 'train', project: 'alpha' }, 0, 30);
    const [second, secondTotal] = await list(db, { taskKind: 'train', project: 'alpha' }, 30, 30);
    assert.equal(total, all.length); assert.equal(secondTotal, total);
    assert.deepEqual([...first, ...second], all);
    assert.ok(all.every(row => row.ownerUserId === 'owner'));
    assert.equal(all.filter(row => row.id === 'multi').length, 1);
}));

module.exports = { ListQuery, createListDatabase };
