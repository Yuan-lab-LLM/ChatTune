#!/usr/bin/env node
const path = require("path");
const Database = require("better-sqlite3");

const args = process.argv.slice(2);
const usage = () => {
  console.error("Usage: node scripts/cleanup-runtime-runs.cjs --database /path/database.sqlite --run-id runtime-node-a [--run-id runtime-node-b]");
  console.error("       node scripts/cleanup-runtime-runs.cjs --database /path/database.sqlite --all-runtime-runs");
  process.exit(2);
};

let databasePath = process.env.MEDFLOW_DATABASE_PATH || "";
const runIds = [];
let allRuntimeRuns = false;
for (let i = 0; i < args.length; i += 1) {
  const arg = args[i];
  if (arg === "--database") {
    databasePath = args[++i] || "";
  } else if (arg === "--run-id") {
    const runId = args[++i] || "";
    if (runId) runIds.push(runId);
  } else if (arg === "--all-runtime-runs") {
    allRuntimeRuns = true;
  } else {
    usage();
  }
}
if (!databasePath || (!runIds.length && !allRuntimeRuns)) usage();

const db = new Database(path.resolve(databasePath));
const selectSql = allRuntimeRuns
  ? "SELECT id, project, status, nodeId FROM run_table WHERE id LIKE 'runtime-%'"
  : `SELECT id, project, status, nodeId FROM run_table WHERE id IN (${runIds.map(() => "?").join(",")})`;
const targets = allRuntimeRuns ? db.prepare(selectSql).all() : db.prepare(selectSql).all(...runIds);
console.log("target runs:", targets);
if (!targets.length) {
  db.close();
  process.exit(0);
}

const ids = targets.map((run) => run.id);
const placeholders = ids.map(() => "?").join(",");
const tx = db.transaction(() => {
  for (const table of ["message_table", "reply_table", "input_request_table"]) {
    db.prepare(`DELETE FROM ${table} WHERE run_id IN (${placeholders})`).run(...ids);
  }
  db.prepare(`DELETE FROM span_table WHERE conversationId IN (${placeholders})`).run(...ids);
  const deleted = db.prepare(`DELETE FROM run_table WHERE id IN (${placeholders})`).run(...ids);
  console.log("deleted runs:", deleted.changes);
});
tx();
db.close();
