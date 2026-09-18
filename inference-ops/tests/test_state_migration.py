import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "migrate-state.py"
SPEC = importlib.util.spec_from_file_location("inference_ops_migrate_state", SCRIPT)
MIGRATION = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MIGRATION)


class StateMigrationTests(unittest.TestCase):
    def test_dry_run_blocks_active_records(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "legacy"
            target = Path(root) / "target"
            run = source / "services" / "runs" / "run-1"
            run.mkdir(parents=True)
            (run / "meta.json").write_text(
                json.dumps({"instance_id": "run-1", "status": "running"}),
                encoding="utf-8",
            )
            result = MIGRATION.migrate(source, target, apply=False)
            self.assertEqual(result["status"], "blocked")
            self.assertFalse(target.exists())

    def test_dry_run_blocks_active_jsonl_registry_record(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "legacy"
            target = Path(root) / "target"
            source.mkdir()
            (source / "inference_pid_registry.jsonl").write_text(
                json.dumps(
                    {
                        "instance_id": "run-1",
                        "launch_status": "starting",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = MIGRATION.migrate(source, target, apply=False)
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["active_records"][0]["id"], "run-1")

    def test_apply_preserves_history_and_rewrites_state_paths(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "legacy"
            target = Path(root) / "target"
            run = source / "services" / "runs" / "run-1"
            run.mkdir(parents=True)
            old_log = source / "services" / "runs" / "run-1" / "service.log"
            old_log.write_text("done", encoding="utf-8")
            (run / "meta.json").write_text(
                json.dumps(
                    {
                        "instance_id": "run-1",
                        "status": "stopped",
                        "log_dir": str(run),
                    }
                ),
                encoding="utf-8",
            )
            old_ops = str(Path(root) / "repository" / "medflow" / "agent")
            new_ops = str(Path(root) / "repository" / "inference-ops")
            (run / "service.runtime.yaml").write_text(
                f"ENV:\n  START_SCRIPT: {old_ops}/scripts/start-service.sh\n",
                encoding="utf-8",
            )
            (source / "inference_pid_registry.jsonl").write_text(
                json.dumps(
                    {
                        "instance_id": "run-1",
                        "status": "stopped",
                        "log_dir": str(run),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = MIGRATION.migrate(
                source,
                target,
                apply=True,
                extra_rewrites=[(old_ops, new_ops)],
            )
            self.assertEqual(result["status"], "migrated")
            migrated = json.loads(
                (target / "services" / "runs" / "run-1" / "meta.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(migrated["schema_version"], 1)
            self.assertEqual(migrated["provider"], "medflow")
            self.assertTrue(migrated["log_dir"].startswith(str(target)))
            self.assertEqual(
                (target / "services" / "runs" / "run-1" / "service.log").read_text(
                    encoding="utf-8"
                ),
                "done",
            )
            migrated_yaml = (
                target / "services" / "runs" / "run-1" / "service.runtime.yaml"
            ).read_text(encoding="utf-8")
            self.assertIn(new_ops, migrated_yaml)
            self.assertNotIn(old_ops, migrated_yaml)
            migrated_registry = json.loads(
                (target / "inference_pid_registry.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(migrated_registry["schema_version"], 1)
            self.assertEqual(migrated_registry["provider"], "medflow")
            self.assertTrue(migrated_registry["log_dir"].startswith(str(target)))


if __name__ == "__main__":
    unittest.main()
