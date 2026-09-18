import sys
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from inference_ops.benchmark_eta import (  # noqa: E402
    benchmark_eta_fields,
    benchmark_log_progress,
    format_benchmark_eta,
    medbench_sample_counts,
)


class BenchmarkEtaTests(unittest.TestCase):
    def test_reads_runner_progress_before_result_is_saved(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            log = Path(directory) / "run.log"
            log.write_text("[1/500] done\n[7/500] done\n", encoding="utf-8")
            self.assertEqual(benchmark_log_progress(log), (7, 500))
            self.assertEqual(benchmark_log_progress(Path(directory) / "missing.log"), (None, None))

    def test_calculates_remaining_time_from_cumulative_throughput(self):
        eta = benchmark_eta_fields("running", "100", 3, 10, now=1000)
        self.assertEqual(eta["eta_status"], "available")
        self.assertEqual(eta["estimated_remaining_seconds"], 2100)
        self.assertEqual(eta["eta_calculated_at"], 1000)

    def test_waits_for_first_completed_sample(self):
        eta = benchmark_eta_fields("running", "100", 0, 10, now=1000)
        self.assertEqual(eta["eta_status"], "calculating")
        self.assertNotIn("estimated_remaining_seconds", eta)

    def test_marks_finished_and_failed_jobs_without_a_running_eta(self):
        finished = benchmark_eta_fields("finished", "invalid", None, None, now=1000)
        failed = benchmark_eta_fields("failed", "100", 4, 10, now=1000)
        self.assertEqual(finished["eta_status"], "complete")
        self.assertEqual(finished["estimated_remaining_seconds"], 0)
        self.assertEqual(failed["eta_status"], "unavailable")

    def test_rejects_invalid_start_time_after_progress_exists(self):
        eta = benchmark_eta_fields("running", "invalid", 1, 10, now=1000)
        self.assertEqual(eta["eta_status"], "unavailable")

    def test_aggregates_medbench_file_progress(self):
        done, total = medbench_sample_counts(
            {
                "files": {
                    "a.jsonl": {"done": 3, "total": 10},
                    "b.jsonl": {"done": 7, "total": 20},
                    "ignored": "invalid",
                }
            }
        )
        self.assertEqual((done, total), (10, 30))

    def test_formats_compact_chinese_durations(self):
        self.assertEqual(format_benchmark_eta(59), "不足1分钟")
        self.assertEqual(format_benchmark_eta(120), "约2分钟")
        self.assertEqual(format_benchmark_eta(3720), "约1小时2分钟")
        self.assertEqual(format_benchmark_eta(90000), "约1天1小时")


if __name__ == "__main__":
    unittest.main()
