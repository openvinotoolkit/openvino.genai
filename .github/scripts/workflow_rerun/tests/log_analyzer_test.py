# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
LogAnalyzer tests
"""

import shutil
import tempfile
import unittest
from pathlib import Path


from workflow_rerun.log_analyzer import LogAnalyzer


class LogAnalyzerTest(unittest.TestCase):
    """
    A class for testing LogAnalyzer
    """

    def setUp(self) -> None:
        print(f'\nIn test: "{self._testMethodName}"', flush=True)
        self._cwd = Path(__file__).parent
        self.logs_dir_with_error = self._cwd.joinpath("data").joinpath("logs_with_error")
        self.logs_dir_wo_error = self._cwd.joinpath("data").joinpath("logs_wo_error")
        self.errors_to_look_for_file = self._cwd.parent.joinpath("errors_to_look_for.json")

    def test_log_analyzer_instantiation(self) -> None:
        """
        Ensure LogAnalyzer is instantiated correctly.
        """
        analyzer = LogAnalyzer(
            path_to_logs=self.logs_dir_wo_error,
            path_to_errors_file=self.errors_to_look_for_file,
        )
        self.assertTrue(
            hasattr(analyzer, "_errors_to_look_for"),
            "Analyzer should have _errors_to_look_for",
        )
        self.assertTrue(hasattr(analyzer, "_log_files"), "Analyzer should have _log_files")

        for error_data in analyzer._errors_to_look_for:
            self.assertTrue(error_data["error_text"], "Each error_data should have text")
            self.assertTrue(error_data["ticket"], "Each error_data should have ticket")
            self.assertTrue(error_data["category"], "Each error_data should have category")

        for log_file in analyzer._log_files:
            self.assertTrue(log_file["file_name"], "Each log_file should have file_name")
            self.assertTrue(log_file["path"], "Each log_file should have path")

    def test_string_cleanup(self) -> None:
        """
        Ensure log cleanup function returns correct results
        """
        analyzer = LogAnalyzer(
            path_to_logs=self.logs_dir_wo_error,
            path_to_errors_file=self.errors_to_look_for_file,
        )

        data = (
            "Connection was reset",
            "Failed to connect to github.com",
            "Could not resolve host: github.com",
        )
        expected = (
            "connection was reset",
            "failed to connect to github com",
            "could not resolve host github com",
        )

        for input_str, expected_str in zip(data, expected):
            self.assertEqual(analyzer._clean_up_string(string=input_str), expected_str)

    def test_analyzer_with_error(self) -> None:
        """
        Ensure LogAnalyzer can find an error
        """
        analyzer = LogAnalyzer(
            path_to_logs=self.logs_dir_with_error,
            path_to_errors_file=self.errors_to_look_for_file,
        )
        analyzer.analyze()
        self.assertTrue(analyzer.found_matching_error)
        self.assertEqual(analyzer.found_error_ticket, 130955)
        self.assertEqual(analyzer.matched_error_text, "Network is unreachable")
        self.assertEqual(analyzer.matched_category, "Network")

    def test_analyzer_wo_error(self) -> None:
        """
        Ensure LogAnalyzer does not find an error in the log files w/o errors
        """
        analyzer = LogAnalyzer(
            path_to_logs=self.logs_dir_wo_error,
            path_to_errors_file=self.errors_to_look_for_file,
        )
        analyzer.analyze()
        self.assertFalse(analyzer.found_matching_error)

    def _analyze_jobs(self, job_logs: dict[str, dict[str, str]]) -> LogAnalyzer:
        """
        Analyzes logs laid out as `collect_logs_for_run` saves them: {job name: {file name: content}}
        """
        with tempfile.TemporaryDirectory() as logs_dir:
            for job_name, files in job_logs.items():
                job_dir = Path(logs_dir, job_name)
                job_dir.mkdir()
                for file_name, content in files.items():
                    job_dir.joinpath(file_name).write_text(content, encoding="utf-8")

            analyzer = LogAnalyzer(path_to_logs=Path(logs_dir), path_to_errors_file=self.errors_to_look_for_file)
            analyzer.analyze()
            return analyzer

    def test_analyzer_with_error_in_every_job(self) -> None:
        """
        Ensure the error of the first job is reported when every job has a known error
        """
        analyzer = self._analyze_jobs(
            {
                "Job A": {"1_Job A.txt": "OSError: [Errno 101] Network is unreachable\n", "system.txt": "ok\n"},
                "Job B": {"system.txt": "fatal: unable to access: Could not resolve host: github.com\n"},
            }
        )
        self.assertTrue(analyzer.found_matching_error)
        self.assertEqual(analyzer.matched_error_text, "Network is unreachable")
        self.assertEqual(analyzer.unmatched_jobs, [])

    def test_analyzer_with_job_without_error(self) -> None:
        """
        Ensure there is no match if one of the jobs has no known error, e.g. a real test failure
        """
        with tempfile.TemporaryDirectory() as logs_dir:
            shutil.copytree(self.logs_dir_with_error, logs_dir, dirs_exist_ok=True)
            Path(logs_dir, "Tests").mkdir()
            Path(logs_dir, "Tests", "1_Tests.txt").write_text(
                "FAILED tests/test_llm.py::test_generate\n", encoding="utf-8"
            )

            analyzer = LogAnalyzer(path_to_logs=Path(logs_dir), path_to_errors_file=self.errors_to_look_for_file)
            analyzer.analyze()

        self.assertFalse(analyzer.found_matching_error)
        self.assertIsNone(analyzer.found_error_ticket)
        self.assertEqual(analyzer.unmatched_jobs, ["Tests"])

    def test_analyzer_with_job_without_logs(self) -> None:
        """
        Ensure there is no match if the logs of one of the jobs were not found
        """
        analyzer = self._analyze_jobs(
            {"Job A": {"system.txt": "OSError: [Errno 101] Network is unreachable\n"}, "Job B": {}}
        )
        self.assertFalse(analyzer.found_matching_error)
        self.assertEqual(analyzer.unmatched_jobs, ["Job B"])
