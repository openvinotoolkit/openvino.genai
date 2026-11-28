# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
log collector tests
"""

import os
import unittest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

import requests
from github import Github, Auth
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from workflow_rerun.log_collector import collect_logs_for_run
from workflow_rerun.tests.gh_test_utils import find_failed_run_with_logs


class LogCollectorTest(unittest.TestCase):
    """
    A class for testing log collection
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.session = requests.Session()
        retry_strategy = Retry(total=5, backoff_factor=3, backoff_jitter=1, status_forcelist=[429, 500, 502, 503, 504])
        cls.session.mount("https://github.com", HTTPAdapter(max_retries=retry_strategy))

        cls.github = Github(auth=Auth.Token(token=os.environ.get("GITHUB_TOKEN")))
        gh_repo = cls.github.get_repo(full_name_or_id="openvinotoolkit/openvino.genai")

        # Looking for reference workflow runs.
        # Their "created_at" time should be within 60 days - the log retention window
        oldest_allowed_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        cls.successful_workflow_run = gh_repo.get_workflow_runs(status="success", created=f">={oldest_allowed_date}")[0]
        print(f"Successful workflow run for testing: {cls.successful_workflow_run}", flush=True)

        cls.failed_workflow_run = find_failed_run_with_logs(
            gh_repo=gh_repo, created_filter=f">={oldest_allowed_date}", max_candidates=50
        )
        print(f"Failed workflow run for testing: {cls.failed_workflow_run}", flush=True)

    def setUp(self):
        print(f'\nIn test: "{self._testMethodName}"', flush=True)

    def test_failed_logs_are_collected(self) -> None:
        """
        Ensure only logs for failed jobs are collected
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir)
            collect_logs_for_run(run=self.failed_workflow_run, logs_dir=logs_dir, session=self.session)
            self.assertTrue(any(logs_dir.iterdir()), "Logs directory should not be empty for failed runs")

    def test_successful_logs_are_not_collected(self) -> None:
        """
        Ensure logs for successful jobs are not collected
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir)
            collect_logs_for_run(run=self.successful_workflow_run, logs_dir=logs_dir, session=self.session)
            self.assertFalse(any(logs_dir.iterdir()), "Logs directory should be empty for successful runs")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.github.close()


class LogCollectorOfflineTest(unittest.TestCase):
    """
    A class for testing log collection from a local log archive
    """

    def test_failed_job_logs_are_saved_per_job(self) -> None:
        """
        Ensure each failed job gets its own logs only, and the status job is skipped
        """
        jobs = []
        for name, conclusion in (
            ("Samples LLM (Release)", "failure"),
            ("LLM (Release)", "failure"),
            ("Tests / Tests", "failure"),
            ("Build", "success"),
            ("ci/gha_overall_status_linux", "failure"),
        ):
            job = MagicMock(conclusion=conclusion)
            job.name = name
            jobs.append(job)
        run = MagicMock(id=1)
        run.jobs.return_value = jobs

        archive_entries = {
            "1_Samples LLM (Release).txt": "samples llm",
            "Samples LLM (Release)/system.txt": "samples llm system",
            "Samples LLM (Release)/3_Run samples.txt": "samples llm step",
            "2_LLM (Release).txt": "llm",
            "LLM (Release)/system.txt": "llm system",
            "3_Build.txt": "build",
            "Build/system.txt": "build system",
            "0_ci_gha_overall_status_linux.txt": "status",
            "ci_gha_overall_status_linux/system.txt": "status system",
        }

        def fake_download(run, log_archive_path: Path, session) -> None:
            with ZipFile(log_archive_path, mode="w") as zip_file:
                for name, content in archive_entries.items():
                    zip_file.writestr(name, content)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("workflow_rerun.log_collector._download_logs_archive", side_effect=fake_download):
                logs_dir = Path(temp_dir)
                collect_logs_for_run(run=run, logs_dir=logs_dir, session=MagicMock())
                collected = {str(p.relative_to(logs_dir)) for p in logs_dir.rglob("*")}

        self.assertEqual(
            collected,
            {
                "Samples LLM (Release)",
                "Samples LLM (Release)/1_Samples LLM (Release).txt",
                "Samples LLM (Release)/system.txt",
                "LLM (Release)",
                "LLM (Release)/2_LLM (Release).txt",
                "LLM (Release)/system.txt",
                # No logs in the archive, the empty directory marks the job as not analyzable
                "Tests _ Tests",
            },
        )
