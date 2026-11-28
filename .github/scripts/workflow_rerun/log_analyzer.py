# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import re
from pathlib import Path
from typing import Optional, TypedDict

from workflow_rerun.constants import LOGGER

# Fallback category recorded when an error entry has no category.
NO_CATEGORY = "NO CATEGORY"


class LogFile(TypedDict):
    file_name: str
    path: Path


class ErrorData(TypedDict):
    error_text: str
    ticket: int
    category: str


class LogAnalyzer:
    def __init__(self, path_to_logs: Path, path_to_errors_file: Path) -> None:
        self._path_to_errors_file = path_to_errors_file

        self._errors_to_look_for: list[ErrorData] = []
        self._collect_errors_to_look_for()

        self._log_dir = path_to_logs

        self._job_log_files: dict[str, list[LogFile]] = {}
        self._log_files: list[LogFile] = []
        self._collect_log_files()
        all_txt_log_files_pretty = "\n".join(map(lambda item: str(item["path"]), self._log_files))
        LOGGER.info(f"ALL .txt LOG FILES: \n{all_txt_log_files_pretty}")

        self.found_matching_error = False
        self.found_error_ticket = None
        self.matched_error_text = None
        self.matched_category = None
        self.unmatched_jobs: list[str] = []

    def _collect_errors_to_look_for(self) -> None:
        with open(file=self._path_to_errors_file, mode="r", encoding="utf-8") as errors_file:
            errors_data = json.load(errors_file)
            for error_data in errors_data:
                self._errors_to_look_for.append(
                    ErrorData(
                        error_text=error_data["error_text"],
                        ticket=error_data["ticket"],
                        category=error_data.get("category", NO_CATEGORY),
                    )
                )

    def _collect_log_files(self) -> None:
        """
        Collects the .txt log files and groups them by job

        Each directory holds the logs of one job, as saved by `collect_logs_for_run`:
            > Job_name_0
                > 1_Job_name_0.txt
                > system.txt
            > Job_name_1
                > system.txt
            ...
        A directory without files is a job whose logs were not found.

        A top-level `<index>_<job name>.txt` file, as in the GitHub Actions pipeline logs archive,
        belongs to the job named without the index prefix.

        We need to only analyze the `*.txt` files
        """

        for _file in sorted(Path(self._log_dir).iterdir()):
            if _file.is_dir():
                job_log_files = self._job_log_files.setdefault(_file.name, [])
                for log_file in sorted(_file.iterdir()):
                    job_log_files.append(LogFile(file_name=log_file.name, path=log_file.resolve()))
            elif _file.suffix == ".txt":
                job_name = re.sub(r"^\d+_", "", _file.stem)
                self._job_log_files.setdefault(job_name, []).append(LogFile(file_name=_file.name, path=_file.resolve()))

        self._log_files = [log_file for job_log_files in self._job_log_files.values() for log_file in job_log_files]

    def _is_error_in_log(self, error_to_look_for: str, log_file_path: Path) -> bool:
        """
        Searches for the error in the provided log
        """

        error_to_look_for = self._clean_up_string(error_to_look_for)

        with open(file=log_file_path, mode="r", encoding="utf-8") as log_file:
            for line in log_file:
                if error_to_look_for in self._clean_up_string(line):
                    return True
        return False

    @staticmethod
    def _clean_up_string(string: str) -> str:
        """
        Replaces special characters with spaces in the string, strips it from leading and following spaces,
        and lowers it

        for "Could not resolve host: github.com" returns "could not resolve host github com"

        This cleanup is applied to both errors to look for and logs themselves for matching
        """
        return re.sub(r"[^A-Za-z0-9]+", " ", string).lower().strip()

    def _find_error_in_job(self, job_name: str, log_files: list[LogFile]) -> Optional[ErrorData]:
        """
        Returns the first known error, in the order of the errors file, found in the logs of the job
        """
        if not log_files:
            LOGGER.info(f'NO LOGS FOUND FOR JOB "{job_name}"')
            return None

        LOGGER.info(f'LOOKING FOR {len(self._errors_to_look_for)} KNOWN ERRORS IN LOGS OF JOB "{job_name}"...')
        for error in self._errors_to_look_for:
            for log_file in log_files:
                if self._is_error_in_log(error_to_look_for=error["error_text"], log_file_path=log_file["path"]):
                    LOGGER.info(
                        f'FOUND "{error["error_text"]}" ERROR IN {log_file["path"]}. '
                        f"TICKET: {error['ticket']}. CATEGORY: {error['category']}"
                    )
                    return error

        LOGGER.info(f'NO KNOWN ERROR FOUND IN LOGS OF JOB "{job_name}"')
        return None

    def analyze(self) -> None:
        """
        Looks for a known error in the logs of every job

        A matching error is reported only if each job has one: a job without a known error
        likely failed for a real reason, which a rerun would not fix.
        The reported error is the one found in the first job.
        """
        if not self._job_log_files:
            LOGGER.info("NO JOB LOGS TO ANALYZE")
            return

        first_error: Optional[ErrorData] = None
        for job_name, log_files in self._job_log_files.items():
            error = self._find_error_in_job(job_name=job_name, log_files=log_files)
            if error is None:
                self.unmatched_jobs.append(job_name)
            elif first_error is None:
                first_error = error

        if self.unmatched_jobs:
            LOGGER.info(f"NO KNOWN ERROR IN {len(self.unmatched_jobs)} JOB(S): {self.unmatched_jobs}")
            return

        self.found_matching_error = True
        self.found_error_ticket = first_error["ticket"]
        self.matched_error_text = first_error["error_text"]
        self.matched_category = first_error["category"]


if __name__ == "__main__":
    # Usage example
    log_analyzer = LogAnalyzer(
        path_to_logs=Path("/tmp/logs_dir"), path_to_errors_file=Path("/tmp/errors_to_look_for.json")
    )
    log_analyzer.analyze()
    if log_analyzer.found_matching_error:
        print("found matching error, see logs above")
