# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
pytest configuration for module_genai Python tests.
"""

import pytest
import sys
from pathlib import Path


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the path to test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def models_dir():
    """Return the path to models directory (if available)."""
    # Try common locations
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / "models",
        Path("./models"),
        Path.home() / "models",
    ]
    for path in possible_paths:
        if path.exists():
            return path
    return None
