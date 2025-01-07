"""Execute a test session."""

from pathlib import Path
from typing import NamedTuple


class TestSetup(NamedTuple):
    name: str
    path: Path


Setup = list[TestSetup]


def run(setup: Setup) -> None:
    """Run the test session."""
    for _test_setup in setup:
        ...
