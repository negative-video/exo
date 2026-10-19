"""Logged exceptions must not carry request content."""

import sys
from pathlib import Path

from loguru import logger

from exo.shared.logging import logger_cleanup, logger_setup

MARKER = "marker-4b1d2a"


def test_logged_exception_has_no_variable_values(tmp_path: Path) -> None:
    log_file = tmp_path / "exo.log"
    logger_setup(log_file)
    try:
        try:
            prompt = MARKER
            raise ValueError(f"prompt of {len(prompt)} characters rejected")
        except ValueError as e:
            logger.opt(exception=e).warning("request failed")
        logger_cleanup()

        logged = log_file.read_text()
        assert "request failed" in logged
        assert "ValueError" in logged
        assert MARKER not in logged
    finally:
        logger.remove()
        logger.add(sys.stderr)
