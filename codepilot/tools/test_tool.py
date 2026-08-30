from __future__ import annotations

import sys
import logging
from typing import Any

from .shell_tool import run_command
from .registry import ToolDefinition


LOGGER = logging.getLogger(__name__)


def run_tests() -> dict[str, Any]:
    """Run the workspace test suite with pytest."""
    command = f"{sys.executable} -m pytest -q"
    LOGGER.info("run_tests started: command=%s", command)
    result = run_command(command)
    LOGGER.info(
        "run_tests completed: success=%s return_code=%s",
        result["success"],
        result.get("return_code", -1),
    )
    return result


RUN_TESTS_TOOL = ToolDefinition(
    name="run_tests",
    description="Run the pytest test suite in the workspace.",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    handler=run_tests,
)

TEST_TOOLS = (RUN_TESTS_TOOL,)
