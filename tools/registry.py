from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


LOGGER = logging.getLogger(__name__)

ToolHandler = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class ToolDefinition:
    """Metadata and callable implementation for one registered tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def to_schema(self) -> dict[str, Any]:
        """Return an OpenAI-compatible function tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Registry for independent tools and their OpenAI-compatible schemas."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition | Mapping[str, Any]) -> None:
        """Register a tool, rejecting malformed or duplicate definitions."""
        if isinstance(tool, Mapping):
            tool = ToolDefinition(
                name=tool["name"],
                description=tool["description"],
                parameters=tool["parameters"],
                handler=tool["handler"],
            )

        if not isinstance(tool, ToolDefinition):
            raise TypeError("tool must be a ToolDefinition or mapping")
        if not tool.name:
            raise ValueError("tool name cannot be empty")
        if not tool.description:
            raise ValueError(f"description is required for tool '{tool.name}'")
        if not isinstance(tool.parameters, dict):
            raise TypeError(f"parameters for '{tool.name}' must be a dict")
        if not callable(tool.handler):
            raise TypeError(f"handler for '{tool.name}' must be callable")
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' is already registered")

        self._tools[tool.name] = tool
        LOGGER.info("Registered tool: %s", tool.name)

    def get(self, name: str) -> ToolDefinition | None:
        """Return a registered tool by name, or None when it is absent."""
        tool = self._tools.get(name)
        LOGGER.debug("Lookup tool: name=%s found=%s", name, tool is not None)
        return tool

    def list_tools(self) -> list[dict[str, Any]]:
        """Return all registered tools in OpenAI function-tool format."""
        tools = [tool.to_schema() for tool in self._tools.values()]
        LOGGER.debug("Listed tools: count=%d", len(tools))
        return tools


def create_default_registry() -> ToolRegistry:
    """Create a registry containing the built-in filesystem and test tools."""
    from .file_tool import FILE_TOOLS
    from .shell_tool import SHELL_TOOLS
    from .test_tool import TEST_TOOLS

    registry = ToolRegistry()
    for tool in (*FILE_TOOLS, *SHELL_TOOLS, *TEST_TOOLS):
        registry.register(tool)
    return registry
