from typing import Any

tool_call_start: str
tool_call_end: str

def parse_tool_call(model_output: str, tools: Any | None = None) -> dict[str, Any]: ...
