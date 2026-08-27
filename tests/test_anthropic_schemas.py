from bim_agents.anthropic_agent import (
    ANALYSIS_TOOL,
    ANSWER_TOOL,
    AUDIT_TOOL,
    INVESTIGATION_TOOLS,
    PLAN_TOOL,
    VERIFICATION_TOOL,
)


UNSUPPORTED_STRICT_KEYWORDS = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
}


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def test_strict_tool_schemas_avoid_unsupported_constraint_keywords():
    tools = [ANALYSIS_TOOL, PLAN_TOOL, *INVESTIGATION_TOOLS, AUDIT_TOOL,
             VERIFICATION_TOOL, ANSWER_TOOL]
    for tool in tools:
        for node in walk(tool["input_schema"]):
            assert not (set(node) & UNSUPPORTED_STRICT_KEYWORDS), tool["name"]
