import unittest

from bim_agents.claude_runtime import (
    ModelAccessError, ModelRateLimitError, RunContextWrapper, _error_from_status,
    _strict_output_schema, function_tool,
)
from bim_agents.models import EvidenceWorkstreamResult


class ClaudeRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_function_tool_injects_context_and_validates_nested_models(self):
        @function_tool
        def sample(ctx: RunContextWrapper[dict], result: EvidenceWorkstreamResult) -> str:
            return f"{ctx.context['scope']}:{result.package_id}"

        output = await sample.invoke(
            {"scope": "authorized"},
            {
                "result": {
                    "status": "unsupported",
                    "package_id": "p",
                    "limitations": ["No governed route."],
                }
            },
        )

        self.assertEqual(output, "authorized:p")
        self.assertNotIn("ctx", sample.params_json_schema["properties"])

    def test_structured_output_schema_has_no_optional_object_fields(self):
        schema = _strict_output_schema(EvidenceWorkstreamResult.model_json_schema())

        def inspect(value):
            if isinstance(value, dict):
                if value.get("type") == "object" and "properties" in value:
                    self.assertEqual(
                        set(value["required"]), set(value["properties"])
                    )
                    self.assertFalse(value["additionalProperties"])
                for nested in value.values():
                    inspect(nested)
            elif isinstance(value, list):
                for nested in value:
                    inspect(nested)

        inspect(schema)

    def test_anthropic_failures_are_classified_by_status_and_message(self):
        self.assertIsInstance(
            _error_from_status(None, "Credit balance is too low"), ModelAccessError
        )
        self.assertIsInstance(
            _error_from_status(429, "Too many requests"), ModelRateLimitError
        )


if __name__ == "__main__":
    unittest.main()
