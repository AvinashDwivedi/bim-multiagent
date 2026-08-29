from __future__ import annotations

import json
from typing import Any

from .project_tools import RawProjectTools
from .pricing import response_usage_record


UNVERIFIED_STANDARDS_DISCLAIMER = (
    "⚠️ Unverified — not sourced from project data or a cited standard."
)


class ModelAssistedTools:
    """Optional model-backed tools selected by the primary agent during execution."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        reasoning_effort: str,
        project_tools: RawProjectTools,
    ):
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.project_tools = project_tools
        self._usage_records: list[dict[str, Any]] = []
        self.prompt_cache_key = _prompt_cache_key(model, project_tools)

    def reset_usage(self) -> None:
        self._usage_records = []

    def usage_records(self) -> list[dict[str, Any]]:
        return list(self._usage_records)

    def _record_usage(
        self, response: Any, purpose: str, *, excluded_tool_fees: list[str] | None = None,
    ) -> None:
        self._usage_records.append(response_usage_record(
            response,
            configured_model=self.model,
            purpose=purpose,
            excluded_tool_fees=excluded_tool_fees,
        ))

    def definitions(self) -> list[dict[str, Any]]:
        return [
            _strict_tool(
                "explore_object_scope",
                "Ask a model-backed scope explorer to inspect the complete project hierarchy and propose multiple candidate object populations for an ambiguous concept. It returns candidate root IDs, descendant/leaf evidence, inclusions, exclusions, alternatives, and unresolved ambiguity. Use before counting when category names, translations, instance depth, or scope boundaries are uncertain; the primary agent chooses the final scope.",
                {
                    "question": {"type": "string"},
                    "candidate_concepts": {"type": "array", "items": {"type": "string"}, "maxItems": 30},
                    "ambiguity_notes": {"type": "string"},
                },
            ),
            _strict_tool(
                "review_scope_and_evidence",
                "Ask a model-backed critic to challenge a proposed BIM answer without using expected answers. It checks alternative interpretations, omitted populations, unsupported exclusions, property-versus-geometry conflicts, missing counterexamples, arithmetic reconciliation, and whether clarification is truly necessary. Returns advice only; the primary agent decides whether to investigate further or finish.",
                {
                    "question": {"type": "string"},
                    "selected_scope": {"type": "string"},
                    "exclusions": {"type": "string"},
                    "evidence": {"type": "string"},
                    "reconciliation": {"type": "string"},
                    "draft_answer": {"type": "string"},
                },
            ),
            _strict_tool(
                "research_standards",
                "Use model-directed web research for a normative BIM, engineering, safety, or code-compliance question. Returns a concise standards summary with source URLs, jurisdiction/edition uncertainty, measurable criteria, and data required for comparison. If no source URL is retrieved, the result is marked unverified and carries a mandatory final-answer disclaimer. Use only when external requirements are necessary; project facts must still come from project tools.",
                {
                    "query": {"type": "string"},
                    "jurisdiction": {"type": "string"},
                    "standard_hint": {"type": "string"},
                },
            ),
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "explore_object_scope":
            return self._explore_scope(arguments)
        if name == "review_scope_and_evidence":
            return self._review(arguments)
        if name == "research_standards":
            return self._research_standards(arguments)
        raise ValueError(f"Unknown model-assisted tool: {name}")

    def _explore_scope(self, arguments: dict[str, Any]) -> dict[str, Any]:
        profile = self.project_tools.scope_profile(
            max_nodes=40,
            sample_limit=3,
            terms=[str(item) for item in arguments.get("candidate_concepts", [])],
        )
        payload = {
            "question": arguments.get("question", ""),
            "candidate_concepts": arguments.get("candidate_concepts", []),
            "ambiguity_notes": arguments.get("ambiguity_notes", ""),
            "project_hierarchy_profile": profile,
        }
        response = self.client.responses.create(
            model=self.model,
            instructions=(
                "You are a BIM scope exploration tool, not the final answerer. Interpret the question against "
                "the supplied raw hierarchy profile. Propose all materially plausible scopes rather than one "
                "hard-coded taxonomy. For each scope, give root object IDs, descendant/leaf evidence, inclusions, "
                "exclusions, possible false positives, and the next model-authored SQL/geometry check. Never use "
                "fixed path depth as an instance rule. Be concise and do not expose private chain-of-thought."
            ),
            input=json.dumps(payload, ensure_ascii=False),
            reasoning={"effort": self.reasoning_effort},
            store=False,
            max_output_tokens=1400,
            prompt_cache_key=f"{self.prompt_cache_key}-scope"[:64],
        )
        self._record_usage(response, "scope_exploration")
        analysis = _required_model_text(response, "scope explorer")
        return {
            "analysis": analysis,
            "response_id": str(getattr(response, "id", "") or ""),
            "profile_summary": {
                "record_count": profile["record_count"],
                "tree_node_count": profile["tree_node_count"],
                "hierarchy_nodes_supplied": len(profile["hierarchy_nodes"]),
                "truncated": profile["truncated"],
            },
        }

    def _review(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = {
            key: str(arguments.get(key, ""))[:30_000]
            for key in (
                "question", "selected_scope", "exclusions", "evidence", "reconciliation", "draft_answer",
            )
        }
        profile = self.project_tools.scope_profile(max_nodes=0, sample_limit=0)
        payload["project_summary"] = {
            key: value for key, value in profile.items() if key != "hierarchy_nodes"
        }
        response = self.client.responses.create(
            model=self.model,
            instructions=(
                "You are a critical BIM evidence-review tool, not the final answerer and not an answer-key judge. "
                "Challenge scope and evidence using only the supplied material. Look for alternative meanings, "
                "omitted categories or descendants, type/instance duplication, identity mismatches, unsupported "
                "exclusions, property-versus-geometry conflicts, unreconciled totals, merged dimensional axes, "
                "missing counterexamples, and relationship directionality. Reject connectivity claims based only "
                "on raw references, and reject rankings that did not compare the complete selected population. "
                "Flag conflicting values for the same resolved identity across tree/properties/IFC and do not "
                "silently select a winner. Check whether revision/export metadata establishes that the supplied "
                "files are synchronized; otherwise require snapshot-scoped wording. "
                "Compare relevant raw-record, "
                "external-ID, IFC-ID, and geometry cardinalities; repeated values are not proof of duplication. "
                "Return prioritized gaps and concrete follow-up tool "
                "checks. If the evidence is sufficient, say so. Do not expose private chain-of-thought."
            ),
            input=json.dumps(payload, ensure_ascii=False),
            reasoning={"effort": self.reasoning_effort},
            store=False,
            max_output_tokens=1200,
            prompt_cache_key=f"{self.prompt_cache_key}-review"[:64],
        )
        self._record_usage(response, "evidence_review")
        review = _required_model_text(response, "evidence reviewer")
        return {
            "review": review,
            "response_id": str(getattr(response, "id", "") or ""),
        }

    def _research_standards(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = {
            "query": arguments.get("query", ""),
            "jurisdiction": arguments.get("jurisdiction", ""),
            "standard_hint": arguments.get("standard_hint", ""),
        }
        response = self.client.responses.create(
            model=self.model,
            instructions=(
                "Research the applicable engineering or safety requirement using authoritative primary sources. "
                "State jurisdiction and edition uncertainty, measurable criteria, and the project data needed to "
                "test compliance. Include source links. Do not make any claim about the user's project itself."
            ),
            input=json.dumps(query, ensure_ascii=False),
            tools=[{"type": "web_search"}],
            reasoning={"effort": self.reasoning_effort},
            store=False,
            max_output_tokens=3000,
            prompt_cache_key=f"{self.prompt_cache_key}-standards"[:64],
        )
        self._record_usage(response, "standards_research", excluded_tool_fees=["web_search"])
        research = _required_model_text(response, "standards researcher")
        sources = _response_urls(response)
        verified = bool(sources)
        if not verified and UNVERIFIED_STANDARDS_DISCLAIMER not in research:
            research = f"{UNVERIFIED_STANDARDS_DISCLAIMER}\n{research}"
        return {
            "research": research,
            "sources": sources,
            "evidence_class": "external_standard" if verified else "unverified_external_guidance",
            "verified": verified,
            "mandatory_disclaimer": None if verified else UNVERIFIED_STANDARDS_DISCLAIMER,
            "response_id": str(getattr(response, "id", "") or ""),
        }


def _strict_tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
        "strict": True,
    }


def _prompt_cache_key(model: str, project_tools: RawProjectTools) -> str:
    import hashlib

    digest = hashlib.sha256()
    for item in project_tools.manifest():
        digest.update(str(item.get("sha256", "")).encode("ascii", errors="ignore"))
    return f"bim-model-tools-{model}-{digest.hexdigest()[:16]}"[:48]


def _required_model_text(response: Any, tool_label: str) -> str:
    value = str(getattr(response, "output_text", "") or "").strip()
    if not value:
        raise RuntimeError(f"The model-backed {tool_label} returned no analysis.")
    return value


def _response_urls(response: Any) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in list(getattr(response, "output", []) or []):
        for content in list(getattr(item, "content", []) or []):
            for annotation in list(getattr(content, "annotations", []) or []):
                url = str(getattr(annotation, "url", "") or "")
                if not url or url in seen:
                    continue
                seen.add(url)
                output.append({
                    "url": url,
                    "title": str(getattr(annotation, "title", "") or ""),
                })
    return output
