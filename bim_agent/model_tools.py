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
        self._last_review_arguments: dict[str, str] | None = None
        self._last_review_result: dict[str, Any] | None = None
        self.prompt_cache_key = _prompt_cache_key(model, project_tools)

    def reset_usage(self) -> None:
        self._usage_records = []
        self._last_review_arguments = None
        self._last_review_result = None

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
                "review_scope_and_evidence",
                "Ask a model-backed critic to challenge a proposed BIM answer without using expected answers. It checks alternative interpretations, omitted populations, unsupported exclusions, property-versus-geometry conflicts, missing counterexamples, arithmetic reconciliation, and whether clarification is truly necessary. Returns an enforced structured finalize decision, follow-ups, disclosure codes, and unsupported claims.",
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
        if name == "review_scope_and_evidence":
            return self._review(arguments)
        if name == "research_standards":
            return self._research_standards(arguments)
        raise ValueError(f"Unknown model-assisted tool: {name}")

    def _review(self, arguments: dict[str, Any]) -> dict[str, Any]:
        current_arguments = {
            key: str(arguments.get(key, ""))[:10_000]
            for key in (
                "question", "selected_scope", "exclusions", "evidence", "reconciliation", "draft_answer",
            )
        }
        incremental = bool(
            self._last_review_arguments
            and self._last_review_result
            and current_arguments["question"] == self._last_review_arguments["question"]
            and current_arguments["selected_scope"] == self._last_review_arguments["selected_scope"]
        )
        changed_inputs = {
            key: value
            for key, value in current_arguments.items()
            if not self._last_review_arguments or value != self._last_review_arguments.get(key)
        }
        if incremental:
            payload: dict[str, Any] = {
                "review_mode": "delta",
                "prior_review": {
                    key: value
                    for key, value in self._last_review_result.items()
                    if key not in {"response_id", "required_disclosure_text"}
                },
                "changed_inputs": changed_inputs,
            }
        else:
            payload = dict(current_arguments)
            profile = self.project_tools.scope_profile(max_nodes=0, sample_limit=0)
            payload["review_mode"] = "full"
            payload["project_summary"] = {
                "record_count": profile["record_count"],
                "tree_node_count": profile["tree_node_count"],
                "identity_signals": profile["identity_signals"],
                "source_inventory_complete": True,
                "hierarchy_nodes_intentionally_omitted": True,
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
                "Do not interpret hierarchy_nodes_intentionally_omitted as a truncated source tree; the summary "
                "states whether the source inventory was loaded completely. Do not require cross-source identity "
                "or geometry reconciliation for classification of hierarchy labels or a routine count over one "
                "clearly bounded SQL population. For a type-name question, a complete selected subtree plus "
                "snapshot-scoped wording is sufficient unless the user asks for unique physical entities. "
                "Keep review bounded to the user's requested population, metric, unit, relationship, and material "
                "exclusions. Require a follow-up only when it can materially change one of those requested outputs. "
                "Do not expand a count, type list, or grouped total into an unrelated domain-property completeness, "
                "geometry, identity, or model-quality audit. Complete grouped SQL plus any required matched "
                "population reconciliation is sufficient; do not request hierarchy rows already represented by "
                "that complete grouped SQL. Distinguish record object IDs, external IDs, property GUIDs, IFC step "
                "IDs, and authored domain fields. Never describe them collectively as missing or unique identifiers. "
                "Missing IFC mappings mean incomplete cross-source coverage, not a source-value mismatch. "
                "Compare relevant raw-record, "
                "external-ID, IFC-ID, and geometry cardinalities; repeated values are not proof of duplication. "
                "Return a compact JSON decision. Set can_finalize=false whenever a material factual, scope, "
                "identity, completeness, or reconciliation gap requires another project tool call. Put only "
                "necessary checks in required_follow_up, caveats that must survive into the final answer in "
                "required_disclosures, using only the allowed disclosure codes, and claims that must be removed "
                "or verified in unsupported_claims. "
                "When review_mode is delta, review only changed_inputs against prior_review. Carry forward "
                "unchanged supported decisions, but clear a prior follow-up when the changed evidence resolves it. "
                "Keep summary under 500 characters and each follow-up or unsupported claim under 180 characters. "
                "Do not expose private chain-of-thought."
            ),
            input=json.dumps(payload, ensure_ascii=False),
            reasoning={"effort": self.reasoning_effort},
            store=False,
            max_output_tokens=750 if incremental else 1000,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "bim_evidence_review",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "can_finalize": {"type": "boolean"},
                            "summary": {"type": "string", "maxLength": 500},
                            "required_follow_up": {
                                "type": "array",
                                "items": {"type": "string", "maxLength": 180},
                                "maxItems": 4,
                            },
                            "required_disclosures": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "loaded_snapshot", "record_count_not_physical_uniqueness",
                                        "selected_scope_only", "cross_source_mismatch",
                                        "unit_uncertainty", "ambiguous_type_semantics"
                                    ],
                                },
                                "maxItems": 6,
                            },
                            "unsupported_claims": {
                                "type": "array",
                                "items": {"type": "string", "maxLength": 180},
                                "maxItems": 4,
                            },
                        },
                        "required": [
                            "can_finalize", "summary", "required_follow_up",
                            "required_disclosures", "unsupported_claims",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            prompt_cache_key=f"{self.prompt_cache_key}-review"[:64],
        )
        self._record_usage(response, "evidence_review")
        review_text = _required_model_text(response, "evidence reviewer")
        review_parse_valid = True
        review_parse_error = ""
        try:
            if str(getattr(response, "status", "") or "").casefold() == "incomplete":
                raise ValueError("the model response status was incomplete")
            parsed = json.loads(review_text)
            if not isinstance(parsed, dict):
                raise ValueError("the review response was not a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            review_parse_valid = False
            detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
            review_parse_error = f"Invalid or truncated structured review output: {detail}."
            parsed = {
                "can_finalize": False,
                "summary": review_parse_error,
                "required_follow_up": [
                    "Repeat review_scope_and_evidence because the prior structured review was incomplete."
                ],
                "required_disclosures": [],
                "unsupported_claims": [],
            }
        required_disclosures = [str(item) for item in parsed.get("required_disclosures", [])]
        if incremental and self._last_review_result:
            required_disclosures = list(dict.fromkeys([
                *self._last_review_result.get("required_disclosures", []),
                *required_disclosures,
            ]))
        result = {
            "review": str(parsed.get("summary") or ""),
            "can_finalize": bool(parsed.get("can_finalize")),
            "required_follow_up": [str(item) for item in parsed.get("required_follow_up", [])],
            "required_disclosures": required_disclosures,
            "required_disclosure_text": [
                _review_disclosure_text(str(item), _contains_hebrew(str(arguments.get("question", ""))))
                for item in required_disclosures
            ],
            "unsupported_claims": [str(item) for item in parsed.get("unsupported_claims", [])],
            "response_id": str(getattr(response, "id", "") or ""),
            "review_mode": "delta" if incremental else "full",
            "reviewed_changes": sorted(changed_inputs),
            "review_parse_valid": review_parse_valid,
            "review_parse_error": review_parse_error or None,
        }
        if review_parse_valid:
            self._last_review_arguments = current_arguments
            self._last_review_result = result
        else:
            self._last_review_arguments = None
            self._last_review_result = None
        return result

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


def _contains_hebrew(value: str) -> bool:
    return any("\u0590" <= character <= "\u05ff" for character in value)


def _review_disclosure_text(code: str, hebrew: bool) -> str:
    english = {
        "loaded_snapshot": "The result applies to the loaded project snapshot; source freshness and cross-file version alignment are unverified.",
        "record_count_not_physical_uniqueness": "Reported record counts are not proof of unique physical entities across project sources.",
        "selected_scope_only": "The result applies only to the selected scope; categories identified as outside that scope are excluded.",
        "cross_source_mismatch": "The project sources disagree for the selected population, so the mismatch remains unresolved.",
        "unit_uncertainty": "Source units were not fully verified, so cross-source dimensional comparisons remain uncertain.",
        "ambiguous_type_semantics": "The reported types are hierarchy labels; their Revit family/type semantics were not independently verified.",
    }
    hebrew_text = {
        "loaded_snapshot": "התוצאה מתייחסת לצילום נתוני הפרויקט שנטען; עדכניות המקורות והתאמת הגרסאות בין הקבצים לא אומתו.",
        "record_count_not_physical_uniqueness": "ספירת הרשומות אינה מוכיחה שמדובר בישויות פיזיות ייחודיות בין מקורות הפרויקט.",
        "selected_scope_only": "התוצאה חלה רק על ההיקף שנבחר; קטגוריות שזוהו כמחוץ להיקף אינן נכללות.",
        "cross_source_mismatch": "קיימת אי־התאמה בין מקורות הפרויקט באוכלוסייה שנבחרה, ולכן הפער נותר בלתי פתור.",
        "unit_uncertainty": "יחידות המקור לא אומתו במלואן, ולכן השוואות ממדיות בין מקורות נותרות לא ודאיות.",
        "ambiguous_type_semantics": "הסוגים שדווחו הם תוויות היררכיה; משמעותם כמשפחה או טיפוס Revit לא אומתה בנפרד.",
    }
    selected = hebrew_text if hebrew else english
    return selected.get(code, code)
