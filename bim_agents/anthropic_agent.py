from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from .config import Settings
from .contracts import (
    AnswerReport, AuditResult, ClaimAudit, IndependentAnalysis, InvestigationPlan,
    InvestigationResult, VerificationResult,
)
from .evidence import EvidenceBudgetExhausted, EvidenceStore
from .graph import BIMGraph
from .llm import LLMProvider, ToolResult, create_provider
from .trace import AgentTraceLog

EventSink = Callable[[dict[str, Any]], Awaitable[None]]


def object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties,
            "required": required if required is not None else list(properties),
            "additionalProperties": False}


STRING_ARRAY = {"type": "array", "items": {"type": "string"}}
ENTITY_ROLE = {"type": "string", "enum": ["physical_instance", "type_definition",
    "space_record", "curated_record", "mixed", "unknown"]}


def semantic_contract_schema() -> dict[str, Any]:
    return object_schema({
        "name": {"type": "string"}, "population": {"type": "string"},
        "entity_role": ENTITY_ROLE, "spatial_scope": {"type": "string"},
        "classification_rule": {"type": "string"}, "measure": {"type": "string"},
        "measurement_basis": {"type": "string"}, "aggregation": {"type": "string"},
        "unit": {"type": "string"}, "identity_key": {"type": "string"},
        "inclusion_rules": STRING_ARRAY, "exclusion_rules": STRING_ARRAY,
        "evidence_required": STRING_ARRAY, "assumptions": STRING_ARRAY,
    })


ANALYSIS_TOOL = {"name": "submit_semantic_analysis", "strict": True,
    "description": "Submit an independent semantic interpretation of the BIM question.",
    "input_schema": object_schema({
        "agent": {"type": "string"}, "language": {"type": "string"},
        "direct_question": {"type": "string"},
        "candidates": {"type": "array", "items": semantic_contract_schema()},
        "unresolved_terms": STRING_ARRAY, "risks": STRING_ARRAY})}

PLAN_TOOL = {"name": "submit_plan", "strict": True,
    "description": "Reconcile independent analyses into a falsifiable BIM investigation plan.",
    "input_schema": object_schema({
        "interpretation": {"type": "string"}, "language": {"type": "string"},
        "answerable_from_graph": {"type": "boolean"},
        "answer_requirements": {"type": "array", "items": object_schema({
            "requirement_id": {"type": "string"}, "description": {"type": "string"},
            "mandatory": {"type": "boolean"}})},
        "selected_contract": semantic_contract_schema(),
        "alternative_contracts": {"type": "array", "items": semantic_contract_schema()},
        "unresolved_terms": STRING_ARRAY, "tasks": STRING_ARRAY,
        "success_criteria": STRING_ARRAY, "stop_conditions": STRING_ARRAY,
        "risks": STRING_ARRAY})}

PROVENANCE_PROPERTIES = {
    "population": {"type": "string"}, "entity_role": ENTITY_ROLE,
    "spatial_scope": {"type": "string"}, "measurement_basis": {"type": "string"},
    "aggregation": {"type": "string"}, "unit": {"type": "string"},
    "identity_key": {"type": "string"}, "inclusion_rules": STRING_ARRAY,
    "exclusion_rules": STRING_ARRAY,
}

EXECUTION_TOOLS = [
    {"name": "search_bim_elements", "strict": True,
     "description": "Discover scoped BIM names, classifications, labels, and available properties.",
     "input_schema": object_schema({"search_text": {"type": "string"},
         "limit": {"type": "integer"}, "purpose": {"type": "string"}})},
    {"name": "search_project_documents", "strict": True,
     "description": "Search document chunks connected to the authorized project.",
     "input_schema": object_schema({"search_text": {"type": "string"},
         "limit": {"type": "integer"}, "purpose": {"type": "string"}})},
    {"name": "inventory_project_documents", "strict": True,
     "description": "Exhaustively inventory project-connected documents; use to prove absence.",
     "input_schema": object_schema({"purpose": {"type": "string"}})},
    {"name": "profile_bim_properties", "strict": True,
     "description": "Profile populated values for exact properties over the complete scoped BIM population.",
     "input_schema": object_schema({"properties": STRING_ARRAY, "purpose": {"type": "string"}})},
    {"name": "run_readonly_cypher", "strict": True,
     "description": "Run scoped read-only Cypher and declare its complete semantic provenance.",
     "input_schema": object_schema({"query": {"type": "string"},
         "parameters_json": {"type": "string"}, "purpose": {"type": "string"},
         **PROVENANCE_PROPERTIES})},
]

CLAIM_SCHEMA = object_schema({
    "claim_id": {"type": "string"}, "statement": {"type": "string"},
    "artifact_ids": STRING_ARRAY, "population": {"type": "string"},
    "measurement_basis": {"type": "string"}, "unit": {"type": "string"},
    "claim_kind": {"type": "string", "enum": ["direct", "supporting", "inference", "unavailable"]},
    "requirement_ids": STRING_ARRAY,
})
INVESTIGATION_TOOL = {"name": "submit_investigation", "strict": True,
    "description": "Finish with atomic cited claims after testing the selected and alternative interpretations.",
    "input_schema": object_schema({"draft_answer": {"type": "string"},
        "claims": {"type": "array", "items": CLAIM_SCHEMA}, "limitations": STRING_ARRAY,
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "tested_interpretations": STRING_ARRAY, "rejected_interpretations": STRING_ARRAY})}
INVESTIGATION_TOOLS = [*EXECUTION_TOOLS, INVESTIGATION_TOOL]

CLAIM_AUDIT_SCHEMA = object_schema({
    "claim_id": {"type": "string"}, "supported": {"type": "boolean"},
    "reasons": STRING_ARRAY, "supporting_artifact_ids": STRING_ARRAY,
    "counterexample_artifact_ids": STRING_ARRAY})
AUDIT_TOOL = {"name": "submit_audit", "strict": True,
    "description": "Submit the adversarial semantic and counterexample audit.",
    "input_schema": object_schema({"semantic_contract_supported": {"type": "boolean"},
        "contract_issues": STRING_ARRAY,
        "claim_audits": {"type": "array", "items": CLAIM_AUDIT_SCHEMA},
        "missing_checks": STRING_ARRAY, "needs_more_investigation": {"type": "boolean"}})}
AUDIT_TOOLS = [*EXECUTION_TOOLS, AUDIT_TOOL]

VERIFICATION_TOOL = {"name": "submit_verification", "strict": True,
    "description": "Submit an independent claim-level evidence verification decision.",
    "input_schema": object_schema({
        "status": {"type": "string", "enum": ["verified", "partially_verified", "insufficient_evidence", "error"]},
        "issues": STRING_ARRAY, "supported_artifact_ids": STRING_ARRAY,
        "needs_more_investigation": {"type": "boolean"},
        "semantic_checks": {"type": "array", "items": object_schema({
            "name": {"type": "string"}, "passed": {"type": "boolean"},
            "explanation": {"type": "string"}})},
        "claim_audits": {"type": "array", "items": CLAIM_AUDIT_SCHEMA},
        "missing_requirement_ids": STRING_ARRAY})}

ANSWER_TOOL = {"name": "submit_answer", "strict": True,
    "description": "Submit a concise user-facing answer containing only accepted claims.",
    "input_schema": object_schema({"answer": {"type": "string"}, "limitations": STRING_ARRAY})}


class ReasoningPolicy:
    """Routes effort from semantic complexity, never from project-specific vocabulary."""

    def __init__(self, *, default: str, high: str, low: str) -> None:
        self.default = default
        self.high = high
        self.low = low

    def investigation(self, plan: InvestigationPlan, *, is_repair: bool) -> str:
        contract = plan.selected_contract
        complex_semantics = any((
            is_repair,
            len(plan.unresolved_terms) >= 2,
            len(plan.alternative_contracts) >= 2,
            contract.entity_role in {"mixed", "unknown"},
            len(contract.assumptions) >= 2,
            len(plan.risks) >= 5,
            len(contract.evidence_required) >= 7,
        ))
        return self.high if complex_semantics else self.default

    def planning(self, ontology: IndependentAnalysis, measurement: IndependentAnalysis) -> str:
        ambiguity = (
            len(set([*ontology.unresolved_terms, *measurement.unresolved_terms])) >= 3
            or len(ontology.candidates) + len(measurement.candidates) >= 6
            or len(ontology.risks) + len(measurement.risks) >= 8
        )
        return self.high if ambiguity else self.default


class BIMAgent:
    def __init__(self, settings: Settings, graph: BIMGraph | None = None,
                 provider: LLMProvider | None = None) -> None:
        self.settings = settings
        self.graph = graph or BIMGraph(settings)
        self.provider = provider or create_provider(settings)
        self.trace = AgentTraceLog(
            enabled=settings.trace_enabled, directory=Path(settings.trace_dir)
        )
        self.reasoning = ReasoningPolicy(
            default=settings.reasoning_effort,
            high=settings.high_reasoning_effort,
            low=settings.low_reasoning_effort,
        )
        if hasattr(self.provider, "set_trace_hook"):
            self.provider.set_trace_hook(
                lambda event, fields: self.trace.log(event, **fields)
            )
        self._sequence: ContextVar[int] = ContextVar("bim_event_sequence", default=0)
        self._model_calls: ContextVar[int] = ContextVar("bim_model_calls", default=0)
        self._history: dict[tuple[str, str, str], list[dict[str, str]]] = {}

    async def close(self) -> None:
        self.graph.close()
        await self.provider.close()

    async def _emit(self, sink: EventSink | None, *, stage: str, agent: str,
                    tool: str | None = None, detail: str | None = None) -> None:
        self.trace.log("pipeline_stage", stage=stage, agent=agent, tool=tool, detail=detail)
        if sink is None:
            return
        sequence = self._sequence.get() + 1
        self._sequence.set(sequence)
        await sink({"type": "pipeline_stage", "sequence": sequence,
            "event_id": f"event-{sequence}", "stage": stage, "agent": agent,
            **({"tool": tool} if tool else {}), **({"detail": detail} if detail else {})})

    def _consume_model_call(self, *, agent: str, reasoning_effort: str) -> None:
        calls = self._model_calls.get()
        if calls >= self.settings.max_model_calls:
            raise RuntimeError("The per-request model-call budget was exhausted.")
        calls += 1
        self._model_calls.set(calls)
        self.trace.log(
            "model_call", call_number=calls,
            provider=self.settings.llm_provider, agent=agent,
            reasoning_effort=reasoning_effort,
        )

    async def _structured(self, *, model: str, instructions: str,
                          payload: dict[str, Any], tool: dict[str, Any], max_tokens: int,
                          agent: str, reasoning_effort: str) -> dict[str, Any]:
        self._consume_model_call(agent=agent, reasoning_effort=reasoning_effort)
        started = asyncio.get_running_loop().time()
        result = await self.provider.structured(model=model, instructions=instructions,
            payload=payload, tool=tool, max_output_tokens=max_tokens,
            reasoning_effort=reasoning_effort)
        self.trace.log(
            "structured_result", agent=agent, tool=tool["name"],
            elapsed_ms=round((asyncio.get_running_loop().time() - started) * 1000, 1),
            result=result,
        )
        return result

    async def _independent_analysis(self, question: str, schema: dict[str, Any],
                                    history: list[dict[str, str]], sink: EventSink | None
                                    ) -> tuple[IndependentAnalysis, IndependentAnalysis]:
        ontology_prompt = """Independently analyze the BIM ontology of the question without calculating.
Distinguish physical instances, IFC type definitions, spaces, systems, and curated records. Produce competing
interpretations for ambiguous terms. Give ordinary construction-domain meaning priority over whichever graph field
is easiest to query. When scope is omitted but groupable, propose an all-groups result rather than refusing. Use the
live schema and its relationship endpoints/value examples; never assume a project-specific name mapping."""
        measurement_prompt = """Independently define the requested measure without calculating. Resolve unit,
identity key, spatial scope, aggregation and measurement basis. Distinguish stored quantity from geometry,
placement from size, gross from net, records from physical objects, and missing from zero. Treat explicit authored
quantities with a named basis as legitimate evidence; do not demand geometric derivation unless the question requires
physical geometry. Preserve and rank material alternatives instead of collapsing them prematurely."""
        payload = {"question": question, "recent_conversation": history, "live_graph_profile": schema}
        await self._emit(sink, stage="agent_start", agent="Ontology Analyst")
        await self._emit(sink, stage="agent_start", agent="Measurement Analyst")
        analysis_effort = self.reasoning.default
        self._consume_model_call(agent="Ontology Analyst", reasoning_effort=analysis_effort)
        self._consume_model_call(agent="Measurement Analyst", reasoning_effort=analysis_effort)
        raw_o, raw_m = await asyncio.gather(
            self.provider.structured(model=self.settings.agent_model, instructions=ontology_prompt,
                payload=payload, tool=ANALYSIS_TOOL, max_output_tokens=4000,
                reasoning_effort=analysis_effort),
            self.provider.structured(model=self.settings.agent_model, instructions=measurement_prompt,
                payload=payload, tool=ANALYSIS_TOOL, max_output_tokens=4000,
                reasoning_effort=analysis_effort))
        self.trace.log("structured_result", agent="Ontology Analyst",
                       tool=ANALYSIS_TOOL["name"], result=raw_o)
        self.trace.log("structured_result", agent="Measurement Analyst",
                       tool=ANALYSIS_TOOL["name"], result=raw_m)
        ontology, measurement = IndependentAnalysis.model_validate(raw_o), IndependentAnalysis.model_validate(raw_m)
        await self._emit(sink, stage="agent_end", agent="Ontology Analyst")
        await self._emit(sink, stage="agent_end", agent="Measurement Analyst")
        return ontology, measurement

    async def _plan(self, question: str, schema: dict[str, Any], ontology: IndependentAnalysis,
                    measurement: IndependentAnalysis, sink: EventSink | None) -> InvestigationPlan:
        await self._emit(sink, stage="agent_start", agent="Semantic Planner")
        planning_effort = self.reasoning.planning(ontology, measurement)
        raw = await self._structured(model=self.settings.agent_model, max_tokens=6500,
            instructions="""Reconcile the independent analyses into an explicit falsifiable semantic contract.
Keep material alternatives and unresolved terms. Tasks must test population completeness, entity role, spatial
membership, identity/deduplication, measurement basis, units, and alternative interpretations. Define one concise
mandatory answer requirement for every part of the user's question. Use at most two material alternatives, six
tasks, six success criteria, and five risks; keep every list item to one sentence. Prefer physical-instance meaning
for counts unless the user asks for types or programme records. When competing interpretations are both plausible,
plan one query that returns them side by side. Do not calculate.""",
            payload={"question": question, "schema": schema,
                "ontology_analysis": ontology.model_dump(), "measurement_analysis": measurement.model_dump()},
            tool=PLAN_TOOL, agent="Semantic Planner", reasoning_effort=planning_effort)
        plan = InvestigationPlan.model_validate(raw)
        await self._emit(sink, stage="agent_end", agent="Semantic Planner", detail=plan.interpretation)
        return plan

    @staticmethod
    def _artifact_payload(artifact: Any, max_rows: int = 80) -> dict[str, Any]:
        data = artifact.model_dump()
        data["rows"] = artifact.rows[:max_rows]
        data["truncated"] = artifact.truncated or len(artifact.rows) > max_rows
        return data

    async def _execute_tool(self, name: str, data: dict[str, Any], *, client_id: str,
                            project_id: str, store: EvidenceStore) -> dict[str, Any]:
        cache_key = store.request_key(name, data)
        cached = store.cached(cache_key)
        if cached is not None:
            payload = self._artifact_payload(cached)
            payload.update({"cached": True, "budget_remaining": store.remaining_by_phase()})
            return payload
        store.require_capacity()
        common = {"client_id": client_id, "project_id": project_id,
                  "store": store, "purpose": data["purpose"]}
        if name == "search_bim_elements":
            artifact = await asyncio.to_thread(self.graph.search_elements,
                search_text=data["search_text"], limit=data["limit"], **common)
        elif name == "search_project_documents":
            artifact = await asyncio.to_thread(self.graph.search_documents,
                search_text=data["search_text"], limit=data["limit"], **common)
        elif name == "inventory_project_documents":
            artifact = await asyncio.to_thread(self.graph.inventory_documents, **common)
        elif name == "profile_bim_properties":
            artifact = await asyncio.to_thread(self.graph.profile_properties,
                properties=data["properties"], **common)
        elif name == "run_readonly_cypher":
            parameters = json.loads(data["parameters_json"] or "{}")
            if not isinstance(parameters, dict):
                raise ValueError("parameters_json must decode to an object.")
            parameters.pop("client_id", None); parameters.pop("project_id", None)
            provenance = {key: data[key] for key in PROVENANCE_PROPERTIES}
            artifact = await asyncio.to_thread(self.graph.run_agent_cypher,
                query=data["query"], parameters=parameters, provenance=provenance, **common)
        else:
            raise ValueError(f"Unknown executable tool: {name}")
        store.remember(cache_key, artifact)
        payload = self._artifact_payload(artifact)
        payload.update({"cached": False, "budget_remaining": store.remaining_by_phase()})
        return payload

    async def _tool_agent(self, *, agent_name: str, instructions: str, payload: dict[str, Any],
                          terminal_name: str, terminal_model: Any, tools: list[dict[str, Any]],
                          client_id: str, project_id: str, store: EvidenceStore,
                          sink: EventSink | None, reasoning_effort: str, phase: str,
                          max_turns: int) -> Any:
        store.set_phase(phase)
        await self._emit(sink, stage="agent_start", agent=agent_name)
        payload = {**payload, "evidence_budget": {
            "active_phase": phase, "remaining": store.remaining_by_phase(),
            "instruction": "When the active phase reaches zero, submit the terminal tool immediately.",
        }}
        session = self.provider.tool_session(model=self.settings.worker_model,
            instructions=instructions, payload=payload, tools=tools, max_output_tokens=5200,
            reasoning_effort=reasoning_effort)
        results: list[ToolResult] | None = None
        last_text = ""
        stop_required = False
        cypher_failures = 0
        for turn_number in range(min(max_turns, self.settings.max_agent_turns)):
            self._consume_model_call(agent=agent_name, reasoning_effort=reasoning_effort)
            turn = await session.next(results)
            last_text = turn.text or last_text
            self.trace.log(
                "model_turn", agent=agent_name, text_chars=len(turn.text),
                tool_calls=[{"name": call.name, "argument_fields": sorted(call.arguments)}
                            for call in turn.tool_calls],
            )
            terminal = next((c for c in turn.tool_calls if c.name == terminal_name), None)
            if terminal:
                self.trace.log(
                    "terminal_submission", agent=agent_name, tool=terminal.name,
                    fields=sorted(terminal.arguments), phase=phase,
                    claim_count=len(terminal.arguments.get("claims", terminal.arguments.get("claim_audits", []))),
                    status=terminal.arguments.get("status"),
                    claims=[{
                        "claim_id": item.get("claim_id"),
                        "artifact_ids": item.get("artifact_ids", item.get("supporting_artifact_ids", [])),
                        "supported": item.get("supported"),
                        "requirement_ids": item.get("requirement_ids", []),
                    } for item in terminal.arguments.get(
                        "claims", terminal.arguments.get("claim_audits", [])
                    )],
                )
                value = terminal_model.model_validate(terminal.arguments)
                await self._emit(sink, stage="agent_end", agent=agent_name)
                return value
            if stop_required:
                self.trace.log(
                    "forced_agent_stop", agent=agent_name, phase=phase,
                    reason="Agent did not submit its terminal tool after a stop-required tool result.",
                    turn_number=turn_number + 1,
                )
                break
            results = []
            for call in turn.tool_calls:
                self.trace.log(
                    "tool_arguments", agent=agent_name, tool=call.name,
                    call_id=call.id, arguments=call.arguments,
                )
                await self._emit(sink, stage="tool_start", agent=agent_name, tool=call.name)
                try:
                    value = await self._execute_tool(call.name, call.arguments,
                        client_id=client_id, project_id=project_id, store=store)
                    result = ToolResult(call.id, json.dumps(value, ensure_ascii=False, default=str))
                    self.trace.log(
                        "tool_result", agent=agent_name, tool=call.name, call_id=call.id,
                        artifact_id=value.get("artifact_id"), row_count=value.get("row_count"),
                        truncated=value.get("truncated"), elapsed_ms=value.get("elapsed_ms"),
                        phase=phase, cached=value.get("cached"),
                        budget_remaining=value.get("budget_remaining"),
                        columns=value.get("columns"), sample_rows=value.get("rows", [])[:3],
                    )
                except EvidenceBudgetExhausted as exc:
                    stop_required = True
                    result = ToolResult(call.id, json.dumps({
                        "error": type(exc).__name__, "message": str(exc),
                        "stop_required": True, "budget_remaining": exc.remaining,
                        "instruction": f"Call {terminal_name} now; do not call another evidence tool.",
                    }, ensure_ascii=False), True)
                    self.trace.log(
                        "evidence_budget_stop", agent=agent_name, tool=call.name,
                        call_id=call.id, phase=phase, remaining=exc.remaining,
                    )
                except Exception as exc:
                    is_cypher_failure = call.name == "run_readonly_cypher"
                    if is_cypher_failure:
                        cypher_failures += 1
                    repairs_left = max(
                        0, self.settings.max_cypher_repairs - cypher_failures + 1
                    ) if is_cypher_failure else self.settings.max_cypher_repairs
                    if is_cypher_failure and repairs_left == 0:
                        stop_required = True
                    result = ToolResult(call.id, json.dumps({"error": type(exc).__name__,
                        "message": str(exc),
                        "cypher_repair_required": is_cypher_failure and repairs_left > 0,
                        "cypher_repairs_remaining": repairs_left,
                        "stop_required": stop_required,
                        "instruction": (
                            "Correct the Cypher using the database error and retry once with the same purpose."
                            if is_cypher_failure and repairs_left > 0
                            else f"Call {terminal_name} now with the evidence already collected."
                            if stop_required else "Correct the tool input before retrying."
                        ),
                    }, ensure_ascii=False), True)
                    self.trace.log(
                        "tool_error", agent=agent_name, tool=call.name, call_id=call.id,
                        error_type=type(exc).__name__, message=str(exc),
                        phase=phase, cypher_repairs_remaining=repairs_left,
                    )
                results.append(result)
                await self._emit(sink, stage="tool_end", agent=agent_name, tool=call.name)
            if not results:
                break
        await self._emit(sink, stage="agent_end", agent=agent_name)
        if terminal_name == "submit_investigation":
            return InvestigationResult(draft_answer=last_text or "Insufficient evidence.",
                limitations=["Investigation budget ended before a supported conclusion."], confidence="low")
        return AuditResult(semantic_contract_supported=False,
            contract_issues=["Audit budget ended before a decision."],
            missing_checks=["Incomplete adversarial audit."], needs_more_investigation=False)

    def _evidence_context(self, store: EvidenceStore) -> list[dict[str, Any]]:
        return [self._artifact_payload(x) for x in store.artifacts]

    async def _investigate(self, *, question: str, client_id: str, project_id: str,
                           schema: dict[str, Any], plan: InvestigationPlan, store: EvidenceStore,
                           history: list[dict[str, str]], repair_issues: list[str],
                           previous_investigation: InvestigationResult | None,
                           sink: EventSink | None) -> InvestigationResult:
        phase = "repair" if repair_issues else "investigation"
        return await self._tool_agent(agent_name="BIM Investigator",
            terminal_name="submit_investigation", terminal_model=InvestigationResult,
            instructions="""Investigate only with authorized graph tools. Follow the semantic contract and test
material alternatives. Scope every BIMElement. Separate instances, types, spaces, systems and curated records.
Establish identity and source overlap before counting. Preserve basis and unit; never substitute placement for
height, name hints for properties, missing for zero, or records for physical objects. Compliance requires both a
scoped requirement and measured fact. Satisfy every mandatory answer requirement, starting with the smallest decisive
query before exploratory profiles. For counts, inventory all scoped instance labels and sources before narrowing;
reconcile candidate identities to physical instances. For authored quantities, report the exact stored basis and also
test competing curated/physical populations. If a requested grouping scope is omitted, return all groups. Reconcile
totals. Stop querying when the active phase budget is exhausted. During
repair, reuse exact existing artifact IDs when they remain valid. Return atomic claims with requirement IDs and exact
artifact IDs.""",
            payload={"question": question, "recent_conversation": history, "plan": plan.model_dump(),
                "schema": schema, "repair_issues": repair_issues,
                "previous_investigation": previous_investigation.model_dump()
                    if previous_investigation else None,
                "existing_evidence": self._evidence_context(store) if repair_issues else []},
            tools=INVESTIGATION_TOOLS,
            client_id=client_id, project_id=project_id, store=store, sink=sink,
            reasoning_effort=self.reasoning.investigation(plan, is_repair=bool(repair_issues)),
            phase=phase, max_turns=(
                self.settings.repair_max_turns if repair_issues
                else self.settings.investigator_max_turns
            ))

    async def _audit(self, *, question: str, client_id: str, project_id: str,
                     schema: dict[str, Any], investigation: InvestigationResult,
                     store: EvidenceStore, sink: EventSink | None,
                     repair_round: bool = False) -> AuditResult:
        phase = "repair_audit" if repair_round else "audit"
        return await self._tool_agent(agent_name="Counterexample Auditor",
            terminal_name="submit_audit", terminal_model=AuditResult,
            instructions="""Independently infer question semantics without the planner. Try to disprove every
direct claim with counterqueries: test entity roles, category boundaries, source overlap, identity, spatial
membership, measurement bases, units, missing values and reconciliation. Search is not proof of absence; query
the complete scoped population. Execute at least one independent counterquery in your reserved audit phase before
supporting any direct claim. Explicitly compare physical-instance, type-definition, curated-record, source-union and
identifier-grouping alternatives when present. Support only claims whose exact population, basis, unit and value
survive.""",
            payload={"question": question, "schema": schema,
                "claims_under_audit": investigation.model_dump(),
                "existing_evidence": self._evidence_context(store)}, tools=AUDIT_TOOLS,
            client_id=client_id, project_id=project_id, store=store, sink=sink,
            reasoning_effort=self.reasoning.high, phase=phase,
            max_turns=(
                self.settings.repair_auditor_max_turns if repair_round
                else self.settings.auditor_max_turns
            ))

    async def _verify(self, *, question: str, plan: InvestigationPlan,
                      investigation: InvestigationResult, audit: AuditResult,
                      store: EvidenceStore, sink: EventSink | None) -> VerificationResult:
        await self._emit(sink, stage="agent_start", agent="Claim Verifier")
        raw = await self._structured(model=self.settings.agent_model, max_tokens=3400,
            instructions="""Verify every atomic claim against evidence and the adversarial audit. A direct claim
needs established population, role, spatial scope, identity/deduplication, basis, unit and value. Unavailability
needs exhaustive absence evidence. An explicit normalized authored quantity may be accepted with its qualified basis
without a geometry reconstruction; it must not be relabeled as a geometric union or physical dimension. Verify the
answer only if all direct claims, every mandatory answer requirement, and the semantic contract survive.""",
            payload={"question": question, "plan": plan.model_dump(),
                "investigation": investigation.model_dump(), "adversarial_audit": audit.model_dump(),
                "evidence": self._evidence_context(store)}, tool=VERIFICATION_TOOL,
            agent="Claim Verifier", reasoning_effort=self.reasoning.high)
        result = self._enforce_gate(
            VerificationResult.model_validate(raw), investigation, audit, store, plan
        )
        await self._emit(sink, stage="agent_end", agent="Claim Verifier", detail=result.status)
        return result

    @staticmethod
    def _enforce_gate(result: VerificationResult, investigation: InvestigationResult,
                      audit: AuditResult, store: EvidenceStore,
                      plan: InvestigationPlan | None = None) -> VerificationResult:
        artifact_ids = {x.artifact_id for x in store.artifacts}
        independent_audit_ids = {
            x.artifact_id for x in store.artifacts if x.phase in {"audit", "repair_audit"}
        }
        verifier = {x.claim_id: x for x in result.claim_audits}
        adversary = {x.claim_id: x for x in audit.claim_audits}
        accepted: list[ClaimAudit] = []
        issues = [*result.issues, *audit.contract_issues, *audit.missing_checks]
        for claim in investigation.claims:
            va, aa = verifier.get(claim.claim_id), adversary.get(claim.claim_id)
            cited = set(claim.artifact_ids)
            citations_valid = bool(cited) and cited <= artifact_ids
            audit_citations = set(
                (aa.supporting_artifact_ids if aa else [])
                + (aa.counterexample_artifact_ids if aa else [])
            )
            independent_audit = bool(audit_citations & independent_audit_ids)
            supported = bool(
                va and aa and va.supported and aa.supported
                and citations_valid and independent_audit
            )
            reasons = list(dict.fromkeys([*(va.reasons if va else ["Verifier omitted claim."]),
                *(aa.reasons if aa else ["Auditor omitted claim."]),
                *([] if citations_valid else ["Claim lacks valid evidence citations."]),
                *([] if independent_audit else [
                    "Claim lacks an independently executed auditor artifact."
                ])]))
            accepted.append(ClaimAudit(claim_id=claim.claim_id, supported=supported,
                reasons=reasons, supporting_artifact_ids=sorted(cited) if supported else [],
                counterexample_artifact_ids=list(dict.fromkeys(
                    (va.counterexample_artifact_ids if va else []) +
                    (aa.counterexample_artifact_ids if aa else [])))))
            if claim.claim_kind == "direct" and not supported:
                issues.append(f"Direct claim {claim.claim_id} failed: {'; '.join(reasons)}")
        direct_ids = {x.claim_id for x in investigation.claims if x.claim_kind == "direct"}
        accepted_direct = {x.claim_id for x in accepted if x.supported} & direct_ids
        mandatory_requirements = {
            item.requirement_id for item in (plan.answer_requirements if plan else [])
            if item.mandatory
        }
        covered_requirements = {
            requirement_id
            for claim in investigation.claims
            if any(item.claim_id == claim.claim_id and item.supported for item in accepted)
            for requirement_id in claim.requirement_ids
        }
        missing_requirements = sorted(mandatory_requirements - covered_requirements)
        if missing_requirements:
            issues.append(
                "Mandatory answer requirements lack accepted claims: "
                + ", ".join(missing_requirements)
            )
        if not direct_ids or not accepted_direct:
            status = "insufficient_evidence"
        elif (
            accepted_direct != direct_ids
            or not audit.semantic_contract_supported
            or bool(missing_requirements)
        ):
            status = "partially_verified"
        else:
            status = "verified"
        supported_ids = sorted({aid for x in accepted if x.supported for aid in x.supporting_artifact_ids})
        return result.model_copy(update={"status": status, "issues": list(dict.fromkeys(issues)),
            "supported_artifact_ids": supported_ids, "claim_audits": accepted,
            "missing_requirement_ids": missing_requirements})

    async def _compose(self, *, question: str, plan: InvestigationPlan,
                       investigation: InvestigationResult, verification: VerificationResult,
                       store: EvidenceStore, sink: EventSink | None) -> tuple[str, list[str]]:
        await self._emit(sink, stage="agent_start", agent="Answer Composer")
        accepted_ids = {x.claim_id for x in verification.claim_audits if x.supported}
        supported_artifacts = set(verification.supported_artifact_ids)
        raw = await self._structured(model=self.settings.agent_model, max_tokens=1800,
            instructions="""Answer in the user's language and lead with the shortest direct result. State only
accepted claims and evidence values; preserve scope, basis and unit. With no accepted direct claim, plainly say
evidence is insufficient for the exact interpretation, but still report any accepted supporting numeric or
categorical result as an explicitly qualified model result. Address every mandatory answer requirement: answer it,
or state in one sentence what evidence is missing. Do not mention internal agents, prompts, Cypher or IDs.""",
            payload={"question": question, "language": plan.language,
                "verification_status": verification.status,
                "answer_requirements": [item.model_dump() for item in plan.answer_requirements],
                "missing_requirement_ids": verification.missing_requirement_ids,
                "accepted_claims": [x.model_dump() for x in investigation.claims if x.claim_id in accepted_ids],
                "issues": verification.issues,
                "accepted_evidence": [x for x in self._evidence_context(store)
                    if x["artifact_id"] in supported_artifacts]}, tool=ANSWER_TOOL,
            agent="Answer Composer", reasoning_effort=self.reasoning.low)
        await self._emit(sink, stage="agent_end", agent="Answer Composer")
        return str(raw["answer"]), list(raw["limitations"])

    async def answer(self, *, question: str, client_id: str, project_id: str,
                     sink: EventSink | None = None, session_id: str | None = None,
                     request_id: str | None = None, evaluation_run_id: str | None = None,
                     evaluation_case_index: int | None = None) -> AnswerReport:
        self._model_calls.set(0)
        self._sequence.set(0)
        self.trace.start(
            question=question, client_id=client_id, project_id=project_id,
            session_id=session_id, provider=self.settings.llm_provider,
            external_request_id=request_id, evaluation_run_id=evaluation_run_id,
            evaluation_case_index=evaluation_case_index,
            agent_model=self.settings.agent_model, worker_model=self.settings.worker_model,
            reasoning_policy={"default": self.reasoning.default, "high": self.reasoning.high,
                              "low": self.reasoning.low},
        )
        trace: list[str] = []
        stages = ["Graph Inspector"]
        await self._emit(sink, stage="agent_start", agent="Graph Inspector")
        schema = await asyncio.to_thread(self.graph.schema_snapshot, client_id, project_id)
        self.trace.log(
            "graph_profile", element_count=schema.get("element_count"),
            source_count=schema.get("source_count"), labels=schema.get("labels"),
            directed_relationship_endpoints=schema.get("directed_relationship_endpoints"),
            identity_profile=schema.get("identity_profile"),
            source_overlap=schema.get("source_overlap"),
            semantic_value_profile=schema.get("semantic_value_profile"),
        )
        await self._emit(sink, stage="agent_end", agent="Graph Inspector",
            detail=f"Profiled {schema['element_count']} scoped BIM elements.")
        trace.append(f"Graph Inspector: profiled {schema['element_count']} elements across {schema['source_count']} sources.")
        history_key = (client_id, project_id, session_id) if session_id else None
        history = list(self._history.get(history_key, [])) if history_key else []
        store = EvidenceStore(
            total_limit=self.settings.max_artifacts,
            phase_limits={
                "investigation": self.settings.investigator_artifact_budget,
                "audit": self.settings.auditor_artifact_budget,
                "repair": self.settings.repair_artifact_budget,
                "repair_audit": self.settings.repair_auditor_artifact_budget,
            },
        )
        ontology, measurement = await self._independent_analysis(question, schema, history, sink)
        stages.extend(["Ontology Analyst", "Measurement Analyst"])
        plan = await self._plan(question, schema, ontology, measurement, sink)
        stages.append("Semantic Planner"); trace.append(f"Semantic contract: {plan.interpretation}")
        investigation = await self._investigate(question=question, client_id=client_id,
            project_id=project_id, schema=schema, plan=plan, store=store, history=history,
            repair_issues=[], previous_investigation=None, sink=sink)
        stages.append("BIM Investigator")
        trace.append(
            f"BIM Investigator: {store.used('investigation')} artifacts, "
            f"{len(investigation.claims)} claims; reserved audit capacity remained."
        )
        audit = await self._audit(question=question, client_id=client_id, project_id=project_id,
            schema=schema, investigation=investigation, store=store, sink=sink)
        stages.append("Counterexample Auditor")
        trace.append(
            f"Counterexample Auditor: created {store.used('audit')} independent artifact(s)."
        )
        verification = await self._verify(question=question, plan=plan,
            investigation=investigation, audit=audit, store=store, sink=sink)
        stages.append("Claim Verifier")
        repairs = 0
        while (
            verification.needs_more_investigation
            and repairs < self.settings.max_repairs
            and store.remaining("repair") > 0
            and store.remaining("repair_audit") > 0
            and self.settings.max_model_calls - self._model_calls.get() >= 7
        ):
            repairs += 1
            self.trace.log(
                "repair_start", repair_number=repairs,
                evidence_budget=store.remaining_by_phase(),
                model_calls_remaining=self.settings.max_model_calls - self._model_calls.get(),
                issues=verification.issues,
            )
            previous_investigation = investigation
            investigation = await self._investigate(question=question, client_id=client_id,
                project_id=project_id, schema=schema, plan=plan, store=store, history=history,
                repair_issues=verification.issues,
                previous_investigation=previous_investigation, sink=sink)
            audit = await self._audit(question=question, client_id=client_id,
                project_id=project_id, schema=schema, investigation=investigation,
                store=store, sink=sink, repair_round=True)
            verification = await self._verify(question=question, plan=plan,
                investigation=investigation, audit=audit, store=store, sink=sink)
            self.trace.log(
                "repair_end", repair_number=repairs, status=verification.status,
                evidence_budget=store.remaining_by_phase(),
            )
        if verification.needs_more_investigation and repairs == 0:
            self.trace.log(
                "repair_skipped", evidence_budget=store.remaining_by_phase(),
                model_calls_remaining=self.settings.max_model_calls - self._model_calls.get(),
            )
        trace.append(f"Claim Verifier: {verification.status}; {sum(x.supported for x in verification.claim_audits)}/{len(verification.claim_audits)} accepted.")
        answer, composer_limits = await self._compose(question=question, plan=plan,
            investigation=investigation, verification=verification, store=store, sink=sink)
        stages.append("Answer Composer")
        limitations = list(dict.fromkeys([*investigation.limitations,
            *verification.issues, *composer_limits]))[:12]
        failures = (["insufficient_graph_evidence"] if verification.status == "insufficient_evidence"
            else ["partial_graph_evidence"] if verification.status == "partially_verified" else [])
        report = AnswerReport(answer=answer, verification_status=verification.status,
            limitations=limitations, stages_used=stages,
            artifact_ids=[x.artifact_id for x in store.artifacts], investigation_trace=trace,
            failure_categories=failures, semantic_checks=verification.semantic_checks)
        if history_key:
            self._history[history_key] = [*history, {"role": "user", "content": question},
                {"role": "assistant", "content": answer}][-12:]
        self.trace.finish(
            status=report.verification_status, artifact_count=len(report.artifact_ids),
            artifact_usage={phase: store.used(phase) for phase in (
                "investigation", "audit", "repair", "repair_audit"
            )}, evidence_budget_remaining=store.remaining_by_phase(),
            model_calls=self._model_calls.get(), stages=report.stages_used,
        )
        return report

    def record_pipeline_error(self, exc: Exception) -> None:
        self.trace.finish(
            status="error", error_type=type(exc).__name__, message=str(exc),
            model_calls=self._model_calls.get(),
        )
