from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from .graph_contract import QueryEntity, QueryField
from .geometry import calculate_project_geometry
from .cypher_handler import CypherQueryHandler
from .models import (
    BimRunContext, BimTaskContract, CompletionGateReport, ConstraintBinding, Evidence,
    InvestigationAction, InvestigationHypothesis, InvestigationObservation, RunArtifact,
    ProjectModelProfile, SemanticIntent,
)
from .schema_mapping import (
    RegisteredSchemaMapping, SchemaFieldMapping, SchemaMappingProposal,
    SchemaRelationshipBinding, SchemaRelationshipStep, SchemaValueBinding,
    SchemaValueMatch,
)


Operation = Literal[
    "count", "list", "group_count", "group_summary", "distinct",
    "count_distinct", "sum", "average", "minimum", "maximum", "maximum_group_sum",
    "multi_group_count", "multi_group_summary", "coverage", "relationship_coverage",
]
QueryRole = Literal["exploratory", "supporting", "answer_producing", "rejected"]
_MODEL_PROFILE_CACHE: dict[str, ProjectModelProfile] = {}
_MODEL_PROFILE_LOCK = Lock()

# Investigation phases are deliberately a small state machine.  A phase may
# be revisited only through an explicitly listed recovery path; arbitrary
# model-authored jumps (for example understand -> respond) are rejected.
INVESTIGATION_PHASE_TRANSITIONS: dict[str, frozenset[str]] = {
    "understand": frozenset({"explore", "analyze"}),
    "explore": frozenset({"analyze"}),
    "analyze": frozenset({"act", "verify", "blocked"}),
    "act": frozenset({"observe"}),
    "observe": frozenset({"analyze", "verify"}),
    "verify": frozenset({"respond", "analyze", "blocked"}),
    "respond": frozenset(),
    "blocked": frozenset(),
}


@dataclass(frozen=True)
class PipelineContext:
    """Small compatibility wrapper used by deterministic pipeline operations."""

    context: BimRunContext


def _transition_investigation_phase(notebook, target: str) -> None:
    """Apply one governed phase transition and append it to the notebook trace."""
    current = notebook.phase
    if target == current:
        return
    allowed = INVESTIGATION_PHASE_TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        raise ValueError(
            f"Invalid investigation phase transition: {current} -> {target}."
        )
    notebook.phase = target
    # A notebook loaded from an older serialized contract may not have a
    # history entry; repair that harmlessly before appending the transition.
    if not notebook.phase_history:
        notebook.phase_history.append(current)
    notebook.phase_history.append(target)


def _record_observation_phase(notebook) -> None:
    """Record the observe event, then return to analyze for the next decision.

    The normal route is act -> observe -> analyze.  Older callers recorded an
    observation directly from explore/analyze, so that event is retained as a
    compatibility checkpoint while the notebook still returns through the
    governed analyze state.
    """
    if notebook.phase == "act":
        _transition_investigation_phase(notebook, "observe")
    elif notebook.phase == "observe":
        pass
    elif notebook.phase in {"explore", "analyze"}:
        # Observation is an event in these legacy flows rather than an
        # actionable state transition.  Keep it visible in the audit trace.
        if not notebook.phase_history:
            notebook.phase_history.append(notebook.phase)
        notebook.phase_history.append("observe")
        notebook.phase_history.append("analyze")
    else:
        raise ValueError(
            f"Invalid investigation phase transition: {notebook.phase} -> observe."
        )
    if notebook.phase != "analyze":
        _transition_investigation_phase(notebook, "analyze")
FilterOperator = Literal[
    "equals", "not_equals", "contains", "starts_with", "in",
    "greater_than", "greater_or_equal", "less_than", "less_or_equal",
    "exists", "is_missing",
]


class BimFilter(BaseModel):
    field: str
    operator: FilterOperator
    value: str = ""
    values: list[str] = Field(default_factory=list)


class BimRelationshipFilter(BaseModel):
    relationship: str = Field(
        min_length=1,
        description="Name of a fixed live-validated relationship binding on the active mapping.",
    )
    operator: Literal["exists", "is_missing"]


class BimQueryPlan(BaseModel):
    entity: str
    mapping_id: str = ""
    operation: Operation
    distinct_by: Literal["identity"] = Field(
        default="identity",
        description="Deduplicate counts by the registered stable entity identity.",
    )
    filters: list[BimFilter] = Field(default_factory=list)
    relationship_filters: list[BimRelationshipFilter] = Field(default_factory=list)
    select: list[str] = Field(default_factory=list)
    metric: str = ""
    coverage_field: str = Field(
        default="",
        description="Semantic property whose populated/missing coverage must be exhaustively measured.",
    )
    relationship: str = Field(
        default="",
        description="Registered relationship binding whose complete base-population coverage is measured.",
    )
    group_by: str = ""
    group_by_fields: list[str] = Field(
        default_factory=list, max_length=3,
        description="One to three semantic fields for multi-dimensional grouping.",
    )
    sort_by: str = ""
    sort_order: Literal["ascending", "descending"] = "ascending"
    limit: int = Field(default=10, ge=1, le=20)
    include_details: bool = Field(
        default=False,
        description="Include bounded technical record details when supported.",
    )
    include_in_answer: bool = Field(
        default=True,
        description="False only for a supporting exploration query whose claim should not be rendered.",
    )
    role: QueryRole = Field(
        default="answer_producing",
        description="Evidence role. Only answer_producing claims may be rendered.",
    )
    answer_key: str = Field(
        default="",
        description="Stable semantic key shared by alternative plans answering the same atomic fact.",
    )
    satisfies: list[str] = Field(
        default_factory=list,
        description="Exact required_outputs from the task contract satisfied by this plan.",
    )
    constraint_bindings: list[ConstraintBinding] = Field(
        default_factory=list,
        description="Explicit package-owned provenance for every semantic constraint.",
    )
    work_package_id: str = ""
    semantic_intent: SemanticIntent = Field(
        default_factory=SemanticIntent,
        description="The query worker's explicit interpretation; replay verifies it against observed semantics.",
    )


class GeometryQueryPlan(BaseModel):
    calculation: str = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
        description=(
            "Server-advertised governed calculation key. The runtime registry and active "
            "project knowledge, never model-authored code, determine its implementation."
        ),
    )
    include_in_answer: bool = True
    role: QueryRole = "answer_producing"
    answer_key: str = ""
    satisfies: list[str] = Field(default_factory=list)
    semantic_intent: SemanticIntent = Field(default_factory=SemanticIntent)


# Compatibility name while external callers migrate from the geometry-only API.
CalculationPlan = GeometryQueryPlan


def _normalize_contract_semantics(contract: BimTaskContract) -> BimTaskContract:
    """Correct mechanically impossible semantic combinations at the trust boundary.

    The architect may describe an ordinary coverage output as if missing data were
    the expected result. Coverage reports a denominator and missing count; it does
    not imply that data must be absent. Requirements availability is different:
    zero scoped requirement records establishes a missing requirement population.
    """
    normalized = contract.model_copy(deep=True)
    requirements_outputs = {
        output
        for package in normalized.work_packages
        if package.route_hint == "requirements" and not any(
            spec.kind == "compliance" for spec in package.output_specs
        )
        for output in package.required_outputs
    }
    explicit_absence = re.compile(
        r"(?<!\w)(?:absent|absence|missing|unpopulated|without|lack|lacks|zero)(?!\w)",
        re.I,
    )
    output_specs = []
    for output, spec in zip(
        normalized.required_outputs, normalized.output_specs, strict=True,
    ):
        intent = spec.semantic_intent.model_copy(deep=True)
        if output in requirements_outputs:
            intent.absence_semantics = "population_missing"
        elif (
            spec.kind == "relationship_coverage"
            and explicit_absence.search(f"{normalized.goal} {output}")
        ):
            # The absent thing is a relationship/property on an existing governed
            # component population, not an absent component population.
            intent.absence_semantics = "property_missing"
        elif (
            spec.kind == "coverage"
            and intent.absence_semantics != "unspecified"
            and not explicit_absence.search(f"{normalized.goal} {output}")
        ):
            intent.absence_semantics = "unspecified"
        output_specs.append(spec.model_copy(update={"semantic_intent": intent}))
    normalized.output_specs = output_specs
    by_output = dict(zip(normalized.required_outputs, output_specs, strict=True))
    for package in normalized.work_packages:
        package.output_specs = [
            by_output[output].model_copy(deep=True)
            for output in package.required_outputs
        ]
    return normalized


def define_bim_task(
    ctx: PipelineContext, contract: BimTaskContract
) -> str:
    """Register the supervisor's goal, constraints, open questions, and definition of done."""
    if ctx.context.task_contract is not None:
        raise ValueError("The BIM task contract has already been defined for this run.")
    if not contract.goal.strip() or not contract.entity_concept.strip():
        raise ValueError("The task goal and entity concept cannot be blank.")
    contract = _normalize_contract_semantics(contract)
    ctx.context.task_contract = contract
    ctx.context.notebook.goal = contract.goal
    ctx.context.notebook.required_outputs = list(contract.required_outputs)
    ctx.context.notebook.unresolved_questions = list(contract.questions_to_resolve)
    _transition_investigation_phase(ctx.context.notebook, "explore")
    # Model-assigned complexity may increase capacity, but must never lower the
    # deployment-configured budget. A linguistically simple count can still need
    # nested schema discovery, mapping, query planning, and deterministic replay.
    if contract.complexity == "complex":
        ctx.context.max_llm_calls = max(ctx.context.max_llm_calls, 35)
        ctx.context.max_tool_calls = max(ctx.context.max_tool_calls, 55)
        ctx.context.max_agent_starts = max(ctx.context.max_agent_starts, 45)
    artifact = RunArtifact(
        artifact_id="task-contract",
        kind="task_contract",
        producer="Pipeline",
        summary=contract.goal,
        payload=contract.model_dump(mode="json"),
    )
    ctx.context.add_artifact(artifact)
    return artifact.model_dump_json()


def register_investigation_hypothesis(
    ctx: PipelineContext, hypothesis: InvestigationHypothesis
) -> str:
    """Register one falsifiable interpretation before testing it against the graph."""
    if any(item.hypothesis_id == hypothesis.hypothesis_id for item in ctx.context.notebook.hypotheses):
        raise ValueError(f"Hypothesis {hypothesis.hypothesis_id!r} already exists.")
    ctx.context.notebook.hypotheses.append(hypothesis)
    _transition_investigation_phase(ctx.context.notebook, "analyze")
    return hypothesis.model_dump_json()


def record_investigation_observation(
    ctx: PipelineContext, observation: InvestigationObservation
) -> str:
    """Record what a completed action established before another action is selected."""
    notebook = ctx.context.notebook
    if notebook.next_action and notebook.next_action.action_id != observation.action_id:
        raise ValueError("The observation must correspond to the currently selected action.")
    if observation.phase != "observe":
        raise ValueError("Investigation observations must use the observe phase.")
    known_evidence = set(ctx.context.evidence) | set(ctx.context.artifacts)
    unknown = sorted(set(observation.evidence_ids) - known_evidence)
    if unknown:
        raise ValueError("Observation references unknown evidence: " + ", ".join(unknown))
    # Validate and apply the phase route before mutating observation or
    # hypothesis state, so a rejected transition leaves the notebook intact.
    _record_observation_phase(notebook)
    notebook.observations.append(observation)
    for hypothesis in notebook.hypotheses:
        if hypothesis.hypothesis_id in observation.supports_hypotheses:
            hypothesis.status = "supported"
            hypothesis.evidence_ids = list(dict.fromkeys(hypothesis.evidence_ids + observation.evidence_ids))
        if hypothesis.hypothesis_id in observation.contradicts_hypotheses:
            hypothesis.status = "rejected"
            hypothesis.evidence_ids = list(dict.fromkeys(hypothesis.evidence_ids + observation.evidence_ids))
            notebook.rejected_interpretations.append(hypothesis.statement)
    notebook.unresolved_questions = list(dict.fromkeys(
        notebook.unresolved_questions + observation.new_questions
    ))
    notebook.next_action = None
    artifact = RunArtifact(
        artifact_id=f"observation-{len(notebook.observations)}",
        kind="investigation_observation", producer="Pipeline",
        summary=observation.result_summary, payload=observation.model_dump(mode="json"),
    )
    ctx.context.add_artifact(artifact)
    return artifact.model_dump_json()


def select_investigation_action(ctx: PipelineContext, action: InvestigationAction) -> str:
    """Select one bounded next action after observing the previous action's result."""
    notebook = ctx.context.notebook
    if notebook.next_action is not None:
        raise ValueError(
            "Record an observation for the current action before selecting another action."
        )
    _transition_investigation_phase(notebook, action.phase)
    notebook.next_action = action
    if action.phase in {"respond", "blocked"}:
        # Terminal decisions do not have a follow-up observation to record.
        notebook.next_action = None
    artifact = RunArtifact(
        artifact_id=f"action-{action.action_id}", kind="investigation_action",
        producer="Pipeline", summary=action.objective, payload=action.model_dump(mode="json"),
    )
    ctx.context.add_artifact(artifact)
    return artifact.model_dump_json()


def inspect_completion_gates(ctx: PipelineContext) -> str:
    """Evaluate whether the investigation may respond or must continue exploring."""
    notebook = ctx.context.notebook
    # Completion inspection is the verification event.  Route through the
    # governed states so even a direct runtime call cannot skip to respond.
    if notebook.phase == "understand":
        _transition_investigation_phase(notebook, "explore")
    if notebook.phase == "explore":
        _transition_investigation_phase(notebook, "analyze")
    if notebook.phase == "observe":
        _transition_investigation_phase(notebook, "verify")
    elif notebook.phase == "analyze":
        _transition_investigation_phase(notebook, "verify")
    # A verification action is completed by this gate inspection.  Clear its
    # pending slot so a failed verification can select a correction action.
    if notebook.phase == "verify":
        notebook.next_action = None
    query_payloads = [
        json.loads(item.payload) for item in ctx.context.evidence.values() if item.kind == "query"
    ]
    verification_payloads = [
        json.loads(item.payload) for item in ctx.context.evidence.values() if item.kind == "verification"
    ]
    checks = [
        check for payload in verification_payloads for check in payload.get("checks") or []
    ]
    verified_checks = [check for check in checks if check.get("verified") is True]
    semantic = [item for check in verified_checks for item in check.get("semantic_checks") or []]
    passed_names = {item.get("name") for item in semantic if item.get("passed") is True}
    adequacy_names = {
        "entity_grain_matches", "measurement_basis_matches", "population_complete",
        "planned_actual_distinguished", "absence_semantics_correct",
        "projection_answers_question", "requested_outputs_present",
    }
    contract_has_semantic_intent = bool(
        ctx.context.task_contract and (
            ctx.context.task_contract.semantic_intent.is_specified
            or any(spec.semantic_intent.is_specified for spec in ctx.context.task_contract.output_specs)
        )
    )
    semantic_adequacy_verified = (
        not contract_has_semantic_intent or adequacy_names.issubset(passed_names)
    )
    verified_outputs = {
        str(output)
        for check in verified_checks
        if check.get("plan", {}).get("role", "answer_producing")
        in {"answer_producing", "supporting"}
        for output in check.get("plan", {}).get("satisfies") or []
    }
    answer_values: dict[str, set[tuple[str, str]]] = {}
    # Only independently verified claims can create a terminal contradiction.
    # Rejected exploratory/correction attempts remain in the audit trail but must
    # not poison a later verified repair that uses the same stable answer key.
    for check in verified_checks:
        plan = check.get("plan") or {}
        key = str(plan.get("answer_key") or "").strip()
        if key and plan.get("role", "answer_producing") == "answer_producing":
            claim = check.get("claim") or {}
            answer_values.setdefault(key, set()).add((str(claim.get("value")), str(claim.get("unit"))))
    contradictions = [key for key, values in answer_values.items() if len(values) > 1]
    required = set(notebook.required_outputs)
    missing_outputs = sorted(required - verified_outputs)
    report = CompletionGateReport(
        scope_verified=bool(ctx.context.scope.allowed_sources),
        entity_verified=bool(query_payloads),
        identity_verified="identity_integrity" in passed_names,
        classification_verified="classification_purity" in passed_names,
        units_verified=all(
            (payload.get("claim") or {}).get("unit") is not None
            for payload in query_payloads
            if (payload.get("plan") or {}).get("role", "answer_producing") == "answer_producing"
        ),
        required_outputs_verified=not missing_outputs,
        contradictions_resolved=not contradictions,
        replay_verified=bool(verified_checks) and "replay_stability" in passed_names,
        semantic_adequacy_verified=semantic_adequacy_verified,
        ready_to_respond=False,
        missing=(
            [f"required output: {item}" for item in missing_outputs]
            + [f"contradiction: {item}" for item in contradictions]
        ),
    )
    report.ready_to_respond = all([
        report.scope_verified, report.entity_verified, report.identity_verified,
        report.classification_verified, report.units_verified,
        report.required_outputs_verified, report.contradictions_resolved, report.replay_verified,
        report.semantic_adequacy_verified,
    ])
    for name in (
        "scope_verified", "entity_verified", "identity_verified", "classification_verified",
        "units_verified", "required_outputs_verified", "contradictions_resolved", "replay_verified",
        "semantic_adequacy_verified",
    ):
        if not getattr(report, name) and name not in report.missing:
            report.missing.append(name)
    notebook.verified_outputs = sorted(verified_outputs)
    if report.ready_to_respond:
        if notebook.phase != "respond":
            _transition_investigation_phase(notebook, "respond")
    elif notebook.phase == "verify":
        _transition_investigation_phase(notebook, "analyze")
    return report.model_dump_json()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _normalize(value: str | None) -> str:
    text = "".join(ch for ch in (value or "") if unicodedata.category(ch) != "Cf")
    return " ".join(text.casefold().strip().split())


def _expanded_semantic_text(context: BimRunContext, concept: str) -> str:
    """Expand multilingual user vocabulary before embedding it against model-language text."""
    terms = [concept]
    ontology = context.bim.ontology
    for method_name in ("keyword_synonyms_for", "retrieval_terms_for"):
        method = getattr(ontology, method_name, None)
        if not callable(method):
            continue
        try:
            terms.extend(str(item) for item in (method(concept) or []) if str(item).strip())
        except (TypeError, ValueError):
            continue
    keyword_method = getattr(ontology, "keyword_synonyms_for", None)
    if callable(keyword_method):
        for token in re.findall(r"[^\W\d_]+", concept, flags=re.UNICODE):
            try:
                terms.extend(
                    str(item) for item in (keyword_method(token) or []) if str(item).strip()
                )
            except (TypeError, ValueError):
                continue
    unique = list(dict.fromkeys(term.strip() for term in terms if term.strip()))
    return "user concept: " + concept + "; BIM vocabulary: " + " | ".join(unique[:40])


def _level_number(value: str) -> int | None:
    normalized = _normalize(value)
    match = re.search(r"\b(\d{1,3})(?:st|nd|rd|th)?\b", normalized)
    return int(match.group(1)) if match else None


def _level_matches(
    requested: str,
    stored: str | None,
    level_aliases: dict[str, list[str]] | None = None,
) -> bool:
    requested_norm = _normalize(requested)
    stored_norm = _normalize(stored)
    if not stored_norm:
        return False
    if requested_norm == stored_norm:
        return True

    def contains_phrase(text: str, phrase: str) -> bool:
        if not phrase:
            return False
        return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None

    for aliases in (level_aliases or {}).values():
        normalized_aliases = {_normalize(alias) for alias in aliases}
        requested_in_group = any(
            alias == requested_norm or contains_phrase(requested_norm, alias)
            for alias in normalized_aliases
        )
        stored_in_group = any(
            alias == stored_norm or contains_phrase(stored_norm, alias)
            for alias in normalized_aliases
        )
        if requested_in_group and stored_in_group:
            return True
    requested_number = _level_number(requested_norm)
    stored_number = _level_number(stored_norm)
    return requested_number is not None and stored_number == requested_number


def _resolve_canonical_type(ctx: BimRunContext, term: str) -> tuple[str, dict[str, Any]]:
    ontology = ctx.bim.ontology
    space_function = ontology.resolve_space_function(term)
    ifc_classes = ontology.resolve_element_classes(term) or []
    # Never invent a canonical value by trimming arbitrary stored/user text. In
    # particular, ``Switches`` must not become the nonexistent ``switche``.
    canonical_type = space_function or _normalize(term)
    return canonical_type, {
        "original_term": term,
        "canonical_type": canonical_type,
        "space_function": space_function,
        "ifc_classes": ifc_classes,
        "retrieval_terms": ontology.retrieval_terms_for(term),
        "absence_notes": ontology.absence_notes_for_query(term),
    }


def _catalog(ctx: BimRunContext) -> dict[str, Any]:
    common_operations = [
        "count", "count_distinct", "list", "group_count", "group_summary", "distinct",
        "sum", "average", "minimum", "maximum", "maximum_group_sum",
        "multi_group_count", "multi_group_summary", "coverage", "relationship_coverage",
    ]
    entities: dict[str, Any] = {}
    for name, entity in ctx.graph_contract.query_entities.items():
        entities[name] = {
            "description": entity.description,
            "operations": ["count"] if entity.kind == "project_graph" else common_operations,
            "default_select": entity.default_select,
            "fields": {
                field_name: {
                    "type": field.data_type,
                    "description": field.description,
                    "unit": field.unit,
                    "source_unit": field.source_unit or field.unit,
                    "conversion_factor": field.conversion_factor,
                }
                for field_name, field in entity.fields.items()
            },
        }
    for mapping_id, registered in ctx.schema_mappings.items():
        if not isinstance(registered, RegisteredSchemaMapping):
            continue
        proposal = registered.proposal
        entities[proposal.entity_name] = {
            "description": "Live-revalidated learned project mapping.",
            "mapping_id": mapping_id,
            "operations": common_operations,
            "identity": proposal.identity_property,
            "counting_unit": proposal.counting_unit,
            "fields": {
                field.semantic_name: {
                    "type": field.data_type,
                    "description": f"Mapped live property {field.property}.",
                    "unit": field.unit,
                    "source_unit": field.source_unit or field.unit,
                    "conversion_factor": field.conversion_factor,
                }
                for field in proposal.fields
            },
            "exact_value_bindings": {
                binding.semantic_name: [match.value for match in binding.matches]
                for binding in proposal.value_bindings
            },
            "relationships": {
                binding.semantic_name: binding.purpose
                for binding in proposal.relationship_bindings
            },
        }
    return {
        "entities": entities,
        "filter_operators": [
            "equals", "not_equals", "contains", "starts_with", "in",
            "greater_than", "greater_or_equal", "less_than", "less_or_equal",
            "exists", "is_missing",
        ],
        "rules": {
            "group_count_and_distinct": "set group_by",
            "group_summary": "set group_by and a numeric metric; returns count and summed metric per group",
            "maximum_group_sum": "set group_by and a numeric metric; returns the group with the largest summed metric",
            "numeric_aggregates": "set metric to a numeric field",
            "list": "select fields or use the entity defaults",
            "planning": "select the entity, fields, operation, and whether record details help answer the question",
            "missing_values": "is_missing already means absent, null, or blank; do not query extra sentinel values",
            "relationship_predicates": (
                "relationship_filters accept only exists/is_missing on live-validated named paths"
            ),
            "list_limit": "use the default 10 rows; the hard maximum is 20",
            "scope": "every operation is restricted to the authorized project",
        },
    }


def _project_graph_structure(
    ctx: BimRunContext,
    sample_limit: int = 3,
    include_properties: bool = False,
    focus: str = "",
) -> dict[str, Any]:
    """Return a bounded, project-scoped profile of live node and relationship types."""
    sample_limit = max(1, min(sample_limit, 10))
    path = ctx.graph_contract.authorization_path
    client = ctx.graph_contract.node(path.start_node)
    project = ctx.graph_contract.node("project")
    hub = ctx.graph_contract.node("bim_hub")
    if not client.identity_property or not project.identity_property:
        raise RuntimeError("The graph contract is missing hierarchy identity properties.")
    client_project = ctx.graph_contract.relationship_types[path.relationships[0]]
    project_hub = ctx.graph_contract.relationship_types[path.relationships[1]]
    scope_match = (
        f"MATCH (c:`{client.label}` {{`{client.identity_property}`: $client_id}}) "
        f"-[:`{client_project.name}`]->"
        f"(p:`{project.label}` {{`{project.identity_property}`: $project_id}}) "
        f"OPTIONAL MATCH (p)-[:`{project_hub.name}`]->(h:`{hub.label}`) "
    )
    parameters = {
        "client_id": ctx.scope.client_id,
        "project_id": ctx.scope.project_id,
        "allowed_sources": ctx.scope.allowed_sources,
        "sample_limit": sample_limit,
    }
    node_property_projection = (
        ", collect(DISTINCT keys(n)) AS property_key_groups"
        if include_properties else ", [] AS property_key_groups"
    )
    node_rows = ctx.bim.query(
        scope_match
        + "WITH c, p, h "
        + f"MATCH (n) WHERE n.`{path.source_property}` IN $allowed_sources "
        + "WITH c, p, h, collect(DISTINCT n) AS source_nodes "
        + "WITH [c, p, h] + source_nodes AS scoped_nodes "
        + "UNWIND scoped_nodes AS n WITH DISTINCT n WHERE n IS NOT NULL "
        + "RETURN labels(n) AS labels, count(DISTINCT n) AS node_count, "
        + "collect(DISTINCT coalesce(n.name, n.long_name, n.id, n.GlobalID, n.object_id))"
        + "[0..$sample_limit] AS sample_names"
        + node_property_projection
        + " ORDER BY node_count DESC",
        parameters,
    )
    relationship_property_projection = (
        "collect(DISTINCT keys(r)) AS property_key_groups, "
        if include_properties else "[] AS property_key_groups, "
    )
    relationship_rows = ctx.bim.query(
        scope_match
        + "WITH c, p, h "
        + "MATCH (a)-[r]->(b) "
        + f"WHERE (a.`{path.source_property}` IN $allowed_sources OR a = c OR a = p OR a = h) "
        + f"AND (b.`{path.source_property}` IN $allowed_sources OR b = c OR b = p OR b = h) "
        + "RETURN labels(a) AS from_labels, type(r) AS relationship_type, "
        + "labels(b) AS to_labels, count(r) AS relationship_count, "
        + relationship_property_projection
        + "collect(DISTINCT coalesce(r.name, r.id))[0..$sample_limit] AS sample_names "
        + "ORDER BY relationship_count DESC",
        parameters,
    )

    contract_nodes_by_label: dict[str, list[str]] = {}
    for contract_name, node in ctx.graph_contract.node_types.items():
        contract_nodes_by_label.setdefault(node.label, []).append(contract_name)
    node_types = []
    for row in node_rows:
        labels = sorted(row.get("labels") or [])
        property_keys = sorted({
            key
            for group in row.get("property_key_groups") or []
            for key in (group or [])
        })
        node_types.append({
            "labels": labels,
            "node_count": int(row.get("node_count") or 0),
            "sample_names": row.get("sample_names") or [],
            "properties": property_keys,
            "contract_node_types": sorted({
                contract_name
                for label in labels
                for contract_name in contract_nodes_by_label.get(label, [])
            }),
        })

    relationship_types = []
    for row in relationship_rows:
        property_keys = sorted({
            key
            for group in row.get("property_key_groups") or []
            for key in (group or [])
        })
        relationship_type = str(row.get("relationship_type") or "")
        from_labels = sorted(row.get("from_labels") or [])
        to_labels = sorted(row.get("to_labels") or [])
        contract_relationship_types = sorted(
            contract_name
            for contract_name, relationship in ctx.graph_contract.relationship_types.items()
            if relationship.name == relationship_type
            and ctx.graph_contract.node(relationship.from_node).label in from_labels
            and ctx.graph_contract.node(relationship.to_node).label in to_labels
        )
        relationship_types.append({
            "type": relationship_type,
            "from_labels": from_labels,
            "to_labels": to_labels,
            "relationship_count": int(row.get("relationship_count") or 0),
            "property_keys": property_keys,
            "sample_names": row.get("sample_names") or [],
            "contract_relationship_types": contract_relationship_types,
        })
    focus_phrases: list[str] = []
    if _normalize(focus):
        try:
            expanded_focus = _expanded_semantic_text(ctx, focus)
        except (AttributeError, TypeError, ValueError):
            expanded_focus = focus
        for item in re.split(r"[|;,]", expanded_focus):
            item = re.sub(r"^(?:user concept|bim vocabulary)\s*:\s*", "", item, flags=re.I)
            normalized = _normalize(item).replace("_", " ")
            if len(normalized) >= 2:
                focus_phrases.append(normalized)
        focus_phrases = list(dict.fromkeys(focus_phrases))[:40]

    def focus_matches(values: list[Any]) -> bool:
        haystack = _normalize(" ".join(str(value) for value in values)).replace("_", " ")
        return bool(focus_phrases) and any(phrase in haystack for phrase in focus_phrases)

    for node_type in node_types:
        node_type["focus_match"] = focus_matches([
            *node_type["labels"], *node_type["sample_names"], *node_type["properties"],
        ])
    for relationship in relationship_types:
        relationship["focus_match"] = focus_matches([
            relationship["type"], *relationship["from_labels"], *relationship["to_labels"],
            *relationship["sample_names"], *relationship["property_keys"],
        ])
    if focus_phrases:
        node_types.sort(key=lambda item: (-int(item["focus_match"]), -item["node_count"], item["labels"]))
        relationship_types.sort(key=lambda item: (
            -int(item["focus_match"]), -item["relationship_count"], item["type"]
        ))
    focus_match_count = sum(int(item["focus_match"]) for item in node_types) + sum(
        int(item["focus_match"]) for item in relationship_types
    )
    unique_node_labels = sorted({
        label for node_type in node_types for label in node_type["labels"]
    })
    unique_relationship_names = sorted({
        relationship["type"] for relationship in relationship_types
    })
    return {
        "project_id": ctx.scope.project_id,
        "authorized_source_count": len(ctx.scope.allowed_sources),
        "focus": focus or None,
        "focus_terms": focus_phrases,
        "focus_mode": "ranking_only" if focus_phrases else None,
        "focus_match_count": focus_match_count,
        "focus_fallback_used": bool(focus_phrases and focus_match_count == 0),
        "properties_included": include_properties,
        "node_types": node_types,
        "relationship_types": relationship_types,
        "unique_node_labels": unique_node_labels,
        "unique_relationship_names": unique_relationship_names,
        "node_type_count": len(node_types),
        "relationship_pattern_count": len(relationship_types),
        "unique_node_label_count": len(unique_node_labels),
        "unique_relationship_name_count": len(unique_relationship_names),
        "trusted_scope_property": path.source_property,
        "scope_note": "Only nodes and relationships belonging to the authorized project are included.",
        "usage_note": (
            "Focus only ranks results and never removes authorized schema. If focus_fallback_used is "
            "true, inspect the complete ranked inventory and use semantic search. Use live names to "
            "select a contract-backed declarative query plan; never generate or execute raw Cypher."
        ),
    }


def build_compact_model_profile(ctx: BimRunContext) -> ProjectModelProfile:
    """Build one bounded structural profile for reuse by isolated schema scouts.

    The profile is routing metadata only. Counts, exact bindings, and absence findings
    are always recomputed by the live query and verifier paths.
    """
    if ctx.model_profile:
        return ctx.model_profile
    cache_key = hashlib.sha256(_json({
        "client_id": ctx.scope.client_id,
        "project_id": ctx.scope.project_id,
        "allowed_sources": sorted(ctx.scope.allowed_sources),
        "contract_version": ctx.graph_contract.version,
        "schema_fingerprint": ctx.schema_fingerprint,
    }).encode("utf-8")).hexdigest()
    cacheable = bool(ctx.schema_fingerprint)
    with _MODEL_PROFILE_LOCK:
        cached = _MODEL_PROFILE_CACHE.get(cache_key) if cacheable else None
    if cached is not None:
        ctx.model_profile = cached
        return cached
    store = ctx.knowledge_store
    load_profile = getattr(store, "load_model_profile", None)
    if cacheable and callable(load_profile):
        persisted = load_profile(cache_key)
        if isinstance(persisted, dict):
            cached = ProjectModelProfile.model_validate(persisted)
            with _MODEL_PROFILE_LOCK:
                _MODEL_PROFILE_CACHE[cache_key] = cached
            ctx.model_profile = cached
            return cached
    structure = _project_graph_structure(ctx, sample_limit=2, include_properties=True)
    profile = ProjectModelProfile(
        profile_version=1,
        schema_fingerprint=ctx.schema_fingerprint,
        authorized_source_count=structure["authorized_source_count"],
        node_types=[
            {
                "labels": item["labels"],
                "node_count": item["node_count"],
                "properties": item["properties"],
            }
            for item in structure["node_types"]
        ],
        relationship_types=[
            {
                "type": item["type"],
                "from_labels": item["from_labels"],
                "to_labels": item["to_labels"],
                "relationship_count": item["relationship_count"],
            }
            for item in structure["relationship_types"]
        ],
        scope_note=structure["scope_note"],
        freshness_note=(
            "Routing cache only; answer values and absence claims require a live exhaustive query."
        ),
    )
    if cacheable:
        with _MODEL_PROFILE_LOCK:
            _MODEL_PROFILE_CACHE[cache_key] = profile
    save_profile = getattr(store, "save_model_profile", None)
    if cacheable and callable(save_profile):
        save_profile(
            cache_key=cache_key,
            client_id=ctx.scope.client_id,
            project_id=ctx.scope.project_id,
            schema_fingerprint=ctx.schema_fingerprint,
            profile=profile.model_dump(mode="json"),
        )
    ctx.model_profile = profile
    return profile


def _record_mapping(ctx: BimRunContext, entity: QueryEntity) -> tuple[str, str, str]:
    if not entity.node_type or not entity.identity_property or not entity.source_property:
        raise ValueError("The selected entity is not a record collection.")
    return (
        ctx.graph_contract.node(entity.node_type).label,
        entity.identity_property,
        entity.source_property,
    )


def _entity_from_registered_mapping(
    ctx: BimRunContext, plan: BimQueryPlan
) -> tuple[QueryEntity, str | None]:
    if not plan.mapping_id:
        try:
            # Versioned contract entities are already trusted semantic mappings.
            # The compiler still validates fields and enforces source scope.
            entity = ctx.graph_contract.query_entity(plan.entity)
            return entity, None
        except KeyError:
            if len(ctx.schema_mappings) == 1:
                plan.mapping_id = next(iter(ctx.schema_mappings))
            else:
                raise ValueError(
                    "Unknown record entities require one unambiguous registered live schema mapping."
                )
    registered = ctx.schema_mappings.get(plan.mapping_id)
    if not isinstance(registered, RegisteredSchemaMapping):
        raise ValueError(f"Unknown or expired schema mapping {plan.mapping_id!r}.")
    proposal = registered.proposal
    fields = {
        field.semantic_name: QueryField(
            property=field.property,
            data_type=field.data_type,
            unit=field.unit,
            source_unit=field.source_unit,
            conversion_factor=field.conversion_factor,
            conversion_basis=field.conversion_basis,
            ontology_kind=field.ontology_kind,
            description=f"Live-mapped field {field.semantic_name}.",
        )
        for field in proposal.fields
    }
    return QueryEntity(
        kind="records",
        description=f"Live-mapped {proposal.entity_name} records.",
        node_type=None,
        identity_property=proposal.identity_property,
        source_property=proposal.source_property,
        classification_source_property=proposal.classification_source_property,
        classification_name_property=proposal.classification_name_property,
        default_select=[name for name in ("object_id", "name", "ifc_class", "type", "level") if name in fields],
        fields=fields,
    ), proposal.label


def _registered_match_clause(mapping: RegisteredSchemaMapping | None, label: str) -> str:
    """Build a safe MATCH clause from a previously validated live relationship path."""
    if not mapping or not mapping.proposal.relationship_path:
        return f"MATCH (n:`{label}`)"
    parts = [f"(p0:`{mapping.proposal.relationship_path[0].from_label}`)"]
    for index, step in enumerate(mapping.proposal.relationship_path):
        target = "n" if index == len(mapping.proposal.relationship_path) - 1 else f"p{index + 1}"
        target_node = f"({target}:`{step.to_label}`)"
        relationship = f"[:`{step.relationship_type}`]"
        parts.append(
            f"-{relationship}->{target_node}"
            if step.direction == "outgoing"
            else f"<-{relationship}-{target_node}"
        )
    return "MATCH " + "".join(parts)


def _relationship_exists_expression(
    mapping: RegisteredSchemaMapping, semantic_name: str,
) -> tuple[str, list[str]]:
    binding = _relationship_binding(mapping, semantic_name)
    pattern, variables = _relationship_match_pattern(mapping, binding)
    scoped = " AND ".join(
        f"{variable}.`{mapping.proposal.source_property}` IN $allowed_sources"
        for variable in variables
    )
    return "EXISTS { MATCH " + pattern + (f" WHERE {scoped}" if scoped else "") + " }", variables


def _relationship_binding(
    mapping: RegisteredSchemaMapping, semantic_name: str,
) -> SchemaRelationshipBinding:
    binding = next(
        (
            item for item in mapping.proposal.relationship_bindings
            if item.semantic_name == semantic_name
        ),
        None,
    )
    if binding is None:
        allowed = sorted(item.semantic_name for item in mapping.proposal.relationship_bindings)
        raise ValueError(
            f"Unknown relationship binding {semantic_name!r}; allowed bindings are {allowed}."
        )
    if binding.steps[0].from_label != mapping.proposal.label:
        raise ValueError("A relationship binding must start at the mapped entity label.")
    return binding


def _relationship_match_pattern(
    mapping: RegisteredSchemaMapping, binding: SchemaRelationshipBinding,
) -> tuple[str, list[str]]:
    pattern = ["(n)"]
    variables: list[str] = []
    previous = mapping.proposal.label
    for index, step in enumerate(binding.steps):
        if step.from_label != previous:
            raise ValueError("Relationship binding steps must form one contiguous path.")
        variable = f"rb{index}"
        variables.append(variable)
        target = f"({variable}:`{step.to_label}`)"
        relationship = f"[:`{step.relationship_type}`]"
        pattern.append(
            f"-{relationship}->{target}"
            if step.direction == "outgoing" else f"<-{relationship}-{target}"
        )
        previous = step.to_label
    return "".join(pattern), variables


def _field(entity: QueryEntity, name: str) -> QueryField:
    try:
        return entity.fields[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown field {name!r}; allowed fields are {sorted(entity.fields)}."
        ) from exc


def _raw_filter_values(item: BimFilter) -> list[str]:
    values = [*item.values]
    if item.value != "":
        values.insert(0, item.value)
    return [str(value) for value in values]


def _semantic_values(
    ctx: BimRunContext,
    label: str,
    source_property: str,
    field: QueryField,
    values: list[str],
) -> tuple[list[str], list[str]]:
    notes: list[str] = []
    if field.ontology_kind == "canonical_type":
        resolved = [_resolve_canonical_type(ctx, value)[0] for value in values]
        return list(dict.fromkeys(resolved)), notes
    if field.ontology_kind == "ifc_class":
        resolved: list[str] = []
        for value in values:
            resolved.extend(ctx.bim.ontology.resolve_element_classes(value) or [value])
        return list(dict.fromkeys(resolved)), notes
    if field.ontology_kind == "level":
        rows = ctx.bim.query(
            f"MATCH (n:`{label}`) WHERE n.`{source_property}` IN $allowed_sources "
            f"AND n.`{field.property}` IS NOT NULL "
            f"RETURN DISTINCT n.`{field.property}` AS value",
            {"allowed_sources": ctx.scope.allowed_sources},
        )
        available = [str(row["value"]) for row in rows if row.get("value") is not None]
        matched = [
            stored for stored in available
            if any(_level_matches(requested, stored, ctx.graph_contract.level_aliases) for requested in values)
        ]
        if values and not matched:
            notes.append(f"No modelled level matched {', '.join(values)}.")
        return list(dict.fromkeys(matched or values)), notes
    if field.property == "load_bearing":
        aliases = {"true": "Yes", "yes": "Yes", "load bearing": "Yes", "false": "No", "no": "No"}
        return [aliases.get(_normalize(value), value) for value in values], notes
    return values, notes


def _compile_filters(
    ctx: BimRunContext,
    entity: QueryEntity,
    label: str,
    source_property: str,
    filters: list[BimFilter],
    exact_value_fields: set[str] | None = None,
) -> tuple[list[str], dict[str, Any], list[dict[str, Any]], list[str]]:
    clauses: list[str] = []
    parameters: dict[str, Any] = {"allowed_sources": ctx.scope.allowed_sources}
    normalized_filters: list[dict[str, Any]] = []
    limitations: list[str] = []
    for index, item in enumerate(filters):
        field = _field(entity, item.field)
        prop = f"n.`{field.property}`"
        key = f"filter_{index}"
        requested_values = _raw_filter_values(item)
        if item.field in (exact_value_fields or set()):
            # Registered value bindings are observations from the live graph,
            # not free text. Preserve them byte-for-byte through compilation.
            values, notes = requested_values, []
        else:
            values, notes = _semantic_values(
                ctx, label, source_property, field, requested_values
            )
        limitations.extend(notes)
        normalized_filters.append({
            "field": item.field,
            "operator": item.operator,
            "values": values,
            "requested_values": requested_values,
        })
        if item.operator in {"exists", "is_missing"}:
            exists = f"{prop} IS NOT NULL AND trim(toString({prop})) <> ''"
            clauses.append(f"({exists})" if item.operator == "exists" else f"(NOT ({exists}))")
            continue
        if not values:
            raise ValueError(f"Filter {item.field!r} requires a value.")
        if item.operator in {"greater_than", "greater_or_equal", "less_than", "less_or_equal"}:
            if field.data_type != "number":
                raise ValueError(f"Numeric comparison is not allowed for {item.field!r}.")
            try:
                parameters[key] = float(values[0]) / field.conversion_factor
            except ValueError as exc:
                raise ValueError(f"Filter {item.field!r} requires a numeric value.") from exc
            symbol = {
                "greater_than": ">", "greater_or_equal": ">=",
                "less_than": "<", "less_or_equal": "<=",
            }[item.operator]
            clauses.append(f"toFloat({prop}) {symbol} ${key}")
            continue
        if field.data_type == "number" and item.operator in {"equals", "not_equals", "in"}:
            try:
                parameters[key] = [float(value) / field.conversion_factor for value in values]
            except ValueError as exc:
                raise ValueError(f"Filter {item.field!r} requires numeric values.") from exc
            numeric_clause = f"toFloat({prop}) IN ${key}"
            clauses.append(
                f"NOT ({numeric_clause})" if item.operator == "not_equals" else numeric_clause
            )
            continue
        if field.data_type == "number" and item.operator in {"contains", "starts_with"}:
            raise ValueError(f"Text matching is not allowed for numeric field {item.field!r}.")
        lowered = [_normalize(value) for value in values]
        if item.operator == "equals":
            parameters[key] = lowered
            clauses.append(f"toLower(trim(toString({prop}))) IN ${key}")
        elif item.operator == "not_equals":
            parameters[key] = lowered
            clauses.append(f"NOT toLower(trim(toString({prop}))) IN ${key}")
        elif item.operator == "in":
            parameters[key] = lowered
            clauses.append(f"toLower(trim(toString({prop}))) IN ${key}")
        elif item.operator in {"contains", "starts_with"}:
            parameters[key] = lowered[0]
            predicate = "CONTAINS" if item.operator == "contains" else "STARTS WITH"
            clauses.append(f"toLower(toString({prop})) {predicate} ${key}")
        else:
            raise ValueError(f"Unsupported filter operator {item.operator!r}.")
    return clauses, parameters, normalized_filters, limitations


def _clean_value(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 500:
        return value[:497] + "..."
    return value


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Embedding vectors must have the same non-zero length.")
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _rank_embedding_candidates(
    query_embedding: list[float],
    candidate_embeddings: list[list[float]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(candidate_embeddings) != len(candidates):
        raise ValueError("Every semantic candidate requires one embedding.")
    ranked = []
    for candidate, embedding in zip(candidates, candidate_embeddings):
        ranked.append({
            **candidate,
            "similarity": round(_cosine_similarity(query_embedding, embedding), 6),
        })
    return sorted(ranked, key=lambda item: (-item["similarity"], item["node_ref"]))


def _local_semantic_embedding(value: str, dimensions: int = 384) -> list[float]:
    """Build a deterministic local vector from words and character n-grams.

    Anthropic does not expose an embeddings endpoint. Ontology expansion supplies
    the cross-language vocabulary; signed feature hashing then provides fuzzy,
    network-free similarity without retaining an OpenAI runtime dependency.
    """
    normalized = _normalize(value).replace("_", " ")
    words = re.findall(r"\w+", normalized, flags=re.UNICODE)
    features = [f"word:{word}" for word in words]
    for word in words:
        padded = f"  {word}  "
        features.extend(
            f"char:{padded[index:index + 3]}"
            for index in range(max(len(padded) - 2, 0))
        )
    vector = [0.0] * dimensions
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign
    return vector


def _local_semantic_embeddings(values: list[str]) -> list[list[float]]:
    return [_local_semantic_embedding(value) for value in values]


def _rank_hybrid_candidates(
    semantic_text: str,
    query_embedding: list[float],
    candidate_embeddings: list[list[float]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine local fuzzy vectors with deterministic exact/token vocabulary matches."""
    ranked = _rank_embedding_candidates(query_embedding, candidate_embeddings, candidates)
    phrases = []
    for item in re.split(r"[|;,]", semantic_text):
        item = re.sub(r"^(?:user concept|bim vocabulary)\s*:\s*", "", item, flags=re.I)
        normalized = _normalize(item).replace("_", " ")
        if len(normalized) >= 2:
            phrases.append(normalized)
    phrase_tokens = [set(re.findall(r"\w+", phrase)) for phrase in phrases]
    for item in ranked:
        target = _normalize(str(item.get("embedding_text") or "")).replace("_", " ")
        target_tokens = set(re.findall(r"\w+", target))
        exact = any(
            re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", target) is not None
            for phrase in phrases
        )
        lexical = 1.0 if exact else max(
            (
                len(tokens & target_tokens) / len(tokens)
                for tokens in phrase_tokens if tokens
            ),
            default=0.0,
        )
        embedding_similarity = float(item["similarity"])
        item["embedding_similarity"] = embedding_similarity
        item["lexical_similarity"] = round(lexical, 6)
        lexical_score = 1.0 if exact else lexical * 0.98
        item["similarity"] = round(max(embedding_similarity, lexical_score), 6)
    return sorted(ranked, key=lambda item: (-item["similarity"], item["node_ref"]))


def _format_filters(filters: list[dict[str, Any]]) -> str:
    if not filters:
        return "without additional filters"
    parts = []
    for item in filters:
        values = ", ".join(
            str(value) for value in (item.get("requested_values") or item["values"])
        )
        parts.append(
            f"{item['field'].replace('_', ' ')} "
            f"{item['operator'].replace('_', ' ')} {values}".strip()
        )
    return "where " + " and ".join(parts)


def _format_rows(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    formatted: list[str] = []
    for row in rows:
        values = [
            f"{field.replace('_', ' ')}: {row.get(field)}"
            for field in fields
            if row.get(field) not in (None, "")
        ]
        formatted.append(", ".join(values) or "no populated selected fields")
    return formatted


def _display_number(value: Any) -> str:
    if value is None:
        return "not modelled"
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _pluralize(value: str, count: int) -> str:
    noun = value.replace("_", " ").strip()
    if count == 1:
        return noun
    if noun.endswith("y") and not noun.endswith(("ay", "ey", "oy", "uy")):
        return noun[:-1] + "ies"
    if noun.endswith("s"):
        return noun
    return noun + "s"


def _space_list_headline(
    matched_count: int,
    displayed_count: int,
    normalized_filters: list[dict[str, Any]],
) -> str:
    type_filter = next(
        (item for item in normalized_filters if item["field"] == "type"), None
    )
    level_filter = next(
        (item for item in normalized_filters if item["field"] == "level"), None
    )
    type_values = (type_filter or {}).get("values") or []
    noun = _pluralize(str(type_values[0]), matched_count) if len(type_values) == 1 else _pluralize("space", matched_count)
    verb = "is" if matched_count == 1 else "are"
    level_text = ""
    requested_levels = (level_filter or {}).get("requested_values") or []
    if len(requested_levels) == 1:
        level_text = f" on the {requested_levels[0]}"
    headline = f"There {verb} {matched_count} {noun}{level_text}."
    if displayed_count < matched_count:
        headline += f" Showing {displayed_count} of {matched_count} detailed records."
    return headline


def _format_space_rows(rows: list[dict[str, Any]]) -> list[str]:
    formatted: list[str] = []
    for row in rows:
        ifc_class = str(row.get("ifc_class") or "IfcSpace")
        object_id = str(row.get("object_id") or row.get("global_id") or "unknown ID")
        lines = [f"{ifc_class} {object_id}"]
        values = [
            ("Name", row.get("name")),
            ("Level", row.get("level")),
            (
                "Area",
                f"{_display_number(row.get('area_m2'))} m²"
                if row.get("area_m2") is not None else None,
            ),
        ]
        segment = row.get("segment")
        if segment not in (None, ""):
            values.append(("Segment", segment))
        values.extend([
            ("Owner", row.get("owner")),
            ("Room Count", row.get("room_count")),
            ("Dwelling Type", row.get("dwelling_type")),
        ])
        lines.extend(
            f"  - {label}: {value}"
            for label, value in values
            if value not in (None, "")
        )
        formatted.append("\n".join(lines))
    return formatted


def _classification_limitations(
    ctx: BimRunContext,
    entity: QueryEntity,
    label: str,
    source_property: str,
    normalized_filters: list[dict[str, Any]],
) -> list[str]:
    if not entity.classification_source_property:
        return []
    type_filters = [
        item for item in normalized_filters
        if item["field"] == "type" and item["operator"] in {"equals", "in"}
    ]
    if not type_filters:
        return []
    type_field = _field(entity, "type")
    name_property = entity.classification_name_property or entity.identity_property
    issues: list[str] = []
    for item in type_filters:
        rows = ctx.bim.query(
            f"MATCH (n:`{label}`) WHERE n.`{source_property}` IN $allowed_sources "
            f"AND toLower(toString(n.`{type_field.property}`)) IN $types "
            f"AND n.`{entity.classification_source_property}` IS NOT NULL "
            f"RETURN DISTINCT n.`{name_property}` AS name, "
            f"n.`{entity.classification_source_property}` AS source_term LIMIT 20",
            {
                "allowed_sources": ctx.scope.allowed_sources,
                "types": [_normalize(value) for value in item["values"]],
            },
        )
        for row in rows:
            source_term = str(row.get("source_term") or "")
            resolved = ctx.bim.ontology.resolve_space_function(source_term)
            if resolved not in item["values"]:
                issues.append(
                    f"{row.get('name') or 'An element'} is stored under type "
                    f"{', '.join(item['values'])}, but its source term {source_term!r} is not "
                    "resolved to that type by the current ontology."
                )
    return issues


def _count_project_nodes(
    ctx: BimRunContext, cypher: CypherQueryHandler | None = None
) -> dict[str, Any]:
    path = ctx.graph_contract.authorization_path
    client = ctx.graph_contract.node(path.start_node)
    project = ctx.graph_contract.node("project")
    hub = ctx.graph_contract.node("bim_hub")
    if not client.identity_property or not project.identity_property:
        raise RuntimeError("The graph contract is missing hierarchy identity properties.")
    client_project = ctx.graph_contract.relationship_types[path.relationships[0]]
    project_hub = ctx.graph_contract.relationship_types[path.relationships[1]]
    execute = cypher.execute if cypher is not None else ctx.bim.query
    rows = execute(
        f"MATCH (c:`{client.label}` {{`{client.identity_property}`: $client_id}}) "
        f"-[:`{client_project.name}`]->"
        f"(p:`{project.label}` {{`{project.identity_property}`: $project_id}}) "
        f"OPTIONAL MATCH (p)-[:`{project_hub.name}`]->(h:`{hub.label}`) "
        "WITH collect(DISTINCT c) + collect(DISTINCT p) + collect(DISTINCT h) AS hierarchy_nodes "
        f"MATCH (n) WHERE n.`{path.source_property}` IN $allowed_sources "
        "WITH hierarchy_nodes, collect(DISTINCT n) AS source_nodes "
        "UNWIND hierarchy_nodes + source_nodes AS scoped_node "
        "RETURN count(DISTINCT scoped_node) AS total_nodes, "
        "size(source_nodes) AS source_scoped_nodes, size(hierarchy_nodes) AS hierarchy_nodes",
        {
            "client_id": ctx.scope.client_id,
            "project_id": ctx.scope.project_id,
            "allowed_sources": ctx.scope.allowed_sources,
        },
    )
    counts = rows[0] if rows else {}
    total = int(counts.get("total_nodes") or 0)
    return {
        "rows": [{
            "total_nodes": total,
            "source_scoped_nodes": int(counts.get("source_scoped_nodes") or 0),
            "hierarchy_nodes": int(counts.get("hierarchy_nodes") or 0),
        }],
        "matched_count": total,
        "claim": {
            "statement": f"The authorized project graph contains {total} nodes.",
            "value": total,
            "unit": "nodes",
            "basis": "Distinct source-scoped nodes plus the authorized Client, Project, and BIMHub nodes.",
        },
        "limitations": ["This is not a count of the entire Neo4j database."],
    }


def _validate_plan(entity: QueryEntity, plan: BimQueryPlan) -> None:
    if plan.role == "rejected":
        raise ValueError("A rejected plan cannot be executed.")
    if entity.kind == "project_graph":
        if plan.operation != "count" or plan.filters or plan.relationship_filters:
            raise ValueError("project_graph supports only an unfiltered count operation.")
        return
    for item in plan.filters:
        _field(entity, item.field)
    if plan.operation == "coverage":
        if not plan.coverage_field:
            raise ValueError("coverage requires coverage_field.")
        _field(entity, plan.coverage_field)
    if plan.operation == "relationship_coverage" and not plan.relationship:
        raise ValueError("relationship_coverage requires relationship.")
    if plan.operation in {"group_count", "group_summary", "maximum_group_sum", "distinct", "count_distinct"}:
        if not plan.group_by:
            raise ValueError(f"{plan.operation} requires group_by.")
        _field(entity, plan.group_by)
    if plan.operation in {"group_summary", "maximum_group_sum"}:
        if not plan.metric:
            raise ValueError("group_summary requires a numeric metric.")
        if _field(entity, plan.metric).data_type != "number":
            raise ValueError(f"Metric {plan.metric!r} is not numeric.")
    if plan.operation in {"multi_group_count", "multi_group_summary"}:
        if not plan.group_by_fields:
            raise ValueError(f"{plan.operation} requires group_by_fields.")
        if len(set(plan.group_by_fields)) != len(plan.group_by_fields):
            raise ValueError("group_by_fields cannot contain duplicates.")
        for field_name in plan.group_by_fields:
            _field(entity, field_name)
    if plan.operation == "multi_group_summary":
        if not plan.metric or _field(entity, plan.metric).data_type != "number":
            raise ValueError("multi_group_summary requires a numeric metric.")
    if plan.operation in {"sum", "average", "minimum", "maximum"}:
        if not plan.metric:
            raise ValueError(f"{plan.operation} requires a numeric metric.")
        if _field(entity, plan.metric).data_type != "number":
            raise ValueError(f"Metric {plan.metric!r} is not numeric.")
    if plan.operation == "list":
        for field_name in plan.select or entity.default_select:
            _field(entity, field_name)
        if plan.sort_by:
            _field(entity, plan.sort_by)


def _execute_plan(ctx: BimRunContext, plan: BimQueryPlan) -> dict[str, Any]:
    cypher = CypherQueryHandler(ctx.bim.query, ctx.scope.allowed_sources)
    entity, live_label = _entity_from_registered_mapping(ctx, plan)
    entity_name = plan.entity.replace("_", " ")
    _validate_plan(entity, plan)
    registered_mapping = ctx.schema_mappings.get(plan.mapping_id)
    if plan.relationship_filters:
        if not isinstance(registered_mapping, RegisteredSchemaMapping):
            raise ValueError("Relationship predicates require a registered live mapping.")
        for relationship_filter in plan.relationship_filters:
            _relationship_exists_expression(
                registered_mapping, relationship_filter.relationship
            )
    if isinstance(registered_mapping, RegisteredSchemaMapping):
        for binding in registered_mapping.proposal.value_bindings:
            selected_values = {_normalize(match.value) for match in binding.matches}
            plan_values = {
                _normalize(value)
                for item in plan.filters
                if item.field == binding.semantic_name
                for value in _raw_filter_values(item)
            }
            if not selected_values.issubset(plan_values):
                raise ValueError(
                    f"The query plan must use the registered exact value binding for "
                    f"{binding.user_concept!r} on semantic field {binding.semantic_name!r}."
                )
    if entity.kind == "project_graph":
        result = _count_project_nodes(ctx, cypher)
    else:
        if live_label is None:
            label, identity_property, source_property = _record_mapping(ctx, entity)
        else:
            if not entity.identity_property or not entity.source_property:
                raise ValueError("The registered schema mapping is incomplete.")
            label, identity_property, source_property = (
                live_label, entity.identity_property, entity.source_property
            )
        match_clause = _registered_match_clause(
            registered_mapping if isinstance(registered_mapping, RegisteredSchemaMapping) else None,
            label,
        )
        exact_value_fields = {
            binding.semantic_name for binding in registered_mapping.proposal.value_bindings
        } if isinstance(registered_mapping, RegisteredSchemaMapping) else set()
        clauses, parameters, normalized_filters, limitations = _compile_filters(
            ctx, entity, label, source_property, plan.filters, exact_value_fields
        )
        if isinstance(registered_mapping, RegisteredSchemaMapping):
            for relationship_filter in plan.relationship_filters:
                exists_expression, _ = _relationship_exists_expression(
                    registered_mapping, relationship_filter.relationship
                )
                clauses.append(
                    exists_expression
                    if relationship_filter.operator == "exists"
                    else f"NOT ({exists_expression})"
                )
                normalized_filters.append({
                    "field": f"relationship:{relationship_filter.relationship}",
                    "operator": relationship_filter.operator,
                    "values": [],
                    "requested_values": [],
                })
        limitations.extend(_classification_limitations(
            ctx, entity, label, source_property, normalized_filters
        ))
        where = f"n.`{source_property}` IN $allowed_sources"
        if clauses:
            where += " AND " + " AND ".join(f"({clause})" for clause in clauses)
        identity = f"n.`{identity_property}`"
        filter_text = _format_filters(normalized_filters)
        rows: list[dict[str, Any]]
        matched_count: int
        if plan.operation == "relationship_coverage":
            if not isinstance(registered_mapping, RegisteredSchemaMapping):
                raise ValueError("relationship_coverage requires a registered live mapping.")
            binding = _relationship_binding(registered_mapping, plan.relationship)
            if binding.target_identity_property:
                relationship_pattern, relationship_variables = _relationship_match_pattern(
                    registered_mapping, binding
                )
                target = relationship_variables[-1]
                target_identity = f"{target}.`{binding.target_identity_property}`"
                relationship_scope = " AND ".join(
                    f"{variable}.`{source_property}` IN $allowed_sources"
                    for variable in relationship_variables
                )
                cardinality_ok = (
                    "target_node_count > 0"
                    if binding.target_cardinality == "any"
                    else "target_node_count = 1"
                )
                valid_topology = (
                    f"target_node_count = target_identity_count AND ({cardinality_ok})"
                )
                rows = cypher.execute(
                    f"{match_clause} WHERE {where} "
                    f"WITH DISTINCT n, {identity} AS root_identity "
                    f"OPTIONAL MATCH {relationship_pattern} "
                    + (f"WHERE {relationship_scope} " if relationship_scope else "")
                    + f"WITH root_identity, count(DISTINCT {target}) AS target_node_count, "
                    f"count(DISTINCT CASE WHEN {target_identity} IS NOT NULL "
                    f"AND trim(toString({target_identity})) <> '' THEN {target_identity} END) "
                    f"AS target_identity_count "
                    f"RETURN count(root_identity) AS candidate_count, "
                    f"sum(CASE WHEN target_node_count > 0 THEN 1 ELSE 0 END) AS connected_count, "
                    f"sum(CASE WHEN {valid_topology} THEN 1 ELSE 0 END) AS resolved_count, "
                    f"sum(CASE WHEN target_node_count = 0 THEN 1 ELSE 0 END) AS missing_count, "
                    f"sum(CASE WHEN target_node_count > 0 AND NOT ({valid_topology}) "
                    f"THEN 1 ELSE 0 END) AS unresolved_count, "
                    f"sum(target_identity_count) AS linked_target_count",
                    parameters,
                )
                row = rows[0] if rows else {}
                candidate_count = int(row.get("candidate_count") or 0)
                connected_count = int(row.get("connected_count") or 0)
                resolved_count = int(row.get("resolved_count") or 0)
                missing_count = int(row.get("missing_count") or 0)
                unresolved_count = int(row.get("unresolved_count") or 0)
                linked_target_count = int(row.get("linked_target_count") or 0)
                matched_count = resolved_count
                exhaustive = unresolved_count == 0
                caveats = []
                if unresolved_count:
                    caveats.append(
                        f"{unresolved_count} records have a relationship edge but its target identity "
                        "or required cardinality is unresolved; they are reported as unknown, not connected."
                    )
                statement = (
                    f"The registered {plan.relationship.replace('_', ' ')} topology resolves to an "
                    f"identified target for {resolved_count} of {candidate_count} project-scoped "
                    f"{entity_name}; {missing_count} have no path and {unresolved_count} have an "
                    "unresolved or conflicting path."
                )
                claim = {
                    "statement": statement,
                    "value": resolved_count,
                    "unit": "topology-validated connected records",
                    "basis": (
                        f"Coverage of the live-validated {plan.relationship!r} binding rooted at "
                        f"{label}, requiring populated {binding.steps[-1].to_label}."
                        f"{binding.target_identity_property} target identities and "
                        f"{binding.target_cardinality.replace('_', ' ')} target cardinality."
                    ),
                    "details": [
                        f"Raw records with a path: {connected_count}",
                        f"Resolved target identities across paths: {linked_target_count}",
                    ],
                    "coverage": {
                        "candidate_count": candidate_count,
                        "evaluated_count": candidate_count - unresolved_count,
                        "matched_count": resolved_count,
                        "missing_count": missing_count,
                        "unknown_count": unresolved_count,
                        "excluded_count": 0,
                        "exhaustive": exhaustive,
                    },
                    "method": f"{binding.evidence_kind}_relationship_coverage",
                    "source_tags": [
                        f"relationship:{plan.relationship}",
                        f"relationship-evidence:{binding.evidence_kind}",
                        f"{label}.{identity_property}",
                        f"{binding.steps[-1].to_label}.{binding.target_identity_property}",
                    ],
                    "caveats": caveats,
                }
            else:
                exists_expression, _ = _relationship_exists_expression(
                    registered_mapping, plan.relationship
                )
                rows = cypher.execute(
                    f"{match_clause} WHERE {where} "
                    f"RETURN count(DISTINCT {identity}) AS candidate_count, "
                    f"count(DISTINCT CASE WHEN {exists_expression} THEN {identity} END) AS connected_count",
                    parameters,
                )
                candidate_count = int(rows[0].get("candidate_count") or 0) if rows else 0
                connected_count = int(rows[0].get("connected_count") or 0) if rows else 0
                missing_count = max(candidate_count - connected_count, 0)
                matched_count = connected_count
                claim = {
                    "statement": (
                        f"The registered {plan.relationship.replace('_', ' ')} path exists for "
                        f"{connected_count} of {candidate_count} project-scoped {entity_name}; "
                        f"{missing_count} records have no such path."
                    ),
                    "value": connected_count,
                    "unit": "connected records",
                    "basis": (
                        f"Exhaustive coverage of the live-validated {plan.relationship!r} relationship "
                        f"binding rooted at {label}."
                    ),
                    "coverage": {
                        "candidate_count": candidate_count,
                        "evaluated_count": candidate_count,
                        "matched_count": connected_count,
                        "missing_count": missing_count,
                        "unknown_count": 0,
                        "excluded_count": 0,
                        "exhaustive": True,
                    },
                    "method": "relationship_coverage",
                    "source_tags": [
                        f"relationship:{plan.relationship}", f"{label}.{identity_property}",
                    ],
                }
        elif plan.operation == "coverage":
            covered = _field(entity, plan.coverage_field)
            rows = cypher.execute(
                f"{match_clause} WHERE {where} "
                f"RETURN count(DISTINCT {identity}) AS candidate_count, "
                f"count(DISTINCT CASE WHEN n.`{covered.property}` IS NOT NULL "
                f"AND trim(toString(n.`{covered.property}`)) <> '' THEN {identity} END) AS populated_count",
                parameters,
            )
            candidate_count = int(rows[0].get("candidate_count") or 0) if rows else 0
            populated_count = int(rows[0].get("populated_count") or 0) if rows else 0
            missing_count = max(candidate_count - populated_count, 0)
            matched_count = populated_count
            fill_rate = (populated_count / candidate_count) if candidate_count else None
            statement = (
                f"The {plan.coverage_field.replace('_', ' ')} property is populated on "
                f"{populated_count} of {candidate_count} project-scoped {entity_name}; "
                f"{missing_count} records are missing it."
            )
            claim = {
                "statement": statement,
                "value": populated_count,
                "unit": "populated records",
                "basis": f"Exhaustive null/blank coverage scan of {label}.{covered.property} in scope.",
                "coverage": {
                    "candidate_count": candidate_count,
                    "evaluated_count": candidate_count,
                    "matched_count": populated_count,
                    "missing_count": missing_count,
                    "unknown_count": 0,
                    "excluded_count": 0,
                    "exhaustive": True,
                },
                "method": "coverage",
                "source_tags": [f"{label}.{covered.property}"],
                "details": [
                    "Fill rate: " + (f"{fill_rate:.1%}" if fill_rate is not None else "not applicable")
                ],
                "total_count": candidate_count,
                "displayed_count": 1 if candidate_count else 0,
            }
        elif plan.operation == "count":
            rows = cypher.execute(
                f"{match_clause} WHERE {where} "
                f"RETURN count(DISTINCT {identity}) AS count",
                parameters,
            )
            matched_count = int(rows[0].get("count") or 0) if rows else 0
            missing_level = next(
                (note for note in limitations if note.startswith("No modelled level matched")), None
            )
            if plan.entity == "spaces" and plan.include_details and matched_count and not missing_level:
                selected = entity.default_select
                projections = ", ".join(
                    f"n.`{_field(entity, name).property}` AS `{name}`" for name in selected
                )
                parameters["limit"] = plan.limit
                rows = cypher.execute(
                    f"{match_clause} WHERE {where} RETURN {projections} "
                    f"ORDER BY {identity} ASC LIMIT $limit",
                    parameters,
                )
                rows = [{key: _clean_value(value) for key, value in row.items()} for row in rows]
                details = _format_space_rows(rows)
                claim = {
                    "statement": _space_list_headline(
                        matched_count, len(details), normalized_filters
                    ),
                    "value": matched_count,
                    "unit": entity_name,
                    "basis": (
                        f"Distinct {label}.{identity_property} values plus a bounded technical "
                        "record breakdown in the authorized source scope."
                    ),
                    "details": details,
                    "total_count": matched_count,
                    "displayed_count": len(details),
                    "coverage": {
                        "candidate_count": matched_count, "evaluated_count": matched_count,
                        "matched_count": matched_count, "missing_count": 0,
                        "unknown_count": 0, "excluded_count": 0, "exhaustive": True,
                    },
                    "method": "count_distinct_identity",
                    "source_tags": [f"{label}.{identity_property}"],
                }
            else:
                claim = {
                    "statement": (
                        f"{missing_level} The requested count cannot be determined."
                        if missing_level
                        else f"The authorized project contains {matched_count} {entity_name} {filter_text}."
                    ),
                    "value": None if missing_level else matched_count,
                    "unit": entity_name,
                    "basis": f"Distinct {label}.{identity_property} values in the authorized source scope.",
                    "coverage": {
                        "candidate_count": matched_count, "evaluated_count": matched_count,
                        "matched_count": matched_count, "missing_count": 0,
                        "unknown_count": 0, "excluded_count": 0, "exhaustive": True,
                    },
                    "method": "count_distinct_identity",
                    "source_tags": [f"{label}.{identity_property}"],
                }
        elif plan.operation == "count_distinct":
            group = _field(entity, plan.group_by)
            rows = cypher.execute(
                f"{match_clause} WHERE {where} AND n.`{group.property}` IS NOT NULL "
                f"RETURN count(DISTINCT n.`{group.property}`) AS count, "
                f"count(DISTINCT {identity}) AS populated_records",
                parameters,
            )
            matched_count = int(rows[0].get("count") or 0) if rows else 0
            populated_records = int(rows[0].get("populated_records") or 0) if rows else 0
            candidate_rows = cypher.execute(
                f"{match_clause} WHERE {where} RETURN count(DISTINCT {identity}) AS count",
                parameters,
            )
            candidate_count = int(candidate_rows[0].get("count") or 0) if candidate_rows else 0
            level_filter = next(
                (item for item in normalized_filters if item["field"] == "level"), None
            )
            if plan.entity == "spaces" and plan.group_by == "dwelling_unit_number":
                location = ""
                if level_filter:
                    requested = (level_filter.get("requested_values") or ["requested level"])[0]
                    location = f" on the {requested}"
                statement = f"There are {matched_count} physical apartments{location}."
                unit = "physical apartments"
            else:
                statement = (
                    f"The authorized project contains {matched_count} distinct "
                    f"{plan.group_by.replace('_', ' ')} values {filter_text}."
                )
                unit = f"distinct {plan.group_by.replace('_', ' ')} values"
            claim = {
                "statement": statement,
                "value": matched_count,
                "unit": unit,
                "basis": f"Distinct populated {label}.{group.property} values in the authorized source scope.",
                "coverage": {
                    "candidate_count": candidate_count, "evaluated_count": candidate_count,
                    "matched_count": populated_records,
                    "missing_count": max(candidate_count - populated_records, 0),
                    "unknown_count": 0, "excluded_count": 0, "exhaustive": True,
                },
                "method": "count_distinct_property",
                "source_tags": [f"{label}.{group.property}", f"{label}.{identity_property}"],
            }
        elif plan.operation == "list":
            selected = plan.select or entity.default_select
            projections = ", ".join(
                f"n.`{_field(entity, name).property}` AS `{name}`" for name in selected
            )
            count_rows = cypher.execute(
                f"{match_clause} WHERE {where} RETURN count(DISTINCT {identity}) AS count",
                parameters,
            )
            matched_count = int(count_rows[0].get("count") or 0) if count_rows else 0
            order_fields = []
            if plan.sort_by:
                direction = "DESC" if plan.sort_order == "descending" else "ASC"
                order_fields.append(f"n.`{_field(entity, plan.sort_by).property}` {direction}")
            order_fields.append(f"{identity} ASC")
            order = " ORDER BY " + ", ".join(order_fields)
            parameters["limit"] = plan.limit
            rows = cypher.execute(
                f"{match_clause} WHERE {where} RETURN {projections}"
                f"{order} LIMIT $limit",
                parameters,
            )
            rows = [{key: _clean_value(value) for key, value in row.items()} for row in rows]
            details = _format_space_rows(rows) if plan.entity == "spaces" else _format_rows(rows, selected)
            claim = {
                "statement": (
                    (
                        "No project-scoped permit knowledge records were found. "
                        "Compliance cannot be assessed from the available BIM evidence."
                    )
                    if matched_count == 0 and plan.entity == "permit_knowledge"
                    else f"No project-scoped {entity_name} were found {filter_text}."
                    if matched_count == 0
                    else _space_list_headline(matched_count, len(details), normalized_filters)
                    if plan.entity == "spaces"
                    else f"Found {matched_count} project-scoped {entity_name} {filter_text}."
                ),
                "value": matched_count,
                "unit": entity_name,
                "basis": f"Project-scoped {label} records; output is limited to {plan.limit} rows.",
                "details": details,
                "total_count": matched_count,
                "displayed_count": len(details),
                "coverage": {
                    "candidate_count": matched_count, "evaluated_count": matched_count,
                    "matched_count": matched_count, "missing_count": 0,
                    "unknown_count": 0, "excluded_count": 0, "exhaustive": True,
                },
                "method": "bounded_list_over_complete_population",
                "source_tags": [
                    f"{label}.{identity_property}",
                    *(f"{label}.{_field(entity, name).property}" for name in selected),
                ],
            }
        elif plan.operation in {"multi_group_count", "multi_group_summary"}:
            groups = [_field(entity, name) for name in plan.group_by_fields]
            group_expressions = [
                (
                    f"coalesce(toString(toFloat(n.`{field.property}`) * "
                    f"{field.conversion_factor!r}), 'not modelled')"
                    if field.data_type == "number" else
                    f"coalesce(toString(n.`{field.property}`), 'not modelled')"
                )
                for field in groups
            ]
            projections = ", ".join(
                f"{expression} AS group_{index}"
                for index, expression in enumerate(group_expressions)
            )
            parameters["limit"] = plan.limit
            metric = _field(entity, plan.metric) if plan.operation == "multi_group_summary" else None
            metric_projection = (
                f", sum(toFloat(n.`{metric.property}`) * {metric.conversion_factor!r}) AS metric_value, "
                f"count(n.`{metric.property}`) AS metric_records"
                if metric else ""
            )
            order_value = "metric_value" if metric else "count"
            rows = cypher.execute(
                f"{match_clause} WHERE {where} "
                f"RETURN {projections}, count(DISTINCT {identity}) AS count{metric_projection} "
                f"ORDER BY {order_value} DESC LIMIT $limit",
                parameters,
            )
            count_rows = cypher.execute(
                f"{match_clause} WHERE {where} "
                f"RETURN count(DISTINCT {identity}) AS total_records"
                + (
                    f", count(n.`{metric.property}`) AS metric_records, "
                    f"sum(toFloat(n.`{metric.property}`) * "
                    f"{metric.conversion_factor!r}) AS metric_total"
                    if metric else ""
                ),
                parameters,
            )
            matched_count = int(count_rows[0].get("total_records") or 0) if count_rows else 0
            total_metric_records = (
                int(count_rows[0].get("metric_records") or 0) if count_rows and metric else 0
            )
            total_metric_value = (
                count_rows[0].get("metric_total") if count_rows and metric else None
            )
            details = []
            populated_metric_records = 0
            for row in rows:
                dimensions = ", ".join(
                    f"{name.replace('_', ' ')}={row.get(f'group_{index}')}"
                    + (
                        f" {groups[index].unit}"
                        if groups[index].data_type == "number"
                        and row.get(f"group_{index}") != "not modelled"
                        and groups[index].unit else ""
                    )
                    for index, name in enumerate(plan.group_by_fields)
                )
                detail = f"{dimensions}: {int(row.get('count') or 0)} records"
                if metric:
                    populated_metric_records += int(row.get("metric_records") or 0)
                    detail += f", {_display_number(row.get('metric_value'))} {metric.unit or plan.metric}"
                details.append(detail + ".")
            if metric and matched_count and populated_metric_records == 0:
                limitations.append(
                    f"Matching {entity_name} exist, but {plan.metric.replace('_', ' ')} is unpopulated; "
                    "this is missing data, not a zero total."
                )
            statement = (
                f"Grouped {matched_count} project-scoped {entity_name} by "
                + ", ".join(name.replace("_", " ") for name in plan.group_by_fields)
                + (
                    f" with total summed {plan.metric.replace('_', ' ')} "
                    f"{_display_number(total_metric_value)} {metric.unit or plan.metric}."
                    if metric and total_metric_value is not None else
                    f" with summed {plan.metric.replace('_', ' ')}."
                    if metric else "."
                )
            )
            claim = {
                "statement": statement,
                "value": matched_count,
                "unit": entity_name,
                "basis": (
                    f"Distinct {label}.{identity_property} records grouped by exact mapped fields "
                    + ", ".join(field.property for field in groups)
                    + (f" and summed {label}.{metric.property}." if metric else ".")
                ),
                "details": details,
                "total_count": matched_count,
                "displayed_count": len(rows),
            }
            if metric:
                claim["coverage"] = {
                    "candidate_count": matched_count, "evaluated_count": matched_count,
                    "matched_count": total_metric_records,
                    "missing_count": max(matched_count - total_metric_records, 0),
                    "unknown_count": 0, "excluded_count": 0, "exhaustive": True,
                }
                claim["measurement"] = {
                    "source_value": (
                        total_metric_value / metric.conversion_factor
                        if isinstance(total_metric_value, (int, float)) else None
                    ),
                    "source_unit": metric.source_unit or metric.unit,
                    "canonical_unit": metric.unit,
                    "conversion_factor": metric.conversion_factor,
                    "conversion_basis": metric.conversion_basis,
                    "source_property": f"{label}.{metric.property}",
                }
                claim["source_tags"] = [
                    *(f"{label}.{field.property}" for field in groups),
                    f"{label}.{metric.property}",
                ]
        elif plan.operation in {"group_summary", "maximum_group_sum"}:
            group = _field(entity, plan.group_by)
            metric = _field(entity, plan.metric)
            parameters["limit"] = 1 if plan.operation == "maximum_group_sum" else plan.limit
            totals = cypher.execute(
                f"{match_clause} WHERE {where} AND n.`{group.property}` IS NOT NULL "
                f"RETURN count(DISTINCT n.`{group.property}`) AS total_groups, "
                f"count(DISTINCT {identity}) AS total_records, "
                f"count(n.`{metric.property}`) AS metric_records",
                parameters,
            )
            total_groups = int(totals[0].get("total_groups") or 0) if totals else 0
            matched_count = int(totals[0].get("total_records") or 0) if totals else 0
            total_metric_records = int(totals[0].get("metric_records") or 0) if totals else 0
            rows = cypher.execute(
                f"{match_clause} WHERE {where} AND n.`{group.property}` IS NOT NULL "
                f"RETURN n.`{group.property}` AS value, count(DISTINCT {identity}) AS count, "
                f"sum(toFloat(n.`{metric.property}`) * {metric.conversion_factor!r}) AS metric_value, "
                f"count(n.`{metric.property}`) AS metric_records "
                "ORDER BY metric_value DESC, value LIMIT $limit",
                parameters,
            )

            is_function_schedule = plan.entity == "spaces" and plan.group_by == "type"
            detail_rows = []
            for row in rows:
                group_name = str(row.get("value") or "unknown").replace("_", " ").title()
                record_count = int(row.get("count") or 0)
                metric_value = _display_number(row.get("metric_value"))
                unit_text = f" {metric.unit}" if metric.unit else ""
                record_unit = (
                    "space" if is_function_schedule and record_count == 1
                    else "spaces" if is_function_schedule
                    else "record" if record_count == 1
                    else "records"
                )
                detail_rows.append(
                    f"{group_name}: {record_count} {record_unit}, covering {metric_value}{unit_text}."
                )
            level_filter = next(
                (item for item in normalized_filters if item["field"] == "level"), None
            )
            if plan.operation == "maximum_group_sum" and rows:
                winner = rows[0]
                metric_value = _display_number(winner.get("metric_value"))
                unit_text = f" {metric.unit}" if metric.unit else ""
                headline = (
                    f"The maximum summed {plan.metric.replace('_', ' ')} by "
                    f"{plan.group_by.replace('_', ' ')} is {metric_value}{unit_text} "
                    f"at {winner.get('value')}."
                )
                detail_rows = []
            elif is_function_schedule:
                location = ""
                if level_filter:
                    requested = (level_filter.get("requested_values") or ["requested level"])[0]
                    stored = ", ".join(str(value) for value in level_filter.get("values") or [])
                    location = f" on the {requested} ({stored})"
                headline = f"There are {total_groups} types of functions{location}:"
            else:
                headline = (
                    f"There are {total_groups} {plan.group_by.replace('_', ' ')} groups "
                    f"across {matched_count} project-scoped {entity_name}:"
                )
            statement = headline
            if detail_rows:
                statement += "\n- " + "\n- ".join(detail_rows)
            claim = {
                "statement": statement,
                "value": winner.get("metric_value") if plan.operation == "maximum_group_sum" and rows else total_groups,
                "unit": metric.unit if plan.operation == "maximum_group_sum" else f"{plan.group_by.replace('_', ' ')} groups",
                "basis": (
                    f"Distinct {label}.{identity_property} counts and summed "
                    f"{label}.{metric.property} values grouped by {group.property}."
                ),
                "total_count": total_groups,
                "displayed_count": len(rows),
                "coverage": {
                    "candidate_count": matched_count, "evaluated_count": matched_count,
                    "matched_count": total_metric_records,
                    "missing_count": max(matched_count - total_metric_records, 0),
                    "unknown_count": 0, "excluded_count": 0, "exhaustive": True,
                },
                "measurement": {
                    "source_value": None,
                    "source_unit": metric.source_unit or metric.unit,
                    "canonical_unit": metric.unit,
                    "conversion_factor": metric.conversion_factor,
                    "conversion_basis": metric.conversion_basis,
                    "source_property": f"{label}.{metric.property}",
                },
                "source_tags": [f"{label}.{group.property}", f"{label}.{metric.property}"],
            }
        elif plan.operation in {"group_count", "distinct"}:
            group = _field(entity, plan.group_by)
            numeric_distinct = plan.operation == "distinct" and group.data_type == "number"
            group_expression = (
                f"round(toFloat(n.`{group.property}`), 4)"
                if numeric_distinct else f"n.`{group.property}`"
            )
            candidate_rows = cypher.execute(
                f"{match_clause} WHERE {where} "
                f"RETURN count(DISTINCT {identity}) AS candidate_count",
                parameters,
            )
            candidate_count = int(candidate_rows[0].get("candidate_count") or 0) if candidate_rows else 0
            totals = cypher.execute(
                f"{match_clause} WHERE {where} AND {group_expression} IS NOT NULL "
                f"RETURN count(DISTINCT {group_expression}) AS total_groups, "
                f"count(DISTINCT {identity}) AS total_records",
                parameters,
            )
            total_groups = int(totals[0].get("total_groups") or 0) if totals else 0
            total_records = int(totals[0].get("total_records") or 0) if totals else 0
            parameters["limit"] = plan.limit
            rows = cypher.execute(
                f"{match_clause} WHERE {where} AND {group_expression} IS NOT NULL "
                f"RETURN {group_expression} AS value, count(DISTINCT {identity}) AS count "
                "ORDER BY count DESC, value LIMIT $limit",
                parameters,
            )
            matched_count = total_records
            if numeric_distinct:
                details = [
                    f"{_display_number(float(row.get('value')) * group.conversion_factor)}"
                    f"{f' {group.unit}' if group.unit else ''}: {row.get('count')} records"
                    for row in rows
                    if row.get("value") is not None
                ]
            else:
                details = [f"{row.get('value')}: {row.get('count')}" for row in rows]
            noun = "Distinct values" if plan.operation == "distinct" else "Counts"
            property_missing = candidate_count > 0 and total_records == 0
            if property_missing:
                limitations.append(
                    f"Matching {entity_name} exist, but {plan.group_by.replace('_', ' ')} is "
                    "unpopulated; this is missing data, not a zero or an empty entity population."
                )
            distinct_values_text = ", ".join(
                detail.split(":", 1)[0] for detail in details
            )
            claim = {
                "statement": (
                    f"{candidate_count} project-scoped {entity_name} exist, but "
                    f"{plan.group_by.replace('_', ' ')} is unpopulated."
                    if property_missing else
                    (
                        f"Distinct modeled {plan.group_by.replace('_', ' ')} values for "
                        f"project-scoped {entity_name}: {distinct_values_text}."
                        if numeric_distinct else
                        f"{noun} for project-scoped {entity_name} by {plan.group_by.replace('_', ' ')}: "
                        f"{total_groups} groups covering {matched_count} records."
                    )
                ),
                "value": None if property_missing else total_groups if plan.operation == "distinct" else matched_count,
                "unit": (
                    group.unit if numeric_distinct else
                    "distinct values" if plan.operation == "distinct" else entity_name
                ),
                "basis": (
                    f"Distinct populated {label}.{group.property} values, with counts of distinct "
                    f"{label}.{identity_property} records, in the authorized source scope."
                    if plan.operation == "distinct" else
                    f"Grouped distinct {label}.{identity_property} values in the authorized source scope."
                ),
                "details": details,
                "total_count": total_groups,
                "displayed_count": len(details),
                "coverage": {
                    "candidate_count": candidate_count,
                    "evaluated_count": candidate_count,
                    "matched_count": total_records,
                    "missing_count": max(candidate_count - total_records, 0),
                    "unknown_count": 0,
                    "excluded_count": 0,
                    "exhaustive": True,
                },
            }
            if numeric_distinct:
                claim["measurement"] = {
                    "source_value": None,
                    "source_unit": group.source_unit or group.unit,
                    "canonical_unit": group.unit,
                    "conversion_factor": group.conversion_factor,
                    "conversion_basis": group.conversion_basis,
                    "source_property": f"{label}.{group.property}",
                }
                claim["method"] = "distinct_numeric_property_values"
                claim["source_tags"] = [
                    f"{label}.{group.property}", f"{label}.{identity_property}",
                ]
        else:
            metric = _field(entity, plan.metric)
            function_name = {
                "sum": "sum", "average": "avg", "minimum": "min", "maximum": "max"
            }[plan.operation]
            rows = cypher.execute(
                f"{match_clause} WHERE {where} AND n.`{metric.property}` IS NOT NULL "
                f"RETURN {function_name}(toFloat(n.`{metric.property}`) * {metric.conversion_factor!r}) AS value, "
                f"count(n.`{metric.property}`) AS matched_records, "
                f"count(DISTINCT {identity}) AS evaluated_records",
                parameters,
            )
            value = rows[0].get("value") if rows else None
            matched_count = int(rows[0].get("matched_records") or 0) if rows else 0
            evaluated_count = int(rows[0].get("evaluated_records") or matched_count) if rows else 0
            display = "no modelled value" if value is None else str(round(value, 4) if isinstance(value, float) else value)
            unit_text = f" {metric.unit}" if metric.unit else ""
            claim = {
                "statement": (
                    f"The {plan.operation} {plan.metric.replace('_', ' ')} for project-scoped {entity_name} {filter_text} "
                    f"is {display}{unit_text}, based on {matched_count} populated records."
                ),
                "value": value,
                "unit": metric.unit or plan.metric,
                "basis": f"{plan.operation} of populated {label}.{metric.property} values in scope.",
                "coverage": {
                    "candidate_count": evaluated_count,
                    "evaluated_count": evaluated_count,
                    "matched_count": matched_count,
                    "missing_count": max(evaluated_count - matched_count, 0),
                    "unknown_count": 0,
                    "excluded_count": 0,
                    "exhaustive": True,
                },
                "measurement": {
                    "source_value": (value / metric.conversion_factor) if isinstance(value, (int, float)) else None,
                    "source_unit": metric.source_unit or metric.unit,
                    "canonical_unit": metric.unit,
                    "conversion_factor": metric.conversion_factor,
                    "conversion_basis": metric.conversion_basis,
                    "source_property": f"{label}.{metric.property}",
                },
                "method": plan.operation,
                "source_tags": [f"{label}.{metric.property}"],
            }
        result = {
            "rows": rows,
            "matched_count": matched_count,
            "claim": claim,
            "limitations": limitations,
            "normalized_filters": normalized_filters,
        }
    claim_payload = result.get("claim") or {}
    claim_payload.setdefault("method", plan.operation)
    claim_payload.setdefault("answer_key", plan.answer_key)
    claim_payload.setdefault("satisfies", list(plan.satisfies))
    # Package ownership is part of the replayed claim digest.  A query is usually
    # executed in an isolated branch and replayed after that branch has been merged
    # into the root context, so the ambient context workstream is not stable.  The
    # plan is the immutable execution contract and must remain authoritative here.
    claim_payload.setdefault(
        "work_package_id", plan.work_package_id or ctx.workstream_id,
    )
    claim_payload.setdefault(
        "constraint_bindings",
        [item.model_dump(mode="json") for item in plan.constraint_bindings],
    )
    plausibility_flags: list[dict[str, Any]] = list(claim_payload.get("plausibility_flags") or [])
    claim_value = claim_payload.get("value")
    claim_unit = str(claim_payload.get("unit") or "").casefold()
    if isinstance(claim_value, (int, float)) and not isinstance(claim_value, bool):
        if not math.isfinite(float(claim_value)):
            plausibility_flags.append({
                "code": "non_finite_measurement", "severity": "error",
                "message": "The computed numeric result is not finite.", "evidence_ids": [],
            })
        if ("percent" in claim_unit or "%" in claim_unit) and not 0 <= float(claim_value) <= 100:
            plausibility_flags.append({
                "code": "percentage_out_of_range", "severity": "error",
                "message": "A percentage result must be between 0 and 100.", "evidence_ids": [],
            })
        measurement = claim_payload.get("measurement") or {}
        canonical_unit = str(measurement.get("canonical_unit") or claim_unit).casefold()
        nonnegative_units = {"m", "mm", "cm", "m²", "m2", "square metres", "square meters"}
        if canonical_unit in nonnegative_units and float(claim_value) < 0:
            plausibility_flags.append({
                "code": "negative_measurement", "severity": "error",
                "message": "Lengths and areas cannot be negative.", "evidence_ids": [],
            })
    claim_payload["plausibility_flags"] = plausibility_flags
    result["claim"] = claim_payload
    stable = {
        "plan": plan.model_dump(mode="json"),
        "rows": result["rows"],
        "matched_count": result["matched_count"],
        "claim": result["claim"],
    }
    result["plan"] = stable["plan"]
    result["result_digest"] = hashlib.sha256(
        json.dumps(stable, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()
    result["graph_contract_version"] = ctx.graph_contract.version
    result["capability"] = "general_bim_query"
    result["cypher_executions"] = cypher.audit_log()
    return result


def get_bim_query_catalog(ctx: PipelineContext) -> str:
    """Return the authorized semantic entities, fields, operations, and filters available for BIM questions."""
    return _json(_catalog(ctx.context))


def get_project_knowledge_catalog(ctx: PipelineContext) -> str:
    """Return active-scope semantic mappings and calculation rules without identifiers."""
    knowledge = ctx.context.bim.ontology.bim_query_knowledge
    return _json({"knowledge": knowledge, "scope": "active client/project only"})


_CONSTRAINT_FIELD_ALIASES = {
    "floor": "level", "storey": "level", "story": "level", "level": "level",
    "function": "type", "usage": "type", "classification": "type", "category": "type",
    "family": "type", "kind": "type", "type": "type", "segment": "segment", "sector": "segment",
    "material": "material",
    "height": "height", "width": "width", "area": "area", "diameter": "diameter",
}


def _constraint_field(concept: str) -> str:
    tokens = re.findall(r"[^\W_]+", _normalize(concept), flags=re.UNICODE)
    aliases = {
        _CONSTRAINT_FIELD_ALIASES[token]
        for token in tokens if token in _CONSTRAINT_FIELD_ALIASES
    }
    # Compound concepts must prefer their semantic head over an incidental word.
    # In particular, ``gross floor area`` is an area boundary, not a floor/level
    # boundary. This ordering is independent of the word order emitted by the task
    # architect and is shared by planning and deterministic verification.
    for semantic_field in (
        "area", "height", "width", "diameter", "material", "segment", "type", "level",
    ):
        if semantic_field in aliases:
            return semantic_field
    return _normalize(concept).replace(" ", "_")


def _measurement_dimension(value: str) -> str:
    """Reduce human and schema metric names to a conservative physical dimension."""
    normalized = _normalize(value).replace("_", " ")
    dimension_terms = {
        "area": ("area", "square metre", "square meter", "sqm", "m2", "m²"),
        # Length, height, distance, elevation, width, and diameter are all the
        # same physical dimension.  Their measurement *basis* remains separately
        # governed by the field mapping and output contract.
        "length": (
            "length", "distance", "height", "elevation", "width", "diameter",
            "metre", "meter", "millimetre", "millimeter", "centimetre", "centimeter",
        ),
        "volume": ("volume", "cubic metre", "cubic meter", "m3", "m³"),
    }
    matches = [
        dimension for dimension, terms in dimension_terms.items()
        if any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized) for term in terms)
    ]
    return matches[0] if len(set(matches)) == 1 else ""


def _is_authorized_container_constraint(concept: str, requested_value: str) -> bool:
    """Recognize generic project/building nouns already enforced by source scope.

    Named sub-scopes such as a tower, wing, block, or zone remain real semantic
    constraints and must still be bound to evidence.
    """
    text = _normalize(f"{concept} {requested_value}")
    named_portions = {"tower", "wing", "block", "section", "zone", "podium"}
    if any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) for term in named_portions):
        return False
    container_terms = {
        "building", "the building", "model", "the model", "project", "the project",
        "current building", "current model", "authorized building", "authorised building",
    }
    concept_text = _normalize(concept)
    value_text = _normalize(requested_value)
    return concept_text in {
        "building membership", "model membership", "project membership", "container", "scope",
        "authorization", "authorisation", "authorized scope", "authorised scope",
        "configured scope", "project scope",
    } or value_text in container_terms


def _plan_constraint_coverage(
    context: BimRunContext, plan: BimQueryPlan,
) -> tuple[bool, list[str]]:
    """Prove every architect constraint is represented by a query boundary."""
    contract = context.task_contract
    if plan.entity == "permit_knowledge":
        # Requirements are selected by authorized project scope; model-element
        # constraints are applied to the separately queried modelled condition.
        return True, []
    if contract is None or not contract.constraints:
        return True, []
    constraints = list(contract.constraints)
    package_id = plan.work_package_id or context.workstream_id
    package = next(
        (item for item in contract.work_packages if item.package_id == package_id), None
    )
    if package is not None:
        if package.constraints is not None:
            constraints = list(package.constraints)
        elif len(contract.work_packages) > 1:
            package_text = " ".join([
                package.objective, *package.required_outputs,
            ]).casefold()
            constraints = [
                item for item in contract.constraints
                if item.concept.casefold().strip() in package_text
                or item.requested_value.casefold().strip() in package_text
            ]
    registered = context.schema_mappings.get(plan.mapping_id)
    missing: list[str] = []
    for constraint in constraints:
        if _is_authorized_container_constraint(
            constraint.concept, constraint.requested_value
        ):
            # Source authorization is injected by the compiler and verified as
            # its own semantic check; it is intentionally not a model field.
            continue
        explicit = next(
            (
                item for item in plan.constraint_bindings
                if _normalize(item.concept) == _normalize(constraint.concept)
                and _normalize(item.requested_value) == _normalize(constraint.requested_value)
                and (not item.package_id or item.package_id == package_id)
            ),
            None,
        )
        if explicit is not None:
            binding_valid = (
                not explicit.mapping_id or explicit.mapping_id == plan.mapping_id
            )
            if explicit.applied_as in {"filter", "entity_boundary"}:
                matching_filters = [
                    item for item in plan.filters if item.field == explicit.semantic_field
                ]
                supplied = {
                    _normalize(value)
                    for item in matching_filters
                    for value in _raw_filter_values(item)
                }
                expected = {_normalize(value) for value in explicit.exact_values}
                expected_operator = (
                    "in" if explicit.operator == "equals" and len(expected) > 1
                    else explicit.operator
                )
                binding_valid = (
                    binding_valid
                    and bool(expected)
                    and expected.issubset(supplied)
                    and any(item.operator == expected_operator for item in matching_filters)
                )
            elif explicit.applied_as == "grouping":
                binding_valid = binding_valid and explicit.semantic_field in {
                    plan.group_by, *plan.group_by_fields,
                }
            elif explicit.applied_as == "metric":
                binding_valid = binding_valid and explicit.semantic_field == plan.metric
            elif explicit.applied_as == "relationship":
                binding_valid = binding_valid and any(
                    item.relationship == explicit.semantic_field
                    for item in plan.relationship_filters
                )
            if binding_valid:
                continue
        target_field = _constraint_field(constraint.concept)
        requested = _normalize(constraint.requested_value)
        grouping_fields = {
            _constraint_field(value)
            for value in ([plan.group_by] if plan.group_by else []) + list(plan.group_by_fields)
        }
        if target_field == "level" and requested in {
            "per floor", "by floor", "each floor", "all floors", "בכל קומה", "לפי קומה", "בקומה",
        } and "level" in grouping_fields:
            continue
        candidates = [
            item for item in plan.filters
            if _constraint_field(item.field) == target_field
            or target_field in _normalize(item.field).replace(" ", "_")
        ]
        covered = bool(candidates)
        if covered and requested:
            raw_values = [_normalize(value) for item in candidates for value in _raw_filter_values(item)]
            value_covered = any(
                requested == value or requested in value or value in requested
                for value in raw_values if value
            )
            numeric_requested = re.findall(r"-?\d+(?:[.,]\d+)?", requested)
            numeric_filter = [
                number for value in raw_values
                for number in re.findall(r"-?\d+(?:[.,]\d+)?", value)
            ]
            value_covered = value_covered or bool(
                numeric_requested and set(numeric_requested) & set(numeric_filter)
            )
            if isinstance(registered, RegisteredSchemaMapping):
                value_covered = value_covered or any(
                    _constraint_field(binding.semantic_name) == target_field
                    and (
                        requested in _normalize(binding.user_concept)
                        or _normalize(binding.user_concept) in requested
                    )
                    for binding in registered.proposal.value_bindings
                )
            covered = value_covered
        if not covered:
            missing.append(f"{constraint.concept}={constraint.requested_value}")
    return not missing, missing


def _validate_answer_metadata(ctx: PipelineContext, plan: Any) -> None:
    required_outputs = set(
        ctx.context.task_contract.required_outputs if ctx.context.task_contract else []
    )
    role = getattr(plan, "role", "answer_producing")
    if role not in {"answer_producing", "supporting"} or not required_outputs:
        return
    if role == "answer_producing" and not str(getattr(plan, "answer_key", "")).strip():
        raise ValueError("An answer-producing plan requires a stable answer_key.")
    satisfies = set(getattr(plan, "satisfies", []) or [])
    if role == "supporting" and not satisfies:
        # A related-population probe is intentionally unable to satisfy the
        # direct output. It may omit unresolved task constraints only because
        # the evidence role prevents promotion to an answer claim.
        return
    if not satisfies:
        raise ValueError("An evidentiary plan must declare which required_outputs it satisfies.")
    unknown = sorted(satisfies - required_outputs)
    if unknown:
        raise ValueError(
            "Plan satisfies values must exactly match the task contract: " + ", ".join(unknown)
        )
    specs_by_output = dict(zip(
        ctx.context.task_contract.required_outputs,
        ctx.context.task_contract.output_specs or [],
        strict=True,
    ))
    if isinstance(plan, GeometryQueryPlan) and role == "answer_producing":
        configs = [
            item for item in ctx.context.bim.ontology.bim_query_knowledge.values()
            if isinstance(item, dict) and item.get("calculation") == plan.calculation
        ]
        if len(configs) > 1:
            raise ValueError(f"Calculation {plan.calculation!r} is ambiguously configured.")
        allowed_kinds = set(configs[0].get("output_kinds") or []) if configs else set()
        incompatible = sorted(
            spec.key for output, spec in specs_by_output.items()
            if output in satisfies and allowed_kinds and spec.kind not in allowed_kinds
        )
        if incompatible:
            raise ValueError(
                f"Calculation {plan.calculation!r} cannot satisfy typed outputs: "
                + ", ".join(incompatible)
            )
    if isinstance(plan, BimQueryPlan):
        covered, missing_constraints = _plan_constraint_coverage(ctx.context, plan)
        if not covered:
            raise ValueError(
                "The query plan omits required task constraints: "
                + ", ".join(missing_constraints)
            )
        if role == "answer_producing":
            specs = [specs_by_output[output] for output in satisfies]
            numeric_operations = {
                "sum", "average", "minimum", "maximum", "maximum_group_sum",
                "group_summary", "multi_group_summary",
            }
            for spec in specs:
                if spec.kind == "measurement" and (
                    (
                        plan.operation not in numeric_operations
                        or not plan.metric
                    )
                    and not (
                        plan.operation == "distinct"
                        and bool(plan.group_by or plan.group_by_fields)
                    )
                ):
                    raise ValueError(
                        f"Output {spec.key!r} requires a numeric aggregate/metric "
                        "or a distinct numeric value field."
                    )
                if spec.kind == "grouped_summary" and plan.operation not in {
                    "group_count", "group_summary", "multi_group_count", "multi_group_summary",
                }:
                    raise ValueError(f"Output {spec.key!r} requires a grouped operation.")
                if spec.kind == "coverage" and plan.operation != "coverage":
                    raise ValueError(f"Output {spec.key!r} requires the coverage operation.")
                if spec.kind == "relationship_coverage" and plan.operation != "relationship_coverage":
                    raise ValueError(
                        f"Output {spec.key!r} requires relationship_coverage."
                    )
                actual_dimensions = {
                    *([plan.group_by] if plan.group_by else []), *plan.group_by_fields,
                }
                registered_mapping = ctx.context.schema_mappings.get(plan.mapping_id)
                field_aliases = {
                    field.semantic_name: [
                        field.semantic_name, field.property, *field.aliases,
                    ]
                    for field in (
                        registered_mapping.proposal.fields
                        if isinstance(registered_mapping, RegisteredSchemaMapping) else []
                    )
                }
                missing_dimensions = {
                    dimension for dimension in spec.grouping_dimensions
                    if not any(
                        _normalize(dimension).replace("_", " ")
                        == _normalize(term).replace("_", " ")
                        or re.search(
                            rf"(?<!\w){re.escape(_normalize(term).replace('_', ' '))}(?!\w)",
                            _normalize(dimension).replace("_", " "),
                            flags=re.UNICODE,
                        ) is not None
                        or _constraint_field(dimension) == _constraint_field(actual)
                        for actual in actual_dimensions
                        for term in field_aliases.get(actual, [actual])
                    )
                }
                if missing_dimensions:
                    raise ValueError(
                        f"Output {spec.key!r} omits grouping dimensions: "
                        + ", ".join(sorted(missing_dimensions))
                    )
                if spec.metric and spec.kind in {"measurement", "grouped_summary"}:
                    normalized_metric = _normalize(spec.metric).replace(" ", "_")
                    entity = _entity_from_registered_mapping(ctx.context, plan)[0]
                    planned_field = _field(entity, plan.metric) if plan.metric else None
                    actual_metric_names = {
                        _normalize(plan.metric).replace(" ", "_"),
                        _normalize(planned_field.property).replace(" ", "_")
                        if planned_field else "",
                        *(
                            _normalize(alias).replace(" ", "_")
                            for alias in (
                                next(
                                    (
                                        field.aliases
                                        for field in registered_mapping.proposal.fields
                                        if field.semantic_name == plan.metric
                                    ),
                                    [],
                                )
                                if isinstance(registered_mapping, RegisteredSchemaMapping)
                                else []
                            )
                        ),
                    }
                    required_dimension = _measurement_dimension(spec.metric)
                    actual_dimension = _measurement_dimension(" ".join(actual_metric_names))
                    dimension_compatible = bool(
                        required_dimension and required_dimension == actual_dimension
                    )
                    alias_compatible = any(
                        name and re.search(
                            rf"(?<!\w){re.escape(name.replace('_', ' '))}(?!\w)",
                            normalized_metric.replace("_", " "),
                            flags=re.UNICODE,
                        ) is not None
                        for name in actual_metric_names
                    )
                    if plan.metric and (
                        normalized_metric not in actual_metric_names
                        and not dimension_compatible
                        and not alias_compatible
                    ):
                        raise ValueError(
                            f"Output {spec.key!r} requires metric {spec.metric!r}."
                        )
            output_text = " ".join(satisfies).casefold()
            measurement_requested = any(
                re.search(pattern, output_text)
                for pattern in (
                    r"\b(length|distance|height)\b",
                    r"\b(area|volume)\s+(value|size|total|sum|measurement|by|per)",
                    r"\b(total|sum|maximum|minimum|average)\s+(area|length|height|distance|volume)",
                )
            )
            if measurement_requested and (
                (
                    plan.operation not in numeric_operations or not plan.metric
                )
                and not (
                    plan.operation == "distinct"
                    and bool(plan.group_by or plan.group_by_fields)
                )
            ):
                raise ValueError(
                    "A numeric measurement output requires an aggregate operation and metric "
                    "or a distinct numeric value field."
                )


def query_project_geometry(ctx: PipelineContext, plan: GeometryQueryPlan) -> str:
    """Run one named, scoped project calculation as auditable evidence."""
    _validate_answer_metadata(ctx, plan)
    _validate_governed_calculation_schema(ctx.context, plan.calculation)
    report = calculate_project_geometry(
        plan.calculation, ctx.context.bim, ctx.context.scope.allowed_sources,
        ctx.context.bim.ontology.bim_query_knowledge,
    )
    if report is None or not report.claims:
        raise ValueError(f"Insufficient active-project geometry for {plan.calculation!r}.")
    claim = report.claims[0].model_dump(mode="json")
    stable = {"plan": plan.model_dump(mode="json"), "claim": claim,
              "calculation": plan.calculation}
    result = {
        **stable, "rows": [{"calculation": plan.calculation, "value": claim.get("value")}],
        "matched_count": 1, "limitations": report.limitations,
        "capability": "project_calculation",
        "result_digest": hashlib.sha256(
            json.dumps(stable, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest(),
    }
    evidence_id = f"query-{uuid4().hex[:10]}"
    ctx.context.add_evidence(Evidence(
        evidence_id=evidence_id, kind="query", summary=claim["statement"], payload=_json(result)
    ))
    ctx.context.add_artifact(RunArtifact(
        artifact_id=f"plan-{evidence_id}", kind="query_plan", producer="Cypher Query Handler",
        summary=claim["statement"], payload={"evidence_id": evidence_id, **result},
    ))
    return _json({"evidence_id": evidence_id, **result})


def _validate_governed_calculation_schema(context: BimRunContext, calculation: str) -> None:
    """Fail closed when a governed recipe references a stale label/property surface."""
    configs = [
        item for item in context.bim.ontology.bim_query_knowledge.values()
        if isinstance(item, dict) and item.get("calculation") == calculation
    ]
    if not configs or not configs[0].get("recipe"):
        return
    if len(configs) != 1:
        raise ValueError(f"Calculation {calculation!r} is ambiguously configured.")
    config = configs[0]
    required: dict[str, set[str]] = {}
    required_relationships: list[tuple[str, str, str, str]] = []
    if config.get("recipe") == "composite_grouped_count":
        entities = config.get("components") or []
        for item in entities:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "")
            required.setdefault(label, set()).update({
                str(item.get("identity_property") or ""),
                str(item.get("source_property") or "source"),
                str(item.get("group_property") or ""),
                *(str(value) for value in item.get("group_fallback_properties") or []),
                *(str(value.get("property") or "") for value in item.get("filters") or []),
            })
            relationship = item.get("group_relationship") or {}
            if isinstance(relationship, dict) and relationship:
                target_label = str(relationship.get("target_label") or "")
                # Composite grouping compiles this as OPTIONAL MATCH and retains a
                # governed direct/fallback-property path. An ingest that omits the
                # relationship surface must therefore yield an explicit unassigned
                # bucket, not fail the entire workstream before producing evidence.
                # A future calculation may opt into a mandatory topology surface.
                if relationship.get("required") is True:
                    required.setdefault(target_label, set()).update({
                        str(item.get("source_property") or "source"),
                        str(relationship.get("target_property") or ""),
                    })
                    required_relationships.append((
                        label,
                        str(relationship.get("relationship_type") or ""),
                        target_label,
                        str(relationship.get("direction") or "either"),
                    ))
    elif config.get("recipe") == "scope_comparison":
        item = config.get("entity") or {}
        if isinstance(item, dict):
            label = str(item.get("label") or "")
            all_filters = [
                *(item.get("filters") or []),
                *(
                    value
                    for scope in config.get("scopes") or []
                    if isinstance(scope, dict)
                    for value in scope.get("filters") or []
                ),
            ]
            required.setdefault(label, set()).update({
                str(item.get("identity_property") or ""),
                str(item.get("source_property") or "source"),
                *(str(value.get("property") or "") for value in all_filters),
            })
    elif config.get("recipe") == "property_coverage":
        item = config.get("entity") or {}
        if isinstance(item, dict):
            label = str(item.get("label") or "")
            required.setdefault(label, set()).update({
                str(item.get("identity_property") or ""),
                str(item.get("source_property") or "source"),
                str(config.get("property") or ""),
                *(str(value.get("property") or "") for value in item.get("filters") or []),
            })
    else:
        return  # The computation registry will reject unrecognized recipe kinds.
    profile = build_compact_model_profile(context)
    for label, properties in required.items():
        observed = {
            prop
            for signature in profile.node_types
            if label in signature.labels
            for prop in signature.properties
        }
        if not observed:
            raise ValueError(f"Governed calculation label {label!r} is absent from the live model.")
        missing = sorted((properties - {""}) - observed)
        if missing:
            raise ValueError(
                f"Governed calculation {calculation!r} references stale properties on "
                f"{label!r}: {missing}."
            )
    for from_label, relationship_type, to_label, direction in required_relationships:
        observed = any(
            item.type == relationship_type
            and (
                from_label in item.from_labels and to_label in item.to_labels
                if direction == "outgoing" else
                from_label in item.to_labels and to_label in item.from_labels
                if direction == "incoming" else
                (
                    from_label in item.from_labels and to_label in item.to_labels
                    or from_label in item.to_labels and to_label in item.from_labels
                )
            )
            for item in profile.relationship_types
        )
        if not observed:
            raise ValueError(
                f"Governed calculation {calculation!r} references a stale relationship "
                f"{from_label!r} {relationship_type!r} {to_label!r}."
            )


def inspect_project_graph_structure(
    ctx: PipelineContext,
    sample_limit: int = 3,
    include_properties: bool = False,
    focus: str = "",
) -> str:
    """Fetch distinct project node signatures and relationship patterns with bounded names and properties."""
    result = _project_graph_structure(
        ctx.context,
        sample_limit=sample_limit,
        include_properties=include_properties,
        focus=focus,
    )
    ctx.context.schema_discovery["project_graph_structure"] = result
    if "graph-discovery" not in ctx.context.artifacts:
        ctx.context.add_artifact(RunArtifact(
            artifact_id="graph-discovery",
            kind="graph_discovery",
            producer="Graph Inspector",
            summary=(
                f"Inspected {result['unique_node_label_count']} node labels and "
                f"{result['relationship_pattern_count']} relationship patterns."
            ),
            payload=result,
        ))
    return _json(result)


def _observed_node_type(ctx: BimRunContext, label: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", label):
        raise ValueError("Live labels must be safe Neo4j identifiers.")
    inventory = _queryable_node_inventory(ctx)
    matches = [item for item in inventory["node_types"] if label in item["labels"]]
    if not matches:
        raise ValueError(f"Label {label!r} was not observed in the authorized project.")
    # Neo4j groups nodes by their complete label signature.  A BIM label can
    # therefore occur in several inventory rows, each with different
    # properties.  Treat the requested label as the union of those signatures;
    # selecting only the largest row can incorrectly hide valid live fields.
    return {
        "labels": sorted({
            observed_label
            for item in matches
            for observed_label in item.get("labels", [])
        }),
        "node_count": sum(int(item.get("node_count") or 0) for item in matches),
        "properties": sorted({
            prop
            for item in matches
            for prop in item.get("properties", [])
        }),
        "sample_names": list(dict.fromkeys(
            sample
            for item in matches
            for sample in item.get("sample_names", [])
        ))[:3],
    }


def _queryable_node_inventory(ctx: BimRunContext) -> dict[str, Any]:
    cache_key = "queryable_node_inventory_v1"
    cached = ctx.schema_discovery.get(cache_key)
    if isinstance(cached, dict):
        return cached
    source_property = ctx.graph_contract.authorization_path.source_property
    rows = ctx.bim.query(
        f"MATCH (n) WHERE n.`{source_property}` IN $allowed_sources "
        "RETURN labels(n) AS labels, count(n) AS node_count, "
        "collect(DISTINCT keys(n)) AS property_key_groups, "
        "collect(DISTINCT coalesce(n.name, n.long_name, n.id))[0..3] AS sample_names "
        "ORDER BY node_count DESC LIMIT 40",
        {"allowed_sources": ctx.scope.allowed_sources},
    )
    node_types = []
    for row in rows:
        node_types.append({
            "labels": sorted(str(value) for value in row.get("labels") or []),
            "node_count": int(row.get("node_count") or 0),
            "properties": sorted({
                str(key)
                for group in row.get("property_key_groups") or []
                for key in (group or [])
            }),
            "sample_names": [str(value) for value in row.get("sample_names") or []],
        })
    result = {
        "project_id": ctx.scope.project_id,
        "trusted_scope_property": source_property,
        "node_types": node_types,
        "scope": "authorized project only",
    }
    ctx.schema_discovery[cache_key] = result
    return result


def inspect_queryable_node_types(ctx: PipelineContext) -> str:
    """Return a compact inventory while retaining the complete property surface internally."""
    inventory = _queryable_node_inventory(ctx.context)
    priority_names = (
        "GlobalID", "id", "object_id", "name", "long_name", "Category", "Family",
        "Family and Type", "Type", "Type Id", "canonical_type", "object_type",
        "Level", "Schedule Level", "Service Type", "Length", "Area", "source",
    )
    priority = {name.casefold(): index for index, name in enumerate(priority_names)}
    compact_types = []
    for item in inventory["node_types"]:
        properties = item.get("properties") or []
        suggested = sorted(
            (
                name for name in properties
                if name.casefold() in priority
                or re.search(
                    r"(^|[ _-])(category|family|type|name|globalid|level|service|length|area)($|[ _-])",
                    name,
                    re.I,
                )
            ),
            key=lambda name: (priority.get(name.casefold(), len(priority)), len(name), name),
        )[:24]
        compact_types.append({
            "labels": item["labels"],
            "node_count": item["node_count"],
            "property_count": len(properties),
            "suggested_properties": suggested,
            "sample_names": item["sample_names"],
        })
    return _json({
        "trusted_scope_property": inventory["trusted_scope_property"],
        "node_types": compact_types,
        "scope": inventory["scope"],
        "usage_note": (
            "Only high-value property names are shown. Use search_properties for the complete "
            "observed property surface of a selected label."
        ),
    })


def _lexical_semantic_score(semantic_text: str, target: str) -> float:
    def tokens(value: str) -> set[str]:
        value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
        normalized_tokens = set()
        for token in re.findall(r"\w+", _normalize(value).replace("_", " ")):
            if len(token) > 4 and token.endswith("ies"):
                token = token[:-3] + "y"
            elif len(token) > 4 and token.endswith("es"):
                token = token[:-2]
            elif len(token) > 3 and token.endswith("s"):
                token = token[:-1]
            normalized_tokens.add(token)
        return normalized_tokens

    phrases = []
    for item in re.split(r"[|;,]", semantic_text):
        item = re.sub(r"^(?:user concept|bim vocabulary)\s*:\s*", "", item, flags=re.I)
        normalized = _normalize(item).replace("_", " ")
        if len(normalized) >= 2:
            phrases.append(normalized)
    normalized_target = _normalize(target).replace("_", " ")
    target_tokens = tokens(target)
    if any(
        re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized_target) is not None
        for phrase in phrases
    ):
        return 1.0
    return round(max(
        (
            len(tokens(phrase) & target_tokens) / len(tokens(phrase))
            for phrase in phrases if tokens(phrase)
        ),
        default=0.0,
    ), 6)


def profile_project_classification_hierarchy(
    ctx: PipelineContext,
    label: str,
    properties: list[str],
    concept: str = "",
    limit: int = 30,
) -> str:
    """Profile exact scoped category/family/type branches before choosing classification filters."""
    node_type = _observed_node_type(ctx.context, label)
    requested = list(dict.fromkeys(properties))
    if not requested or len(requested) > 3:
        raise ValueError("Profile between one and three hierarchy properties.")
    unknown = set(requested) - set(node_type["properties"])
    if unknown:
        raise ValueError(f"Properties were not observed on {label!r}: {sorted(unknown)}")
    if any("`" in prop or any(ord(ch) < 32 for ch in prop) for prop in requested):
        raise ValueError("Live property names contain unsafe characters.")
    limit = max(1, min(limit, 50))
    cache_key = "classification_hierarchy:" + hashlib.sha256(
        _json([label, requested, concept, limit]).encode("utf-8")
    ).hexdigest()
    cached = ctx.context.schema_discovery.get(cache_key)
    if isinstance(cached, dict):
        return _json(cached)
    source_property = ctx.context.graph_contract.authorization_path.source_property
    projections = ", ".join(
        f"n.`{property_name}` AS value_{index}"
        for index, property_name in enumerate(requested)
    )
    rows = ctx.context.bim.query(
        f"MATCH (n:`{label}`) WHERE n.`{source_property}` IN $allowed_sources "
        f"RETURN {projections}, count(DISTINCT n) AS record_count "
        "ORDER BY record_count DESC LIMIT $candidate_limit",
        {"allowed_sources": ctx.context.scope.allowed_sources, "candidate_limit": 300},
    )
    expanded = _expanded_semantic_text(ctx.context, concept) if concept.strip() else ""
    branches = []
    for row in rows:
        path = {
            property_name: _clean_value(row.get(f"value_{index}"))
            for index, property_name in enumerate(requested)
        }
        component_scores = [
            _lexical_semantic_score(expanded, str(value)) if expanded and value not in (None, "") else 0.0
            for value in path.values()
        ]
        weighted_scores = [
            score * ((index + 1) / len(requested))
            for index, score in enumerate(component_scores)
        ]
        hierarchy_score = round(max(weighted_scores, default=0.0), 6)
        matching_indices = [index for index, score in enumerate(component_scores) if score > 0]
        branches.append({
            "path": path,
            "record_count": int(row.get("record_count") or 0),
            "lexical_similarity": hierarchy_score,
            "component_similarities": dict(zip(requested, component_scores)),
            "deepest_matching_property": (
                requested[max(matching_indices)] if matching_indices else None
            ),
            "broad_only_match": bool(
                matching_indices and max(matching_indices) == 0 and len(requested) > 1
            ),
        })
    if expanded:
        branches.sort(key=lambda item: (-item["lexical_similarity"], -item["record_count"], _json(item["path"])))
    else:
        branches.sort(key=lambda item: (-item["record_count"], _json(item["path"])))
    result = {
        "label": label,
        "properties": requested,
        "concept": concept or None,
        "expanded_concept": expanded or None,
        "candidate_branch_count": len(rows),
        "branches": branches[:limit],
        "scope": "authorized project only",
        "usage_note": (
            "These are exact observed diagnostic branches. When a broad category contains unrelated "
            "families, bind the relevant leaf values through search_property_values, register the "
            "mapping, and execute a fresh answer-producing query."
        ),
    }
    ctx.context.schema_discovery[cache_key] = result
    return _json(result)


def profile_project_properties(
    ctx: PipelineContext,
    label: str,
    properties: list[str],
    sample_limit: int = 5,
) -> str:
    """Profile bounded values and cardinality for observed properties on one authorized live label."""
    node_type = _observed_node_type(ctx.context, label)
    observed = set(node_type["properties"])
    requested = list(dict.fromkeys(properties))
    if not requested or len(requested) > 8:
        raise ValueError("Profile between one and eight properties per call.")
    unknown = set(requested) - observed
    if unknown:
        raise ValueError(f"Properties were not observed on {label!r}: {sorted(unknown)}")
    if any("`" in prop or any(ord(ch) < 32 for ch in prop) for prop in requested):
        raise ValueError("Live property names contain unsafe characters.")
    sample_limit = max(1, min(sample_limit, 10))
    cache_key = "property_profile:" + hashlib.sha256(
        _json([label, requested, sample_limit]).encode("utf-8")
    ).hexdigest()
    cached = ctx.context.schema_discovery.get(cache_key)
    if isinstance(cached, dict):
        return _json(cached)
    source_property = ctx.context.graph_contract.authorization_path.source_property
    profiles = []
    for property_name in requested:
        rows = ctx.context.bim.query(
            f"MATCH (n:`{label}`) WHERE n.`{source_property}` IN $allowed_sources "
            f"RETURN count(n) AS node_count, count(n.`{property_name}`) AS populated_count, "
            f"count(DISTINCT n.`{property_name}`) AS distinct_count, "
            f"collect(DISTINCT n.`{property_name}`)[0..$sample_limit] AS sample_values",
            {"allowed_sources": ctx.context.scope.allowed_sources, "sample_limit": sample_limit},
        )
        row = rows[0] if rows else {}
        profiles.append({
            "property": property_name,
            "node_count": int(row.get("node_count") or 0),
            "populated_count": int(row.get("populated_count") or 0),
            "distinct_count": int(row.get("distinct_count") or 0),
            "sample_values": [_clean_value(value) for value in row.get("sample_values") or []],
        })
    result = {"label": label, "profiles": profiles, "scope": "authorized project only"}
    ctx.context.schema_discovery[cache_key] = result
    return _json(result)


def semantic_search_project_nodes(
    ctx: PipelineContext,
    query: str,
    label: str,
    name_properties: list[str],
    top_k: int = 5,
    candidate_limit: int = 500,
) -> str:
    """Rank authorized live nodes using ontology-expanded local semantic similarity."""
    if not query.strip():
        raise ValueError("Semantic node search requires a non-empty concept.")
    node_type = _observed_node_type(ctx.context, label)
    observed = set(node_type["properties"])
    properties = list(dict.fromkeys(name_properties))
    if not properties or len(properties) > 6:
        raise ValueError("Choose between one and six observed name properties.")
    unknown = set(properties) - observed
    if unknown:
        raise ValueError(f"Name properties were not observed on {label!r}: {sorted(unknown)}")
    if any("`" in prop or any(ord(ch) < 32 for ch in prop) for prop in properties):
        raise ValueError("Live property names contain unsafe characters.")
    top_k = max(1, min(top_k, 20))
    candidate_limit = max(top_k, min(candidate_limit, 1000))
    source_property = ctx.context.graph_contract.authorization_path.source_property
    model = "local-feature-hashing-v1"
    cache_key = "semantic_nodes:" + hashlib.sha256(
        _json([query, label, properties, top_k, candidate_limit, model]).encode("utf-8")
    ).hexdigest()
    cached = ctx.context.schema_discovery.get(cache_key)
    if isinstance(cached, dict):
        return _json(cached)
    expanded_query = _expanded_semantic_text(ctx.context, query)
    rows = ctx.context.bim.query(
        f"MATCH (n:`{label}`) WHERE n.`{source_property}` IN $allowed_sources "
        "WITH labels(n) AS labels, [property IN $name_properties | n[property]] AS name_values, "
        "min(elementId(n)) AS node_ref, count(n) AS population "
        "RETURN node_ref, labels, name_values, population "
        "ORDER BY population DESC, toString(name_values) LIMIT $candidate_limit",
        {
            "allowed_sources": ctx.context.scope.allowed_sources,
            "name_properties": properties,
            "candidate_limit": candidate_limit,
        },
    )
    candidates = []
    texts = []
    for row in rows:
        labels = [str(value) for value in row.get("labels") or []]
        values = [str(value)[:200] for value in row.get("name_values") or [] if value not in (None, "")]
        text_value = "type: " + ", ".join(labels) + "; name: " + " | ".join(values)
        candidate = {
            "node_ref": str(row.get("node_ref") or ""),
            "labels": labels,
            "name_values": values,
            "embedding_text": text_value,
            "population": int(row.get("population") or 0),
        }
        candidates.append(candidate)
        texts.append(text_value)
    if not candidates:
        return _json({"query": query, "matches": [], "scope": "authorized project only"})
    vectors = _local_semantic_embeddings([expanded_query, *texts])
    ranked = _rank_hybrid_candidates(
        expanded_query, vectors[0], vectors[1:], candidates
    )[:top_k]
    result = {
        "query": query,
        "expanded_query": expanded_query,
        "label": label,
        "name_properties": properties,
        "embedding_model": model,
        "candidate_count": len(candidates),
        "matches": ranked,
        "scope": "authorized project only",
    }
    ctx.context.schema_discovery[cache_key] = result
    return _json(result)


def _project_property_candidates(ctx: BimRunContext, label: str) -> list[dict[str, Any]]:
    cache_key = f"property_candidates:{label}"
    cached = ctx.schema_discovery.get(cache_key)
    if isinstance(cached, list):
        return cached
    node_type = _observed_node_type(ctx, label)
    properties = sorted(set(node_type["properties"]))
    if not properties:
        return []
    source_property = ctx.graph_contract.authorization_path.source_property
    rows = ctx.bim.query(
        f"MATCH (n:`{label}`) WHERE n.`{source_property}` IN $allowed_sources "
        "UNWIND $properties AS property "
        "RETURN property, count(n) AS node_count, count(n[property]) AS populated_count, "
        "count(DISTINCT n[property]) AS distinct_count, "
        "collect(DISTINCT n[property])[0..$sample_limit] AS sample_values "
        "ORDER BY property",
        {
            "allowed_sources": ctx.scope.allowed_sources,
            "properties": properties,
            "sample_limit": 6,
        },
    )
    candidates = []
    for row in rows:
        property_name = str(row.get("property") or "")
        sample_values = [
            str(value)[:160] for value in row.get("sample_values") or [] if value not in (None, "")
        ]
        node_count = int(row.get("node_count") or 0)
        populated_count = int(row.get("populated_count") or 0)
        distinct_count = int(row.get("distinct_count") or 0)
        embedding_text = (
            f"property: {property_name}; samples: {' | '.join(sample_values)}; "
            f"populated: {populated_count}/{node_count}; distinct: {distinct_count}"
        )
        candidates.append({
            "node_ref": property_name,
            "property": property_name,
            "node_count": node_count,
            "populated_count": populated_count,
            "distinct_count": distinct_count,
            "sample_values": sample_values,
            "embedding_text": embedding_text,
        })
    ctx.schema_discovery[cache_key] = candidates
    return candidates


def semantic_search_project_properties(
    ctx: PipelineContext,
    label: str,
    concepts: list[str],
    top_k: int = 8,
) -> str:
    """Rank observed live properties for semantic roles using names, values, and cardinality context."""
    requested_concepts = [concept.strip() for concept in concepts if concept.strip()]
    if not requested_concepts or len(requested_concepts) > 6:
        raise ValueError("Search between one and six semantic property concepts.")
    top_k = max(1, min(top_k, 15))
    candidates = _project_property_candidates(ctx.context, label)
    if not candidates:
        return _json({"label": label, "matches_by_concept": {}, "scope": "authorized project only"})
    model = "local-feature-hashing-v1"
    similarity_threshold = float(os.getenv("BIM_SCHEMA_SIMILARITY_THRESHOLD", "0.35"))
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("BIM_SCHEMA_SIMILARITY_THRESHOLD must be between -1 and 1.")
    cache_key = "semantic_properties:" + hashlib.sha256(
        _json([label, requested_concepts, top_k, model, similarity_threshold]).encode("utf-8")
    ).hexdigest()
    cached = ctx.context.schema_discovery.get(cache_key)
    if isinstance(cached, dict):
        return _json(cached)
    semantic_queries = [_expanded_semantic_text(ctx.context, concept) for concept in requested_concepts]
    texts = [candidate["embedding_text"] for candidate in candidates]
    vectors = _local_semantic_embeddings([*semantic_queries, *texts])
    candidate_vectors = vectors[len(requested_concepts):]
    matches = {}
    for concept, vector in zip(requested_concepts, vectors[:len(requested_concepts)]):
        ranked = _rank_hybrid_candidates(
            semantic_queries[len(matches)], vector, candidate_vectors, candidates
        )[:top_k]
        matches[concept] = [
            {**item, "meets_threshold": item["similarity"] >= similarity_threshold}
            for item in ranked
        ]
    result = {
        "label": label,
        "embedding_model": model,
        "property_count": len(candidates),
        "expanded_concepts": dict(zip(requested_concepts, semantic_queries)),
        "similarity_threshold": similarity_threshold,
        "matches_by_concept": matches,
        "scope": "authorized project only",
    }
    ctx.context.schema_discovery[cache_key] = result
    return _json(result)


def semantic_search_property_values(
    ctx: PipelineContext,
    label: str,
    property: str,
    concepts: list[str],
    top_k: int = 12,
    value_limit: int = 300,
) -> str:
    """Rank exact stored values of an observed property against user concepts in project scope."""
    node_type = _observed_node_type(ctx.context, label)
    if property not in set(node_type["properties"]):
        raise ValueError(f"Property {property!r} was not observed on {label!r}.")
    if "`" in property or any(ord(ch) < 32 for ch in property):
        raise ValueError("Live property names contain unsafe characters.")
    requested_concepts = [concept.strip() for concept in concepts if concept.strip()]
    if not requested_concepts or len(requested_concepts) > 6:
        raise ValueError("Search between one and six value concepts.")
    top_k = max(1, min(top_k, 20))
    value_limit = max(top_k, min(value_limit, 500))
    model = "local-feature-hashing-v1"
    similarity_threshold = float(os.getenv("BIM_SCHEMA_SIMILARITY_THRESHOLD", "0.35"))
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("BIM_SCHEMA_SIMILARITY_THRESHOLD must be between -1 and 1.")
    cache_key = "semantic_values:" + hashlib.sha256(
        _json([label, property, requested_concepts, top_k, value_limit, model, similarity_threshold]).encode("utf-8")
    ).hexdigest()
    cached = ctx.context.schema_discovery.get(cache_key)
    if isinstance(cached, dict):
        return _json(cached)
    source_property = ctx.context.graph_contract.authorization_path.source_property
    rows = ctx.context.bim.query(
        f"MATCH (n:`{label}`) WHERE n.`{source_property}` IN $allowed_sources "
        f"AND n.`{property}` IS NOT NULL "
        f"RETURN DISTINCT n.`{property}` AS value ORDER BY toString(value) LIMIT $value_limit",
        {"allowed_sources": ctx.context.scope.allowed_sources, "value_limit": value_limit},
    )
    candidates = []
    for row in rows:
        value = str(row.get("value"))[:300]
        candidates.append({
            "node_ref": value,
            "value": value,
            "embedding_text": f"stored value: {value}",
        })
    if not candidates:
        result = {
            "label": label,
            "property": property,
            "similarity_threshold": similarity_threshold,
            "matches_by_concept": {},
            "scope": "authorized project only",
        }
        ctx.context.schema_discovery[cache_key] = result
        return _json(result)
    semantic_queries = [_expanded_semantic_text(ctx.context, concept) for concept in requested_concepts]
    vectors = _local_semantic_embeddings(
        [*semantic_queries, *[item["embedding_text"] for item in candidates]]
    )
    candidate_vectors = vectors[len(requested_concepts):]
    matches = {}
    for concept, vector in zip(requested_concepts, vectors[:len(requested_concepts)]):
        ranked = _rank_hybrid_candidates(
            semantic_queries[len(matches)], vector, candidate_vectors, candidates
        )[:top_k]
        matches[concept] = [
            {**item, "meets_threshold": item["similarity"] >= similarity_threshold}
            for item in ranked
        ]
    result = {
        "label": label,
        "property": property,
        "embedding_model": model,
        "expanded_concepts": dict(zip(requested_concepts, semantic_queries)),
        "similarity_threshold": similarity_threshold,
        "candidate_count": len(candidates),
        "matches_by_concept": matches,
        "scope": "authorized project only",
    }
    ctx.context.schema_discovery[cache_key] = result
    return _json(result)


def _validate_hierarchy_binding_coverage(
    context: BimRunContext, proposal: SchemaMappingProposal,
) -> None:
    """Prevent a secondary classification field from silently dropping BIM branches."""
    fields_by_name = {field.semantic_name: field for field in proposal.fields}
    classification_bindings = [
        binding for binding in proposal.value_bindings
        if (
            (fields_by_name.get(binding.semantic_name) is not None)
            and fields_by_name[binding.semantic_name].ontology_kind
            in {"canonical_type", "ifc_class"}
        )
    ]
    hierarchy_fields = {
        field.semantic_name: field.property
        for field in proposal.fields
        if field.semantic_name.casefold() in {"category", "family", "type"}
    }
    if not classification_bindings or len(hierarchy_fields) < 2:
        return
    hierarchy_properties = set(hierarchy_fields.values())
    profiles = [
        value for key, value in context.schema_discovery.items()
        if key.startswith("classification_hierarchy:")
        and isinstance(value, dict)
        and value.get("label") == proposal.label
        and hierarchy_properties.issubset(set(value.get("properties") or []))
    ]
    if not profiles:
        raise ValueError(
            "Profile the observed category/family/type hierarchy before registering "
            "this entity classification."
        )
    for binding in classification_bindings:
        relevant_profile = next(
            (
                profile for profile in profiles
                if _lexical_semantic_score(
                    str(profile.get("expanded_concept") or profile.get("concept") or ""),
                    binding.user_concept,
                ) > 0
                if any(
                    float(branch.get("lexical_similarity") or 0.0) > 0
                    and not branch.get("broad_only_match", False)
                    for branch in profile.get("branches") or []
                )
            ),
            None,
        )
        if relevant_profile is not None and binding.property not in hierarchy_properties:
            raise ValueError(
                f"Classification binding {binding.property!r} is outside the profiled "
                "category/family/type hierarchy and can silently omit supported entity branches. "
                "Bind an exact profiled hierarchy value instead."
            )
        if relevant_profile is None:
            continue
        selected_values = {_normalize(match.value) for match in binding.matches}
        branches = relevant_profile.get("branches") or []
        relevant_branches = [
            branch for branch in branches
            if float(branch.get("lexical_similarity") or 0.0) > 0
            and not branch.get("broad_only_match", False)
        ]
        selected_branches = [
            branch for branch in relevant_branches
            if _normalize((branch.get("path") or {}).get(binding.property)) in selected_values
        ]
        if not selected_branches:
            raise ValueError(
                "The selected exact classification values do not cover any supported "
                "profiled hierarchy branch."
            )
        first_property = (relevant_profile.get("properties") or [None])[0]
        if binding.property == first_property:
            polluted = [
                branch for branch in branches
                if _normalize((branch.get("path") or {}).get(binding.property)) in selected_values
                and branch.get("broad_only_match", False)
            ]
            if polluted:
                raise ValueError(
                    "The selected category also contains unrelated family/type branches; "
                    "bind the complete supported leaf values instead."
                )
        else:
            selected_parents = {
                _normalize((branch.get("path") or {}).get(first_property))
                for branch in selected_branches
            }
            missing_siblings = [
                branch for branch in relevant_branches
                if _normalize((branch.get("path") or {}).get(first_property)) in selected_parents
                and _normalize((branch.get("path") or {}).get(binding.property)) not in selected_values
            ]
            if missing_siblings:
                raise ValueError(
                    "The selected leaf classification values omit supported sibling branches "
                    "inside the same BIM category."
                )


def register_schema_mapping(
    ctx: PipelineContext, proposal: SchemaMappingProposal
) -> str:
    """Validate and register a semantic mapping observed in the authorized project."""
    structure = ctx.context.schema_discovery.get("project_graph_structure")
    if not isinstance(structure, dict):
        raise ValueError(
            "Inspect project graph structure, including nodes and relationships, before registering a mapping."
        )
    node_type = _observed_node_type(ctx.context, proposal.label)
    if proposal.relationship_path:
        patterns = structure.get("relationship_types") or []
        previous_label = proposal.relationship_path[0].from_label
        for step in proposal.relationship_path:
            if any(
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier) is None
                for identifier in (step.from_label, step.relationship_type, step.to_label)
            ):
                raise ValueError("Relationship paths contain an unsafe label or relationship type.")
            if step.from_label != previous_label:
                raise ValueError("The proposed relationship path must be contiguous.")
            if step.direction == "outgoing":
                observed = any(
                    step.relationship_type == item.get("type")
                    and step.from_label in (item.get("from_labels") or [])
                    and step.to_label in (item.get("to_labels") or [])
                    for item in patterns
                )
            else:
                observed = any(
                    step.relationship_type == item.get("type")
                    and step.from_label in (item.get("to_labels") or [])
                    and step.to_label in (item.get("from_labels") or [])
                    for item in patterns
                )
            if not observed:
                raise ValueError(
                    f"Relationship {step.from_label!r} {step.relationship_type!r} "
                    f"{step.to_label!r} with direction {step.direction!r} was not observed."
                )
            previous_label = step.to_label
        if previous_label != proposal.label:
            raise ValueError("The relationship path must end at the mapped entity label.")
    relationship_names = [item.semantic_name for item in proposal.relationship_bindings]
    if len(relationship_names) != len(set(relationship_names)):
        raise ValueError("Relationship binding names must be unique.")
    patterns = structure.get("relationship_types") or []
    for binding in proposal.relationship_bindings:
        previous_label = proposal.label
        for step in binding.steps:
            if any(
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier) is None
                for identifier in (step.from_label, step.relationship_type, step.to_label)
            ):
                raise ValueError("Relationship bindings contain an unsafe identifier.")
            if step.from_label != previous_label:
                raise ValueError("A relationship binding must be rooted at the mapped entity and contiguous.")
            observed = any(
                step.relationship_type == item.get("type")
                and (
                    step.from_label in (item.get("from_labels") or [])
                    and step.to_label in (item.get("to_labels") or [])
                    if step.direction == "outgoing" else
                    step.from_label in (item.get("to_labels") or [])
                    and step.to_label in (item.get("from_labels") or [])
                )
                for item in patterns
            )
            if not observed:
                raise ValueError(
                    f"Relationship binding {binding.semantic_name!r} contains an unobserved step."
                )
            previous_label = step.to_label
        if binding.target_identity_property:
            target_type = _observed_node_type(ctx.context, binding.steps[-1].to_label)
            if binding.target_identity_property not in set(target_type["properties"]):
                raise ValueError(
                    f"Relationship binding {binding.semantic_name!r} target identity property "
                    f"{binding.target_identity_property!r} was not observed on "
                    f"{binding.steps[-1].to_label!r}."
                )
    observed = set(node_type["properties"])
    required = {
        proposal.identity_property,
        proposal.source_property,
        *(field.property for field in proposal.fields),
    }
    for optional in (proposal.classification_source_property, proposal.classification_name_property):
        if optional:
            required.add(optional)
    unknown = required - observed
    if unknown:
        raise ValueError(f"The proposed properties were not observed: {sorted(unknown)}")
    if any("`" in prop or any(ord(ch) < 32 for ch in prop) for prop in required):
        raise ValueError("The proposed properties contain unsafe characters.")
    trusted_source = ctx.context.graph_contract.authorization_path.source_property
    if proposal.source_property != trusted_source:
        raise ValueError("The mapping must use the trusted authorization source property.")
    semantic_names = [field.semantic_name for field in proposal.fields]
    if len(semantic_names) != len(set(semantic_names)):
        raise ValueError("Mapped semantic field names must be unique.")
    fields_by_name = {field.semantic_name: field for field in proposal.fields}
    for binding in proposal.value_bindings:
        field = fields_by_name.get(binding.semantic_name)
        if field is None or field.property != binding.property:
            raise ValueError(
                f"Value binding {binding.semantic_name!r} must reference its mapped semantic field."
            )
    _validate_hierarchy_binding_coverage(ctx.context, proposal)
    required_evidence = {
        "identity": proposal.identity_property,
        **{
            binding.semantic_name: binding.property
            for binding in proposal.value_bindings
        },
    }
    supplied_evidence = {item.semantic_name: item for item in proposal.match_evidence}
    missing_evidence = set(required_evidence) - set(supplied_evidence)
    if missing_evidence:
        raise ValueError(f"Semantic similarity evidence is required for: {sorted(missing_evidence)}")
    semantic_results = [
        value for key, value in ctx.context.schema_discovery.items()
        if key.startswith("semantic_properties:") and isinstance(value, dict)
    ]
    for semantic_name, property_name in required_evidence.items():
        evidence = supplied_evidence[semantic_name]
        if evidence.property != property_name:
            raise ValueError(f"Similarity evidence for {semantic_name!r} names a different property.")
        verified_match = None
        verified_threshold = None
        for result in semantic_results:
            concept_matches = (result.get("matches_by_concept") or {}).get(evidence.concept) or []
            match = next((item for item in concept_matches if item.get("property") == property_name), None)
            if match is not None:
                verified_match = match
                verified_threshold = float(result.get("similarity_threshold", 0.35))
                break
        if verified_match is None:
            raise ValueError(f"No live semantic-search evidence supports {semantic_name!r}.")
        verified_similarity = float(verified_match.get("similarity") or 0.0)
        if abs(verified_similarity - evidence.similarity) > 1e-6:
            raise ValueError(f"Reported similarity for {semantic_name!r} does not match live evidence.")
        if verified_similarity < verified_threshold:
            raise ValueError(
                f"Semantic match for {semantic_name!r} scored {verified_similarity:.3f}, below "
                f"the required threshold {verified_threshold:.3f}."
            )
    semantic_value_results = [
        value for key, value in ctx.context.schema_discovery.items()
        if key.startswith("semantic_values:") and isinstance(value, dict)
    ]
    for binding in proposal.value_bindings:
        verified_result = next(
            (
                result for result in semantic_value_results
                if result.get("label") == proposal.label
                and result.get("property") == binding.property
                and binding.user_concept in (result.get("matches_by_concept") or {})
            ),
            None,
        )
        if verified_result is None:
            raise ValueError(f"No live value-search evidence supports {binding.user_concept!r}.")
        available_matches = {
            str(item.get("value")): item
            for item in verified_result["matches_by_concept"][binding.user_concept]
        }
        threshold = float(verified_result.get("similarity_threshold", 0.35))
        for selected in binding.matches:
            match = available_matches.get(selected.value)
            if match is None:
                raise ValueError(f"Stored value {selected.value!r} was not returned by semantic search.")
            similarity = float(match.get("similarity") or 0.0)
            if abs(similarity - selected.similarity) > 1e-6:
                raise ValueError(f"Reported value similarity for {selected.value!r} was altered.")
            if similarity < threshold:
                raise ValueError(
                    f"Stored value {selected.value!r} scored {similarity:.3f}, below the "
                    f"required threshold {threshold:.3f}."
                )
    digest_input = {
        "project_id": ctx.context.scope.project_id,
        "proposal": proposal.model_dump(mode="json"),
    }
    mapping_id = "mapping-" + hashlib.sha256(
        _json(digest_input).encode("utf-8")
    ).hexdigest()[:12]
    existing = ctx.context.schema_mappings.get(mapping_id)
    if isinstance(existing, RegisteredSchemaMapping):
        return existing.model_dump_json()
    pending_mapping = RegisteredSchemaMapping(
        mapping_id="pending",
        proposal=proposal,
        node_count=1,
        populated_identity_count=1,
        distinct_identity_count=1,
    )
    match_clause = _registered_match_clause(pending_mapping, proposal.label)
    rows = ctx.context.bim.query(
        f"{match_clause} WHERE n.`{proposal.source_property}` IN $allowed_sources "
        f"RETURN count(DISTINCT n) AS node_count, "
        f"count(DISTINCT CASE WHEN n.`{proposal.identity_property}` IS NOT NULL THEN n END) AS populated_count, "
        f"count(DISTINCT n.`{proposal.identity_property}`) AS distinct_count",
        {"allowed_sources": ctx.context.scope.allowed_sources},
    )
    row = rows[0] if rows else {}
    node_count = int(row.get("node_count") or 0)
    populated = int(row.get("populated_count") or 0)
    distinct = int(row.get("distinct_count") or 0)
    if not node_count:
        raise ValueError("The proposed entity has no authorized project records.")
    if populated != node_count or distinct != node_count:
        raise ValueError("The proposed identity property must be populated and unique for every record.")
    for field in proposal.fields:
        if field.data_type != "number":
            continue
        numeric_rows = ctx.context.bim.query(
            f"{match_clause} WHERE n.`{proposal.source_property}` IN $allowed_sources "
            f"RETURN count(DISTINCT CASE WHEN n.`{field.property}` IS NOT NULL THEN n END) AS populated_count, "
            f"count(DISTINCT CASE WHEN toFloat(n.`{field.property}`) IS NOT NULL THEN n END) AS numeric_count",
            {"allowed_sources": ctx.context.scope.allowed_sources},
        )
        numeric_row = numeric_rows[0] if numeric_rows else {}
        if int(numeric_row.get("populated_count") or 0) != int(numeric_row.get("numeric_count") or 0):
            raise ValueError(f"Mapped numeric field {field.semantic_name!r} contains nonnumeric values.")
    registered = RegisteredSchemaMapping(
        mapping_id=mapping_id,
        proposal=proposal,
        node_count=node_count,
        populated_identity_count=populated,
        distinct_identity_count=distinct,
    )
    ctx.context.schema_mappings[mapping_id] = registered
    ctx.context.add_artifact(RunArtifact(
        artifact_id=mapping_id,
        kind="mapping",
        producer="Schema Mapper",
        summary=f"Mapped {proposal.entity_name} to live label {proposal.label}.",
        payload=registered.model_dump(mode="json"),
    ))
    return registered.model_dump_json()


def activate_project_knowledge_mapping(ctx: PipelineContext, knowledge_key: str) -> str:
    """Live-validate and activate one server-governed project semantic mapping.

    Unlike open-ended schema discovery, the caller selects only a key. Labels,
    properties, exact values, and counting semantics come from project-scoped
    trusted configuration and are still checked against the authorized graph.
    """
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", knowledge_key):
        raise ValueError("Project mapping keys must be safe identifiers.")
    knowledge = ctx.context.bim.ontology.bim_query_knowledge.get(knowledge_key)
    if not isinstance(knowledge, dict):
        raise ValueError(f"No governed project mapping named {knowledge_key!r} exists.")
    required_keys = {
        "entity_concept", "label", "source_property", "identity_property",
        "classification_property", "exact_family_values", "counting_unit",
        "counting_unit_semantics",
    }
    missing = sorted(required_keys - set(knowledge))
    if missing:
        raise ValueError("The governed project mapping is incomplete: " + ", ".join(missing))
    label = str(knowledge["label"])
    source_property = str(knowledge["source_property"])
    identity_property = str(knowledge["identity_property"])
    classification_property = str(knowledge["classification_property"])
    type_property = str(knowledge.get("type_property") or "")
    category_property = str(knowledge.get("category_property") or "")
    category_value = str(knowledge.get("category_value") or "")
    configured_fields = knowledge.get("fields") or []
    if not isinstance(configured_fields, list) or any(
        not isinstance(item, dict) for item in configured_fields
    ):
        raise ValueError("Governed mapping fields must be a list of field definitions.")
    configured_relationships = knowledge.get("relationship_bindings") or []
    if not isinstance(configured_relationships, list) or any(
        not isinstance(item, dict) for item in configured_relationships
    ):
        raise ValueError("Governed relationship bindings must be a list of definitions.")
    exact_values = [str(value) for value in knowledge["exact_family_values"] if str(value)]
    identifiers = [label, source_property, identity_property, classification_property]
    identifiers.extend(value for value in (type_property, category_property) if value)
    identifiers.extend(str(item.get("property") or "") for item in configured_fields)
    identifiers.extend(
        str(value or "")
        for binding in configured_relationships
        for step in (binding.get("steps") or [])
        if isinstance(step, dict)
        for value in (
            step.get("from_label"), step.get("relationship_type"), step.get("to_label"),
        )
    )
    if any("`" in value or any(ord(character) < 32 for character in value) for value in identifiers):
        raise ValueError("The governed project mapping contains unsafe identifiers.")
    if source_property != ctx.context.graph_contract.authorization_path.source_property:
        raise ValueError("The governed mapping must use the trusted authorization property.")
    if not exact_values:
        raise ValueError("The governed mapping requires exact classification values.")

    observed = _observed_node_type(ctx.context, label)
    required_properties = {
        source_property, identity_property, classification_property,
        *(value for value in (type_property, category_property) if value),
        *(
            str(item.get("property") or "") for item in configured_fields
            if not item.get("optional", False)
        ),
    }
    unknown = required_properties - set(observed["properties"])
    if unknown:
        raise ValueError("Governed mapping properties are absent from the live graph: " + str(sorted(unknown)))
    rows = ctx.context.bim.query(
        f"MATCH (n:`{label}`) WHERE n.`{source_property}` IN $allowed_sources "
        f"RETURN count(DISTINCT n) AS node_count, "
        f"count(DISTINCT CASE WHEN n.`{identity_property}` IS NOT NULL THEN n END) AS populated_count, "
        f"count(DISTINCT n.`{identity_property}`) AS distinct_count, "
        f"collect(DISTINCT CASE WHEN n.`{classification_property}` IN $exact_values "
        + (
            f"AND n.`{category_property}` = $category_value "
            if category_property and category_value else ""
        )
        + f"THEN n.`{classification_property}` END) AS observed_values",
        {
            "allowed_sources": ctx.context.scope.allowed_sources,
            "exact_values": exact_values,
            "category_value": category_value,
        },
    )
    row = rows[0] if rows else {}
    node_count = int(row.get("node_count") or 0)
    populated = int(row.get("populated_count") or 0)
    distinct = int(row.get("distinct_count") or 0)
    if not node_count or populated != node_count or distinct != node_count:
        raise ValueError("The governed identity is not populated and unique across the live label.")
    observed_values = {str(value) for value in row.get("observed_values") or [] if value is not None}
    if observed_values != set(exact_values):
        raise ValueError(
            "The governed exact classification boundary changed; expected "
            + str(sorted(exact_values)) + ", observed " + str(sorted(observed_values))
        )

    fields = []
    if category_property:
        fields.append(SchemaFieldMapping(
            semantic_name="category", property=category_property,
            ontology_kind="canonical_type",
        ))
    classification_semantic_name = str(
        knowledge.get("classification_semantic_name") or "family"
    )
    fields.append(SchemaFieldMapping(
        semantic_name=classification_semantic_name, property=classification_property,
        aliases=[str(value) for value in knowledge.get("classification_aliases") or []],
        ontology_kind="canonical_type",
    ))
    if type_property:
        fields.append(SchemaFieldMapping(
            semantic_name="type", property=type_property,
            ontology_kind="canonical_type",
        ))
    existing_semantic_names = {field.semantic_name for field in fields}
    for item in configured_fields:
        semantic_name = str(item.get("semantic_name") or "")
        property_name = str(item.get("property") or "")
        if not semantic_name or not property_name:
            raise ValueError("Each governed field requires semantic_name and property.")
        if semantic_name in existing_semantic_names:
            raise ValueError(f"Duplicate governed semantic field: {semantic_name!r}.")
        fields.append(SchemaFieldMapping(
            semantic_name=semantic_name,
            property=property_name,
            aliases=[str(value) for value in item.get("aliases") or []],
            data_type=str(item.get("data_type") or "string"),
            unit=str(item.get("unit")) if item.get("unit") is not None else None,
            source_unit=(
                str(item.get("source_unit")) if item.get("source_unit") is not None
                else str(item.get("unit")) if item.get("unit") is not None else None
            ),
            conversion_factor=float(item.get("conversion_factor", 1.0)),
            conversion_basis=str(item.get("conversion_basis") or "identity conversion"),
            ontology_kind=item.get("ontology_kind"),
        ))
        existing_semantic_names.add(semantic_name)
    value_bindings = []
    if category_property and category_value:
        value_bindings.append(SchemaValueBinding(
            semantic_name="category",
            property=category_property,
            user_concept=f"{knowledge['entity_concept']} category",
            matches=[SchemaValueMatch(value=category_value, similarity=1.0)],
        ))
    governed_boundary_terms = list(dict.fromkeys(
        str(value).strip()
        for value in [
            knowledge["entity_concept"],
            *(knowledge.get("aliases") or []),
            *(knowledge.get("retrieval_terms") or []),
        ]
        if str(value).strip()
    ))
    value_bindings.append(SchemaValueBinding(
        semantic_name=classification_semantic_name,
        property=classification_property,
        # Retain multilingual/domain aliases in the governed boundary so package
        # constraints can be bound without relying on English-only token overlap.
        user_concept=" | ".join(governed_boundary_terms),
        matches=[SchemaValueMatch(value=value, similarity=1.0) for value in exact_values],
    ))
    relationship_bindings: list[SchemaRelationshipBinding] = []
    relationship_names: set[str] = set()
    structure = ctx.context.schema_discovery.get("project_graph_structure")
    if configured_relationships and not isinstance(structure, dict):
        structure = _project_graph_structure(ctx.context, include_properties=False)
        ctx.context.schema_discovery["project_graph_structure"] = structure
    observed_patterns = (structure or {}).get("relationship_types") or []
    for item in configured_relationships:
        semantic_name = str(item.get("semantic_name") or "")
        if not semantic_name or semantic_name in relationship_names:
            raise ValueError("Governed relationship binding names must be nonblank and unique.")
        raw_steps = item.get("steps") or []
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError(f"Relationship binding {semantic_name!r} requires at least one step.")
        steps = [SchemaRelationshipStep.model_validate(step) for step in raw_steps]
        previous_label = label
        for step in steps:
            if step.from_label != previous_label:
                raise ValueError(
                    f"Relationship binding {semantic_name!r} must be rooted at {label!r} and contiguous."
                )
            observed_step = any(
                step.relationship_type == pattern.get("type")
                and (
                    step.from_label in (pattern.get("from_labels") or [])
                    and step.to_label in (pattern.get("to_labels") or [])
                    if step.direction == "outgoing" else
                    step.from_label in (pattern.get("to_labels") or [])
                    and step.to_label in (pattern.get("from_labels") or [])
                )
                for pattern in observed_patterns
            )
            if not observed_step:
                raise ValueError(
                    f"Governed relationship binding {semantic_name!r} changed in the live graph."
                )
            previous_label = step.to_label
        relationship_bindings.append(SchemaRelationshipBinding(
            semantic_name=semantic_name,
            steps=steps,
            purpose=str(item.get("purpose") or "governed relationship evidence"),
            evidence_kind=str(item.get("evidence_kind") or "unspecified"),
            target_identity_property=str(item.get("target_identity_property") or ""),
            target_cardinality=str(item.get("target_cardinality") or "any"),
        ))
        target_identity_property = str(item.get("target_identity_property") or "")
        if target_identity_property:
            target_type = _observed_node_type(ctx.context, steps[-1].to_label)
            if target_identity_property not in set(target_type["properties"]):
                raise ValueError(
                    f"Governed relationship binding {semantic_name!r} target identity property "
                    f"{target_identity_property!r} changed in the live graph."
                )
        relationship_names.add(semantic_name)
    proposal = SchemaMappingProposal(
        entity_name=str(knowledge["entity_concept"]).replace(" ", "_"),
        label=label,
        identity_property=identity_property,
        source_property=source_property,
        fields=fields,
        value_bindings=value_bindings,
        relationship_bindings=relationship_bindings,
        counting_unit=str(knowledge["counting_unit"]),
        counting_unit_evidence=str(knowledge["counting_unit_semantics"]),
        reasoning_summary=(
            f"Activated governed project mapping {knowledge_key!r} after live identity and exact-value validation."
        ),
    )
    digest_input = {"project_id": ctx.context.scope.project_id, "proposal": proposal.model_dump(mode="json")}
    mapping_id = "mapping-" + hashlib.sha256(_json(digest_input).encode("utf-8")).hexdigest()[:12]
    existing = ctx.context.schema_mappings.get(mapping_id)
    if isinstance(existing, RegisteredSchemaMapping):
        return existing.model_dump_json()
    registered = RegisteredSchemaMapping(
        mapping_id=mapping_id,
        proposal=proposal,
        node_count=node_count,
        populated_identity_count=populated,
        distinct_identity_count=distinct,
    )
    ctx.context.schema_mappings[mapping_id] = registered
    ctx.context.add_artifact(RunArtifact(
        artifact_id=mapping_id, kind="mapping", producer="Schema Mapper",
        summary=f"Activated governed live mapping {knowledge_key}.",
        payload=registered.model_dump(mode="json"),
    ))
    return registered.model_dump_json()


def _query_plan_cache_key(context: BimRunContext, plan: BimQueryPlan) -> str:
    """Identify one exact executable plan within an immutable run scope."""
    mapping = context.schema_mappings.get(plan.mapping_id)
    mapping_payload = (
        mapping.model_dump(mode="json")
        if isinstance(mapping, RegisteredSchemaMapping)
        else None
    )
    stable = {
        "plan": plan.model_dump(mode="json", exclude_none=False),
        "scope": {
            "client_id": context.scope.client_id,
            "project_id": context.scope.project_id,
            "allowed_sources": sorted(set(context.scope.allowed_sources)),
        },
        "mapping": mapping_payload,
        "schema_fingerprint": context.schema_fingerprint,
        "graph_contract_version": context.graph_contract.version,
    }
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def _enforce_specialist_query_scope(context: BimRunContext, plan: BimQueryPlan) -> None:
    """Make specialist job boundaries executable policy, not prompt-only guidance."""
    role = context.active_specialist
    if role == "runtime":
        return
    is_requirements = plan.entity == "permit_knowledge"
    is_relationship = bool(
        plan.operation == "relationship_coverage"
        or plan.relationship
        or plan.relationship_filters
    )
    if role == "requirements" and not is_requirements:
        raise PermissionError("Requirements Specialist may query only scoped permit knowledge.")
    if role == "relationship" and (is_requirements or not is_relationship):
        raise PermissionError(
            "Relationship Specialist requires an explicit registered relationship operation."
        )
    if role == "quantity" and (is_requirements or is_relationship):
        raise PermissionError(
            "Quantity Specialist cannot execute requirements or relationship evidence work."
        )
    if role == "geometry":
        raise PermissionError("Geometry Specialist cannot execute an ordinary BIM query plan.")


def query_bim(ctx: PipelineContext, plan: BimQueryPlan) -> str:
    """Execute or reuse one exact validated project-scoped BIM query plan."""
    if ctx.context.task_contract is None:
        raise ValueError("The pipeline must define the BIM task contract before querying.")
    _enforce_specialist_query_scope(ctx.context, plan)
    if not plan.work_package_id:
        plan.work_package_id = ctx.context.workstream_id
    if not plan.constraint_bindings:
        plan.constraint_bindings = [
            ConstraintBinding.model_validate(item)
            for item in ctx.context.schema_discovery.get("active_constraint_bindings") or []
        ]
    missing_probe = any(item.operator == "is_missing" for item in plan.filters)
    if missing_probe:
        plan.role = "supporting"
        plan.include_in_answer = False
    elif plan.operation == "distinct" and not plan.satisfies:
        plan.role = "exploratory"
        plan.include_in_answer = False
    _validate_answer_metadata(ctx, plan)
    cache_key = _query_plan_cache_key(ctx.context, plan)
    # One branch owns one cache. Holding its re-entrant state lock through the
    # miss prevents simultaneous identical tool calls from executing twice.
    with ctx.context._lock:
        cache = ctx.context.schema_discovery.setdefault("_exact_query_result_cache", {})
        if not isinstance(cache, dict):
            raise RuntimeError("The per-run query-result cache is corrupted.")
        cached = cache.get(cache_key)
        if isinstance(cached, str):
            return cached
        response = _query_bim_uncached(ctx, plan)
        cache[cache_key] = response
        return response


def _query_bim_uncached(ctx: PipelineContext, plan: BimQueryPlan) -> str:
    result = _execute_plan(ctx.context, plan)
    diagnostics: list[str] = []
    coverage = (result.get("claim") or {}).get("coverage") or {}
    if (
        plan.operation in {"coverage", "relationship_coverage", "group_count", "distinct"}
        and int(coverage.get("candidate_count") or 0) > 0
        and int(coverage.get("matched_count") or 0) == 0
    ):
        diagnostics.append(
            "property_missing: entities exist but exhaustive coverage found no populated values"
            if plan.operation in {"group_count", "distinct"}
            else "verified_absence: exhaustive coverage found no matching values or paths"
        )
    elif result.get("matched_count") == 0:
        diagnostics.append(
            "zero_match: no records matched this exact entity/classification/filter hypothesis"
        )
    if any("missing data, not a zero" in item for item in result.get("limitations") or []):
        diagnostics.append("missing_metric: entities exist but the requested metric is unpopulated")
    if plan.answer_key:
        current_signature = (
            str((result.get("claim") or {}).get("value")),
            str((result.get("claim") or {}).get("unit")),
        )
        previous_signatures = set()
        for evidence in ctx.context.evidence.values():
            if evidence.kind != "query":
                continue
            previous = json.loads(evidence.payload)
            previous_plan = previous.get("plan") or {}
            if previous_plan.get("answer_key") == plan.answer_key:
                previous_claim = previous.get("claim") or {}
                previous_signatures.add((str(previous_claim.get("value")), str(previous_claim.get("unit"))))
        if previous_signatures and current_signature not in previous_signatures:
            diagnostics.append(
                "contradiction: this answer_key produced a value different from earlier evidence"
            )
    result["diagnostics"] = diagnostics
    evidence_id = f"query-{uuid4().hex[:10]}"
    ctx.context.add_evidence(Evidence(
        evidence_id=evidence_id,
        kind="query",
        summary=result["claim"]["statement"],
        payload=_json(result),
        work_package_id=ctx.context.workstream_id,
        constraint_bindings=list(plan.constraint_bindings),
    ))
    ctx.context.add_artifact(RunArtifact(
        artifact_id=f"plan-{evidence_id}",
        kind="query_plan",
        producer="Query Planner",
        summary=result["claim"]["statement"],
        payload={"evidence_id": evidence_id, **result},
    ))
    ctx.context.add_artifact(RunArtifact(
        artifact_id=f"cypher-{evidence_id}",
        kind="cypher_query",
        producer="Cypher Query Handler",
        summary=f"Executed {len(result.get('cypher_executions') or [])} read-only Cypher calculation(s).",
        payload={
            "evidence_id": evidence_id,
            "executions": result.get("cypher_executions") or [],
        },
    ))
    return _json({"evidence_id": evidence_id, **result})


_ADEQUACY_CHECK_NAMES = {
    "entity_grain_matches", "measurement_basis_matches", "population_complete",
    "planned_actual_distinguished", "absence_semantics_correct",
    "projection_answers_question", "requested_outputs_present",
}


def _semantic_token(value: str) -> str:
    """Canonical lexical form used only after typed/ontology anchors are selected."""
    words = re.findall(r"[^\W_]+", _normalize(value).replace("_", " "), flags=re.UNICODE)
    return " ".join(word[:-1] if len(word) > 3 and word.endswith("s") else word for word in words)


_SEMANTIC_ANCHORS: dict[str, tuple[str, ...]] = {
    "dwelling": ("apartment", "dwelling", "home", "housing unit", "residential unit"),
    "floor": ("floor", "storey", "story", "floor plate", "floorplate"),
    "space": ("space", "room"),
    "function": ("function", "functional", "use category"),
    "building": ("building", "tower"),
    "section": ("section", "massing section"),
    "facade": ("facade", "façade", "wall envelope"),
    "cable_tray": ("cable tray", "tray segment", "tray"),
    "conduit": ("conduit", "electrical conduit"),
    "switch": ("switch", "lighting switch"),
    "endpoint": ("endpoint", "socket", "outlet", "light fixture"),
    "type": ("type", "classification", "family"),
    "material": ("material",),
    "requirement": ("requirement", "criterion", "permit"),
}


def _semantic_anchor_set(*values: str) -> set[str]:
    """Extract stable concepts from free-form architect prose.

    Task contracts are model-authored and may describe one concept with a sentence,
    while mappings use short schema keys.  Verification therefore compares a small
    controlled concept vocabulary, not exact sentences.  Unrecognised prose never
    becomes an invented semantic assertion.
    """
    text = " ".join(_normalize(value).replace("_", " ") for value in values if value)
    return {
        anchor
        for anchor, terms in _SEMANTIC_ANCHORS.items()
        if any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) for term in terms)
    }


def _measurement_anchor_set(*values: str) -> set[str]:
    text = " ".join(_normalize(value).replace("_", " ") for value in values if value)
    anchors: set[str] = set()
    terms = {
        # Preserve measurement basis instead of collapsing every area-like term
        # into one permissive anchor. A generic area property does not prove that
        # the authored value is gross/BVO or net/NVO/VVO.
        "area": ("area", "m2", "m²"),
        "gross_area": ("bvo", "gfa", "gross floor", "gross area"),
        "net_area": ("nvo", "vvo", "net floor", "net area"),
        "length": ("length", "distance", "metre", "meter", "centimetre", "centimeter"),
        "vertical_position": ("elevation", "height above", "distance from floor", "offset"),
        "clear_height": ("clear height", "space height", "vertical extent"),
        "count": ("count", "number", "quantity", "how many"),
        "ratio": ("ratio", "percentage", "percent"),
        "material": ("material",),
        "classification": ("type", "classification", "family", "distinct value"),
    }
    for anchor, candidates in terms.items():
        if any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) for term in candidates):
            anchors.add(anchor)
    return anchors


def _field_semantics_match(expected: str, observed: str) -> bool:
    if _semantic_token(expected) == _semantic_token(observed):
        return True
    return _constraint_field(expected) == _constraint_field(observed)


def _boundary_value_matches(
    context: BimRunContext, field: str, expected: str, observed: str,
) -> bool:
    if _semantic_token(expected) == _semantic_token(observed):
        return True
    if _constraint_field(field) == "level" and _level_matches(
        expected, observed, context.graph_contract.level_aliases,
    ):
        return True
    expected_numbers = re.findall(r"-?\d+(?:[.,]\d+)?", _normalize(expected))
    observed_numbers = re.findall(r"-?\d+(?:[.,]\d+)?", _normalize(observed))
    if expected_numbers and {
        value.replace(",", ".") for value in expected_numbers
    } == {
        value.replace(",", ".") for value in observed_numbers
    }:
        return True
    expected_anchors = _semantic_anchor_set(expected)
    observed_anchors = _semantic_anchor_set(observed)
    return bool(expected_anchors and expected_anchors.issubset(observed_anchors))


def _expected_semantic_intents(
    context: BimRunContext, satisfies: list[str]
) -> list[SemanticIntent]:
    contract = context.task_contract
    if contract is None:
        return []
    specs = dict(zip(contract.required_outputs, contract.output_specs, strict=True))
    intents = [
        specs[output].semantic_intent
        for output in satisfies
        if output in specs and specs[output].semantic_intent.is_specified
    ]
    if not intents and contract.semantic_intent.is_specified:
        intents = [contract.semantic_intent]
    return intents


def _semantic_adequacy_checks(
    context: BimRunContext,
    plan: BimQueryPlan | GeometryQueryPlan,
    claim: dict[str, Any],
    *,
    registered: RegisteredSchemaMapping | None = None,
    matched_count: int = 0,
) -> list[dict[str, Any]]:
    """Challenge replayed evidence against the task's typed meaning.

    The observed side comes from compiler-owned fields, mapping metadata, exact
    filters and structured claim coverage.  A worker's prose is never sufficient.
    """
    if plan.role == "supporting" and not plan.satisfies:
        return [
            {
                "name": name,
                "passed": True,
                "explanation": (
                    "This replay-stable related fact is intentionally supporting-only "
                    "and is not treated as satisfying the direct semantic contract."
                ),
            }
            for name in sorted(_ADEQUACY_CHECK_NAMES)
        ]
    expected = _expected_semantic_intents(context, list(plan.satisfies))
    if not expected:
        return [
            {"name": name, "passed": True,
             "explanation": "No typed semantic constraint was supplied by this legacy contract."}
            for name in sorted(_ADEQUACY_CHECK_NAMES)
        ]

    if isinstance(plan, BimQueryPlan):
        observed_grain = (
            registered.proposal.counting_unit
            if isinstance(registered, RegisteredSchemaMapping)
            else plan.entity
        )
        observed_grain_terms = [observed_grain, plan.entity]
        if plan.group_by:
            observed_grain_terms.append(plan.group_by)
        observed_grain_terms.extend(plan.group_by_fields)
        observed_grain_terms.extend(
            value
            for item in plan.filters
            if item.operator in {"equals", "in"}
            for value in _raw_filter_values(item)
        )
        metric_field = plan.metric or plan.coverage_field or plan.group_by
        observed_metric_terms = {
            metric_field, plan.operation,
            plan.group_by,
            *plan.group_by_fields,
            *(item.field for item in plan.filters),
            *(value for item in plan.filters for value in _raw_filter_values(item)),
        }
        # Derive compound measurement meaning from the typed executable plan, not
        # from answer prose. ``group_summary`` and ``multi_group_summary`` compile
        # to both a distinct-record count and a summed numeric metric for every
        # grouping key. Their operation name alone does not contain the lexical
        # token "count", even though that count is part of the returned structure.
        if plan.operation in {"group_summary", "multi_group_summary"}:
            observed_metric_terms.add("count")
        if isinstance(registered, RegisteredSchemaMapping) and metric_field:
            mapped = next(
                (field for field in registered.proposal.fields if field.semantic_name == metric_field),
                None,
            )
            if mapped:
                observed_metric_terms.update([mapped.property, *mapped.aliases])
        if isinstance(registered, RegisteredSchemaMapping):
            grouping_names = {plan.group_by, *plan.group_by_fields} - {""}
            for grouping_name in grouping_names:
                mapped_group = next(
                    (
                        field for field in registered.proposal.fields
                        if field.semantic_name == grouping_name
                    ),
                    None,
                )
                if mapped_group:
                    observed_metric_terms.update([
                        mapped_group.property, *mapped_group.aliases,
                    ])
        observed_boundaries = {
            (_semantic_token(item.field), _semantic_token(value))
            for item in plan.filters
            if item.operator in {"equals", "in"}
            for value in _raw_filter_values(item)
        }
        # Instance-level graph queries are observations of the authored model unless
        # the selected field explicitly declares a planned/type-defined value.  This
        # inference is compiler-owned and cannot be overridden by worker prose.
        observed_origin = plan.semantic_intent.value_origin
        if observed_origin == "unspecified":
            origin_text = _normalize(" ".join(str(item) for item in observed_metric_terms))
            observed_origin = (
                "planned"
                if any(term in origin_text for term in ("planned", "default", "design", "type defined"))
                else "actual"
            )
        projection = {"value"}
        if claim.get("unit") is not None:
            projection.add("unit")
        if any(claim.get(key) for key in (
            "details", "basis", "method", "measurement", "source_tags", "coverage",
        )):
            projection.add("details")
        if plan.operation in {
            "group_count", "group_summary", "multi_group_count", "multi_group_summary",
            "maximum_group_sum", "distinct", "count_distinct",
        }:
            projection.add("groups")
        if claim.get("coverage") is not None:
            projection.add("coverage")
        if observed_origin == "comparison":
            projection.add("comparison")
        if plan.entity == "permit_knowledge":
            projection.add("compliance")
    else:
        # Governed geometry recipes do not expose an ordinary schema mapping.  Their
        # plan must therefore declare the configured semantic identity explicitly.
        observed_grain = plan.semantic_intent.entity_grain or plan.calculation
        observed_grain_terms = [observed_grain, plan.calculation, str(claim.get("basis") or "")]
        observed_metric_terms = {
            plan.semantic_intent.measurement_basis,
            plan.calculation,
            str(claim.get("basis") or ""),
            str((claim.get("measurement") or {}).get("source_property") or ""),
        }
        observed_boundaries = {
            (_semantic_token(boundary.semantic_field), _semantic_token(value))
            for boundary in plan.semantic_intent.population_boundary
            for value in boundary.exact_values
        }
        observed_origin = plan.semantic_intent.value_origin
        if observed_origin == "unspecified":
            observed_origin = (
                "comparison"
                if "comparison" in _normalize(
                    f"{plan.calculation} {claim.get('method') or ''} {claim.get('basis') or ''}"
                )
                else "actual"
            )
        projection = {"value"}
        if claim.get("unit") is not None:
            projection.add("unit")
        if any(claim.get(key) for key in (
            "details", "basis", "method", "measurement", "source_tags", "coverage",
        )):
            projection.add("details")
        if claim.get("details"):
            projection.add("groups")
        if claim.get("coverage") is not None:
            projection.add("coverage")
        if observed_origin == "comparison":
            projection.add("comparison")

    expected_grains = [item.entity_grain for item in expected if item.entity_grain]
    observed_grain_anchors = _semantic_anchor_set(*observed_grain_terms)
    grain_results = []
    for item in expected_grains:
        expected_anchors = _semantic_anchor_set(item)
        # Exact identifiers remain supported. Free-form prose is challenged only on
        # concepts the verifier can identify without guessing.
        grain_results.append(
            _semantic_token(item) == _semantic_token(observed_grain)
            or not expected_anchors
            or bool(expected_anchors & observed_grain_anchors)
        )
    grain_ok = all(grain_results)
    expected_bases = [item.measurement_basis for item in expected if item.measurement_basis]
    observed_basis_tokens = {_semantic_token(str(item)) for item in observed_metric_terms if item}
    observed_basis_anchors = _measurement_anchor_set(
        *(str(item) for item in observed_metric_terms if item)
    )
    basis_results = []
    for item in expected_bases:
        expected_anchors = _measurement_anchor_set(item)
        basis_results.append(
            _semantic_token(item) in observed_basis_tokens
            or not expected_anchors
            or expected_anchors.issubset(observed_basis_anchors)
        )
    basis_ok = all(basis_results)
    expected_boundaries = [
        (boundary.semantic_field, value)
        for intent in expected
        for boundary in intent.population_boundary
        for value in boundary.exact_values
    ]
    observed_boundary_values = [
        (item.field, value)
        for item in (plan.filters if isinstance(plan, BimQueryPlan) else [])
        if item.operator in {"equals", "in"}
        for value in _raw_filter_values(item)
    ]
    if isinstance(plan, BimQueryPlan):
        observed_boundary_values.extend(
            (binding.semantic_field, value)
            for binding in plan.constraint_bindings
            for value in [binding.requested_value, *binding.exact_values]
            if value
        )
    else:
        observed_boundary_values.extend(
            (boundary.semantic_field, value)
            for boundary in plan.semantic_intent.population_boundary
            for value in boundary.exact_values
        )
    def boundary_is_executable(expected_field: str, expected_value: str) -> bool:
        direct = any(
            _field_semantics_match(expected_field, observed_field)
            and _boundary_value_matches(
                context, observed_field, expected_value, observed_value,
            )
            for observed_field, observed_value in observed_boundary_values
        )
        if direct or not isinstance(plan, BimQueryPlan):
            return direct

        # A user's semantic boundary can be stored under a different physical
        # schema dimension.  For example, "non-metallic" may be encoded in an
        # exact Family-and-Type classification rather than a Material property.
        # Accept that remap only through the full compiler-owned chain:
        # typed boundary -> original task constraint -> registered binding ->
        # exact executable filter.  Neither model prose nor lexical similarity to
        # an arbitrary filter is sufficient on its own.
        for binding in plan.constraint_bindings:
            if not (
                _field_semantics_match(expected_field, binding.concept)
                and _boundary_value_matches(
                    context, expected_field, expected_value, binding.requested_value,
                )
            ):
                continue
            exact_values = [str(value) for value in binding.exact_values if str(value)]
            if not exact_values:
                continue
            executable = any(
                _field_semantics_match(binding.semantic_field, observed_field)
                and any(
                    _boundary_value_matches(
                        context, binding.semantic_field, exact_value, observed_value,
                    )
                    for exact_value in exact_values
                )
                for observed_field, observed_value in observed_boundary_values
            )
            if executable:
                return True
        return False

    boundary_ok = all(
        boundary_is_executable(expected_field, expected_value)
        for expected_field, expected_value in expected_boundaries
    )
    # Compiler-owned task-constraint coverage and governed classification checks
    # already prove the executable boundary. Do not let a redundant prose boundary
    # reject it solely because the architect used user-language rather than stored values.
    if isinstance(plan, BimQueryPlan):
        constraints_ok, _ = _plan_constraint_coverage(context, plan)
        boundary_ok = boundary_ok or constraints_ok and not expected_boundaries
        permit_coverage = claim.get("coverage") or {}
        if (
            plan.entity == "permit_knowledge"
            and bool(permit_coverage.get("exhaustive"))
            and int(permit_coverage.get("candidate_count") or 0) == 0
        ):
            # An exhaustive empty project requirements population proves that no
            # narrower model-element boundary can have an applicable record.
            boundary_ok = True
    elif expected_boundaries and not plan.semantic_intent.population_boundary:
        # A named geometry recipe is an immutable, project-governed population
        # compiler. Its configured boundary was checked before execution; the worker
        # cannot broaden it or supply arbitrary filters.
        boundary_ok = True
    expected_origins = {
        item.value_origin for item in expected if item.value_origin != "unspecified"
    }
    origin_ok = not expected_origins or (
        len(expected_origins) == 1 and observed_origin in expected_origins
    )
    requested_projection = {
        value for item in expected for value in item.requested_projection
    }
    projection_ok = requested_projection.issubset(projection)
    coverage = claim.get("coverage") or {}
    candidate_count = int(coverage.get("candidate_count") or 0)
    populated_count = int(coverage.get("matched_count") or 0)
    missing_count = int(coverage.get("missing_count") or 0)
    exhaustive = bool(coverage.get("exhaustive"))
    relevant_specs = []
    if context.task_contract:
        specs_by_output = dict(zip(
            context.task_contract.required_outputs,
            context.task_contract.output_specs,
            strict=True,
        ))
        relevant_specs = [
            specs_by_output[output] for output in plan.satisfies if output in specs_by_output
        ]
    absence_expectations = {
        item.absence_semantics for item in expected
        if item.absence_semantics != "unspecified"
    }
    if absence_expectations and not any(
        spec.kind in {"coverage", "relationship_coverage"} for spec in relevant_specs
    ):
        # "Unsupported" is a possible run outcome, not the intended meaning of an
        # ordinary count/measurement. Absence semantics gate only typed coverage work.
        absence_expectations = set()
    absence_ok = True
    absence_explanation = "No absence interpretation was requested."
    if len(absence_expectations) > 1:
        absence_ok = False
        absence_explanation = "Outputs assigned to one plan require incompatible absence meanings."
    elif absence_expectations:
        absence = next(iter(absence_expectations))
        if absence == "property_missing":
            absence_ok = (
                exhaustive and candidate_count > 0 and missing_count > 0
                and populated_count + missing_count == candidate_count
            )
        elif absence == "population_missing":
            absence_ok = exhaustive and candidate_count == 0
        elif absence == "verified_zero":
            absence_ok = claim.get("value") == 0 and (
                not coverage or exhaustive
            )
        else:  # unsupported
            absence_ok = claim.get("value") is None and bool(claim.get("caveats"))
        absence_explanation = (
            f"Structured evidence establishes {absence}." if absence_ok
            else f"Structured population evidence does not establish {absence}."
        )

    related_support = plan.role == "supporting" and not plan.satisfies
    requested_outputs_ok = related_support or (
        bool(plan.satisfies) and all(
            output in (context.task_contract.required_outputs if context.task_contract else [])
            for output in plan.satisfies
        )
    )
    return [
        {"name": "entity_grain_matches", "passed": grain_ok,
         "explanation": (
             f"Observed counting grain is {observed_grain!r}." if grain_ok
             else f"Observed counting grain {observed_grain!r} does not match {expected_grains!r}."
         )},
        {"name": "measurement_basis_matches", "passed": basis_ok,
         "explanation": (
             f"Observed measurement basis is {sorted(observed_metric_terms)!r}." if basis_ok
             else f"Observed measurement basis {sorted(observed_metric_terms)!r} does not match {expected_bases!r}."
         )},
        {"name": "population_complete", "passed": boundary_ok,
         "explanation": (
             "Every typed population boundary is an exact query filter." if boundary_ok
             else "The query omits or broadens a typed population boundary."
         )},
        {"name": "planned_actual_distinguished", "passed": origin_ok,
         "explanation": (
             f"Value origin is explicitly {observed_origin}." if origin_ok
             else f"Expected value origin {sorted(expected_origins)!r}, observed {observed_origin!r}."
         )},
        {"name": "absence_semantics_correct", "passed": absence_ok,
         "explanation": absence_explanation},
        {"name": "projection_answers_question", "passed": projection_ok,
         "explanation": (
             f"Returned projection includes {sorted(requested_projection)!r}." if projection_ok
             else f"Returned projection {sorted(projection)!r} omits {sorted(requested_projection - projection)!r}."
         )},
        {"name": "requested_outputs_present", "passed": requested_outputs_ok,
         "explanation": (
             "This replay-stable related fact is intentionally supporting-only."
             if related_support else
             "The plan is assigned to declared required outputs." if requested_outputs_ok
             else "The plan is not assigned to a declared required output."
         )},
    ]


def _verify_query_evidence(
    context: BimRunContext, requested_evidence_ids: list[str]
) -> dict[str, Any]:
    """Replay every query produced in this run, regardless of model-supplied IDs."""
    checks: list[dict[str, Any]] = []
    available_evidence_ids = [
        evidence_id
        for evidence_id, evidence in context.evidence.items()
        if evidence.kind == "query"
    ]
    for evidence_id in available_evidence_ids:
        evidence = context.evidence[evidence_id]
        original = json.loads(evidence.payload)
        if original.get("capability") in {"project_geometry", "project_calculation"}:
            plan = GeometryQueryPlan.model_validate(original["plan"])
            _validate_governed_calculation_schema(context, plan.calculation)
            report = calculate_project_geometry(
                plan.calculation, context.bim, context.scope.allowed_sources,
                context.bim.ontology.bim_query_knowledge,
            )
            claim = report.claims[0].model_dump(mode="json") if report and report.claims else {}
            stable = {"plan": plan.model_dump(mode="json"), "claim": claim,
                      "calculation": plan.calculation}
            digest = hashlib.sha256(
                json.dumps(stable, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
            ).hexdigest()
            replay_ok = digest == original.get("result_digest") and bool(claim.get("statement"))
            calculation_ok = bool(
                report and report.verification_status == "verified" and claim.get("basis")
            )
            counting_unit_ok = bool(claim.get("unit"))
            plausibility_flags = claim.get("plausibility_flags") or []
            plausibility_ok = not any(
                item.get("severity") == "error" for item in plausibility_flags
            )
            semantic_checks = [
                {"name": "replay_stability", "passed": replay_ok,
                 "explanation": "The geometry derivation is unchanged." if replay_ok else "The geometry derivation changed."},
                {"name": "authorized_scope", "passed": bool(context.scope.allowed_sources),
                 "explanation": "The derivation queried only authorized project sources."},
                {"name": "identity_integrity", "passed": calculation_ok,
                 "explanation": "The registered calculation uses its code-owned identity/deduplication recipe." if calculation_ok else "The calculation did not return structured provenance."},
                {"name": "constraint_binding", "passed": calculation_ok,
                 "explanation": "Project interpretation and exact boundaries came from the active governed calculation configuration."},
                {"name": "counting_unit", "passed": counting_unit_ok,
                 "explanation": "The calculation declared its result unit/counting unit." if counting_unit_ok else "The calculation omitted its result unit."},
                {"name": "boundary_exactness", "passed": calculation_ok,
                 "explanation": "The allowlisted compiler applied the governed population boundary."},
                {"name": "source_deduplication", "passed": calculation_ok,
                 "explanation": "The calculation recipe executes over the complete authorized source set with deterministic identity handling."},
                {"name": "classification_purity", "passed": calculation_ok,
                 "explanation": "Classification semantics came from active project configuration rather than model-authored query text."},
                {"name": "constraint_coverage", "passed": calculation_ok,
                 "explanation": "The calculation key and typed output were validated against the active project contract."},
                {"name": "measurement_plausibility", "passed": plausibility_ok,
                 "explanation": "No invariant-level plausibility error was found." if plausibility_ok else "The calculation violates a numeric invariant."},
            ]
            semantic_checks.extend(_semantic_adequacy_checks(
                context, plan, claim, matched_count=1,
            ))
            checks.append({
                "evidence_id": evidence_id, "verified": all(x["passed"] for x in semantic_checks),
                "expected_digest": original.get("result_digest"), "actual_digest": digest,
                "claim": claim, "limitations": report.limitations if report else [],
                "plan": plan.model_dump(mode="json"), "semantic_checks": semantic_checks,
                "plausibility_flags": plausibility_flags,
            })
            continue
        rerun = _execute_plan(context, BimQueryPlan.model_validate(original["plan"]))
        plan = BimQueryPlan.model_validate(rerun["plan"])
        registered = context.schema_mappings.get(plan.mapping_id)
        replay_ok = rerun["result_digest"] == original.get("result_digest")
        scope_ok = bool(context.scope.allowed_sources)
        identity_ok = True
        binding_ok = True
        identity_explanation = "The contract-backed entity defines a distinct identity property."
        binding_explanation = "No registered live value bindings apply to this plan."
        if isinstance(registered, RegisteredSchemaMapping):
            identity_ok = (
                registered.node_count == registered.populated_identity_count
                == registered.distinct_identity_count
            )
            identity_explanation = (
                f"{registered.distinct_identity_count} of {registered.node_count} records have "
                "populated, unique identities."
            )
            missing_bindings: list[str] = []
            for binding in registered.proposal.value_bindings:
                selected = {_normalize(match.value) for match in binding.matches}
                supplied = {
                    _normalize(value)
                    for item in plan.filters
                    if item.field == binding.semantic_name
                    for value in _raw_filter_values(item)
                }
                if not selected.issubset(supplied):
                    missing_bindings.append(binding.user_concept)
            binding_ok = not missing_bindings
            binding_explanation = (
                "Every live-mapped user constraint is present with its exact stored value."
                if binding_ok else f"Missing exact bindings for: {', '.join(missing_bindings)}."
            )
        counting_unit_ok = True
        counting_unit_explanation = "The contract-backed identity defines the record counting unit."
        boundary_ok = all(
            item.operator in {"equals", "in"}
            for item in plan.filters
            if _normalize(item.field) in {"level", "floor", "storey", "story"}
        )
        boundary_explanation = (
            "Level constraints use exact values, so ground cannot match underground."
            if boundary_ok else "A level constraint uses a non-exact operator."
        )
        source_dedup_ok = identity_ok and plan.distinct_by == "identity"
        source_dedup_explanation = (
            "The query counts the globally unique identity across all authorized sources."
            if source_dedup_ok else "Cross-source identity deduplication is not established."
        )
        classification_ok = True
        classification_explanation = "No mapped classification field applies to this plan."
        if isinstance(registered, RegisteredSchemaMapping):
            counting_unit_ok = bool(
                registered.proposal.counting_unit.strip()
                and registered.proposal.counting_unit_evidence.strip()
            )
            counting_unit_explanation = (
                f"One distinct identity represents {registered.proposal.counting_unit}: "
                f"{registered.proposal.counting_unit_evidence}"
                if counting_unit_ok else
                "The mapping does not prove whether one identity is the requested entity or a child record."
            )
            classification_fields = {
                field.semantic_name for field in registered.proposal.fields
                if field.ontology_kind in {"canonical_type", "ifc_class"}
            }
            if classification_fields:
                requested_classification_fields = {
                    binding.semantic_name
                    for binding in registered.proposal.value_bindings
                    if binding.semantic_name in classification_fields
                }
                classification_ok = not requested_classification_fields or all(
                    any(
                        item.field == semantic_name
                        and item.operator in {"equals", "in", "is_missing"}
                        for item in plan.filters
                    )
                    for semantic_name in requested_classification_fields
                )
                classification_explanation = (
                    "No specific classification was requested; the query retains the mapped entity scope."
                    if not requested_classification_fields else
                    "Every requested classification uses an exact mapped boundary."
                    if classification_ok else
                    "A requested classification is not constrained by its exact mapped boundary."
                )
            answer_producing = (
                plan.role == "answer_producing" and plan.include_in_answer is not False
            )
            coverage = (rerun.get("claim") or {}).get("coverage") or {}
            zero_population = int(
                coverage.get("candidate_count")
                if coverage.get("candidate_count") is not None
                else rerun.get("matched_count") or 0
            ) == 0
            unexpected_zero = (
                answer_producing
                and bool(registered.proposal.value_bindings)
                and zero_population
            )
            if unexpected_zero:
                classification_ok = False
                classification_explanation = (
                    "The exact live-mapped classification produced zero records. "
                    "This is an unresolved mapping contradiction, not a verified absence."
                )
        related_support = plan.role == "supporting" and not plan.satisfies
        constraints_ok, missing_constraints = _plan_constraint_coverage(context, plan)
        if related_support:
            constraints_ok = True
            missing_constraints = []
        constraint_explanation = (
            "This related supporting fact is not assigned to a required output."
            if related_support else
            "Every task constraint is represented by a query filter."
            if constraints_ok else
            "Missing task constraints: " + ", ".join(missing_constraints) + "."
        )
        coverage = (rerun.get("claim") or {}).get("coverage") or {}
        coverage_ok = not coverage or bool(coverage.get("exhaustive"))
        relationship_names = {
            item.semantic_name for item in registered.proposal.relationship_bindings
        } if isinstance(registered, RegisteredSchemaMapping) else set()
        requested_relationships = {
            item.relationship for item in plan.relationship_filters
        }
        if plan.operation == "relationship_coverage":
            requested_relationships.add(plan.relationship)
        relationship_ok = (
            not requested_relationships
            or bool(relationship_names)
            and requested_relationships.issubset(relationship_names)
        )
        relationship_explanation = (
            "Every relationship predicate expands from a fixed live-validated mapping path."
            if requested_relationships and relationship_ok else
            "This plan does not use a relationship predicate."
            if not requested_relationships else
            "The plan references an unregistered relationship path."
        )
        if relationship_ok and requested_relationships and context.task_contract:
            output_text = " ".join([
                context.task_contract.goal,
                *[
                    output for output in plan.satisfies
                    if output in context.task_contract.required_outputs
                ],
            ])
            physical_topology_required = bool(re.search(
                r"\b(?:physical topology|physical connect(?:ion|ivity)?|continuity|"
                r"port[- ]to[- ]port|wiring path|network path)\b",
                _normalize(output_text),
            ))
            if physical_topology_required:
                selected_bindings = [
                    binding for binding in registered.proposal.relationship_bindings
                    if binding.semantic_name in requested_relationships
                ]
                relationship_ok = bool(selected_bindings) and all(
                    binding.evidence_kind == "physical_topology"
                    and bool(binding.target_identity_property)
                    for binding in selected_bindings
                )
                relationship_explanation = (
                    "Physical continuity uses a live-validated topology path ending at "
                    "an independently identified target."
                    if relationship_ok else
                    "Physical continuity requires a physical_topology binding with an "
                    "independently identified final target; a generic edge or assignment "
                    "property is not sufficient."
                )
        plausibility_flags = (rerun.get("claim") or {}).get("plausibility_flags") or []
        plausibility_ok = not any(item.get("severity") == "error" for item in plausibility_flags)
        semantic_checks = [
            {"name": "replay_stability", "passed": replay_ok,
             "explanation": "The rerun result digest matches the original." if replay_ok
             else "The rerun result digest differs from the original."},
            {"name": "authorized_scope", "passed": scope_ok,
             "explanation": f"Execution is restricted to {len(context.scope.allowed_sources)} authorized source(s)."},
            {"name": "identity_integrity", "passed": identity_ok,
             "explanation": identity_explanation},
            {"name": "constraint_binding", "passed": binding_ok,
             "explanation": binding_explanation},
            {"name": "counting_unit", "passed": counting_unit_ok,
             "explanation": counting_unit_explanation},
            {"name": "boundary_exactness", "passed": boundary_ok,
             "explanation": boundary_explanation},
            {"name": "source_deduplication", "passed": source_dedup_ok,
             "explanation": source_dedup_explanation},
            {"name": "classification_purity", "passed": classification_ok,
             "explanation": classification_explanation},
            {"name": "constraint_coverage", "passed": binding_ok and constraints_ok,
             "explanation": constraint_explanation if binding_ok else binding_explanation},
            {"name": "population_coverage", "passed": coverage_ok,
             "explanation": (
                 "The result exposes an exhaustive evaluated population."
                 if coverage else "This operation does not claim property-absence coverage."
             )},
            {"name": "relationship_binding", "passed": relationship_ok,
             "explanation": relationship_explanation},
            {"name": "measurement_plausibility", "passed": plausibility_ok,
             "explanation": (
                 "No impossible numeric invariant was detected."
                 if plausibility_ok else "The computed result violates a numeric invariant."
            )},
        ]
        semantic_checks.extend(_semantic_adequacy_checks(
            context, plan, rerun.get("claim") or {},
            registered=registered if isinstance(registered, RegisteredSchemaMapping) else None,
            matched_count=int(rerun.get("matched_count") or 0),
        ))
        checks.append({
            "evidence_id": evidence_id,
            "verified": all(check["passed"] for check in semantic_checks),
            "expected_digest": original.get("result_digest"),
            "actual_digest": rerun["result_digest"],
            "claim": rerun["claim"],
            "limitations": rerun.get("limitations", []),
            "plan": rerun["plan"],
            "matched_count": rerun.get("matched_count"),
            "diagnostics": (
                ["verified_absence"]
                if plan.operation in {"coverage", "relationship_coverage", "group_count", "distinct"}
                and int(((rerun.get("claim") or {}).get("coverage") or {}).get("candidate_count") or 0) > 0
                and int(rerun.get("matched_count") or 0) == 0
                else ["zero_match"] if int(rerun.get("matched_count") or 0) == 0 else []
            ),
            "plausibility_flags": plausibility_flags,
            "semantic_checks": semantic_checks,
        })
    failed_adequacy_checks = sorted({
        item["name"]
        for check in checks
        for item in check.get("semantic_checks") or []
        if item.get("name") in _ADEQUACY_CHECK_NAMES and item.get("passed") is not True
    })
    semantic_adequacy = {
        "verified": bool(checks) and not failed_adequacy_checks,
        "failed_checks": failed_adequacy_checks,
        "layer": "deterministic_semantic_challenger",
    }
    return {
        "verified": bool(checks) and all(check.get("verified") is True for check in checks),
        "checks": checks,
        "semantic_adequacy": semantic_adequacy,
        "requested_evidence_ids": requested_evidence_ids,
        "verified_evidence_ids": available_evidence_ids,
    }


def ensure_bim_verification(context: BimRunContext) -> str | None:
    """Record deterministic verification when query evidence exists but the reviewer did not finish."""
    query_ids = [
        evidence_id for evidence_id, evidence in context.evidence.items()
        if evidence.kind == "query"
    ]
    existing = next(
        (
            evidence_id for evidence_id, evidence in reversed(list(context.evidence.items()))
            if evidence.kind == "verification"
            and set(json.loads(evidence.payload).get("verified_evidence_ids") or []) == set(query_ids)
        ),
        None,
    )
    if existing is not None:
        return existing
    if not query_ids:
        return None
    result = _verify_query_evidence(context, query_ids)
    verification_id = f"verification-{uuid4().hex[:10]}"
    context.add_evidence(Evidence(
        evidence_id=verification_id,
        kind="verification",
        summary=f"Runtime verification of {len(result['checks'])} general BIM query result(s)",
        payload=_json(result),
    ))
    context.add_artifact(RunArtifact(
        artifact_id=f"review-{verification_id}",
        kind="semantic_review",
        producer="Verifier",
        summary=f"Reviewed {len(result['checks'])} query artifact(s).",
        payload={"verification_evidence_id": verification_id, **result},
    ))
    return verification_id


def ensure_compliance_evidence(context: BimRunContext) -> str | None:
    """Ensure a compliance question has an explicit scoped requirements-availability query."""
    for evidence_id, evidence in context.evidence.items():
        if evidence.kind != "query":
            continue
        payload = json.loads(evidence.payload)
        if (payload.get("plan") or {}).get("entity") == "permit_knowledge":
            return evidence_id
    if context.task_contract is None:
        return None
    requirement_outputs = [
        output for output in context.task_contract.required_outputs
        if re.search(r"\b(compliance|requirement|permit|applicable)\b", output, re.I)
    ]
    return json.loads(query_bim(
        PipelineContext(context),
        BimQueryPlan(
            entity="permit_knowledge",
            operation="list",
            select=["name", "summary", "knowledge"],
            limit=20,
            role="supporting" if requirement_outputs else "exploratory",
            include_in_answer=False,
            satisfies=requirement_outputs,
        ),
    ))["evidence_id"]


def verify_bim_evidence(
    ctx: PipelineContext, evidence_ids: list[str]
) -> str:
    """Independently rerun saved general BIM query plans and compare their complete result digests."""
    verification_id = ensure_bim_verification(ctx.context)
    if verification_id is None:
        return _json({
            "evidence_id": None,
            "verified": False,
            "checks": [],
            "requested_evidence_ids": evidence_ids,
            "verified_evidence_ids": [],
        })
    result = json.loads(ctx.context.evidence[verification_id].payload)
    return _json({"evidence_id": verification_id, **result})
