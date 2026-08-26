import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bim_context import Settings
from bim_agents.graph_contract import load_graph_contract
from bim_agents.models import (
    BimRunContext,
    BimTaskContract,
    Evidence,
    EvidenceHandoff,
    EvidenceWorkPackage,
    EvidenceWorkstreamResult,
    OutputSpec,
    ProjectScope,
    TaskConstraint,
)
from bim_agents.observability import PipelineEvents
from bim_agents.orchestration import (
    _branch_limit, _checkpoint_handoff, _deterministic_geometry_plan, _deterministic_mapping_plans,
    _numeric_filter_request, _registered_mapping_handoff,
    _deterministic_requirements_plan, _scoped_requirements_are_absent,
    evidence_work_packages,
    package_contract, resolve_package_specialist, run_evidence_workstreams,
    _dependency_policy, _failed_dependencies,
    trusted_geometry_handoff, trusted_governed_mapping_handoff,
)
from bim_agents.registry import build_agent_registry
from bim_agents.runtime import _merge_isolated_workstream
from bim_agents.schema_mapping import RegisteredSchemaMapping, SchemaFieldMapping, SchemaMappingProposal


class FakeBimContext:
    def __init__(self):
        self.connected = False
        self.closed = False

    def connect(self):
        self.connected = True

    def close(self):
        self.closed = True


class RuntimeOrchestrationTests(unittest.IsolatedAsyncioTestCase):
    def contract(self):
        return BimTaskContract(
            goal="Count and classify switches", operation="count",
            entity_concept="switches",
            required_outputs=["switch count", "switch types"],
            work_packages=[
                EvidenceWorkPackage(
                    package_id="count", objective="Count physical switches",
                    required_outputs=["switch count"],
                ),
                EvidenceWorkPackage(
                    package_id="types", objective="Group switches by type",
                    required_outputs=["switch types"],
                ),
            ],
            success_criteria=["Both results are replayable"],
        )

    def test_numeric_threshold_constraints_compile_to_typed_operators(self):
        self.assertEqual(
            _numeric_filter_request("gross floor area strictly below 50 m²"),
            ("less_than", "50"),
        )
        self.assertEqual(
            _numeric_filter_request("height at least 2.7 m"),
            ("greater_or_equal", "2.7"),
        )
        self.assertIsNone(_numeric_filter_request("area should be reported"))

    def test_home_count_preserves_governed_aggregate_identity(self):
        contract = BimTaskContract(
            goal="Count homes below 50 square metres",
            operation="count",
            entity_concept="homes",
            constraints=[TaskConstraint(
                concept="gross floor area",
                requested_value="below 50 m2",
            )],
            required_outputs=["home count"],
            output_specs=[OutputSpec(key="home_count", kind="count")],
            success_criteria=["Count physical homes, not space records."],
        )
        package = EvidenceWorkPackage(
            package_id="small-homes",
            objective=contract.goal,
            required_outputs=contract.required_outputs,
            output_specs=contract.output_specs,
            constraints=contract.constraints,
        )
        mapping = RegisteredSchemaMapping(
            mapping_id="mapping-homes",
            proposal=SchemaMappingProposal(
                entity_name="spaces",
                label="IfcSpace",
                identity_property="GlobalID",
                source_property="source",
                fields=[
                    SchemaFieldMapping(
                        semantic_name="dwelling_unit_number",
                        property="EenheidNummer",
                        ontology_kind="aggregate_identity",
                    ),
                    SchemaFieldMapping(
                        semantic_name="area_m2",
                        property="Area",
                        aliases=["gross floor area"],
                        data_type="number",
                        unit="m2",
                    ),
                ],
                counting_unit="space record",
                counting_unit_evidence="GlobalID is unique per IfcSpace record.",
                reasoning_summary="A dwelling number groups the child space records into homes.",
            ),
            node_count=10,
            populated_identity_count=10,
            distinct_identity_count=10,
        )
        context = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
            question=contract.goal,
            task_contract=contract,
            schema_mappings={mapping.mapping_id: mapping},
        )

        handoff = _registered_mapping_handoff(
            context, mapping, contract, package, contract.goal,
        )
        plans = _deterministic_mapping_plans(context, handoff, package)

        self.assertEqual(handoff.operation, "count_distinct")
        self.assertEqual(handoff.grouping_fields, ["dwelling_unit_number"])
        self.assertIsNotNone(plans)
        self.assertEqual(plans[0].operation, "count_distinct")
        self.assertEqual(plans[0].group_by, "dwelling_unit_number")
        self.assertEqual(plans[0].filters[0].field, "area_m2")
        self.assertEqual(plans[0].filters[0].operator, "less_than")
        self.assertEqual(plans[0].filters[0].value, "50")

    def test_distinct_numeric_measurement_compiles_as_property_values(self):
        contract = BimTaskContract(
            goal="List distinct modeled heights above floor",
            operation="distinct",
            entity_concept="socket boxes",
            required_outputs=["distinct modeled heights"],
            output_specs=[OutputSpec(
                key="distinct_heights",
                kind="measurement",
                metric="elevation from level",
                required_unit="cm",
            )],
            success_criteria=["Return every distinct modeled numeric value."],
        )
        package = EvidenceWorkPackage(
            package_id="heights",
            objective=contract.goal,
            required_outputs=contract.required_outputs,
            output_specs=contract.output_specs,
        )
        mapping = RegisteredSchemaMapping(
            mapping_id="mapping-sockets",
            proposal=SchemaMappingProposal(
                entity_name="socket_boxes",
                label="IfcBuildingElementProxy",
                identity_property="GlobalID",
                source_property="source",
                fields=[SchemaFieldMapping(
                    semantic_name="elevation_from_level",
                    property="Elevation from Level",
                    aliases=["height above floor"],
                    data_type="number",
                    source_unit="cm",
                    unit="cm",
                )],
                counting_unit="physical socket-box instance",
                counting_unit_evidence="GlobalID is unique per instance.",
                reasoning_summary="The numeric instance property is the requested offset.",
            ),
            node_count=8,
            populated_identity_count=8,
            distinct_identity_count=8,
        )
        context = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
            question=contract.goal,
            task_contract=contract,
            schema_mappings={mapping.mapping_id: mapping},
        )
        handoff = _registered_mapping_handoff(
            context, mapping, contract, package, contract.goal,
        )

        plans = _deterministic_mapping_plans(context, handoff, package)

        self.assertIsNotNone(plans)
        self.assertEqual(plans[0].operation, "distinct")
        self.assertEqual(plans[0].group_by, "elevation_from_level")
        self.assertEqual(plans[0].metric, "")

    def test_unresolved_classification_compiles_related_population_support(self):
        contract = BimTaskContract(
            goal="Count apartments by an unavailable market sector",
            operation="count_distinct",
            entity_concept="residential apartments",
            constraints=[TaskConstraint(
                concept="market sector", requested_value="free sector",
            )],
            required_outputs=["free-sector apartment count"],
            output_specs=[OutputSpec(key="free_sector_count", kind="count")],
            success_criteria=["Do not infer an unavailable classification."],
        )
        package = EvidenceWorkPackage(
            package_id="sector-count",
            objective=contract.goal,
            required_outputs=contract.required_outputs,
            output_specs=contract.output_specs,
            constraints=contract.constraints,
        )
        mapping = RegisteredSchemaMapping(
            mapping_id="mapping-apartment-population",
            proposal=SchemaMappingProposal(
                entity_name="apartment_spaces",
                label="IfcSpace",
                identity_property="GlobalID",
                source_property="source",
                fields=[SchemaFieldMapping(
                    semantic_name="dwelling_unit_number",
                    property="EenheidNummer",
                    ontology_kind="aggregate_identity",
                )],
                counting_unit="apartment-space record",
                counting_unit_evidence="GlobalID is unique per space record.",
                reasoning_summary="Dwelling number groups apartment-space records.",
            ),
            node_count=105,
            populated_identity_count=105,
            distinct_identity_count=105,
        )
        context = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
            task_contract=contract,
            schema_mappings={mapping.mapping_id: mapping},
        )
        handoff = EvidenceHandoff(
            package_id=package.package_id,
            status="ready_for_query",
            route="live_mapping",
            entity="apartment_spaces",
            mapping_id=mapping.mapping_id,
            operation="count_distinct",
        )

        plans = _deterministic_mapping_plans(context, handoff, package)

        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].role, "supporting")
        self.assertFalse(plans[0].include_in_answer)
        self.assertEqual(plans[0].satisfies, [])
        self.assertEqual(plans[0].operation, "count_distinct")
        self.assertEqual(plans[0].group_by, "dwelling_unit_number")

    def test_requirements_absence_compiles_deterministic_compliance_outcome(self):
        root = BimRunContext(
            bim=object(), scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
        )
        root.add_evidence(Evidence(
            evidence_id="requirements-zero",
            kind="query",
            summary="No scoped requirements",
            payload=json.dumps({
                "plan": {"entity": "permit_knowledge"},
                "claim": {
                    "value": 0,
                    "coverage": {"candidate_count": 0, "exhaustive": True},
                },
            }),
        ))
        package = EvidenceWorkPackage(
            package_id="compliance",
            objective="Determine compliance when requirements are unavailable",
            required_outputs=["compliance determination"],
            output_specs=[OutputSpec(key="compliance", kind="compliance")],
        )
        contract = BimTaskContract(
            goal=package.objective,
            operation="list",
            entity_concept="requirements",
            required_outputs=package.required_outputs,
            output_specs=package.output_specs,
            success_criteria=["Return an explicit not-assessable outcome."],
        )

        plan = _deterministic_requirements_plan(root, contract, package)

        self.assertTrue(_scoped_requirements_are_absent(root))
        self.assertIsNotNone(plan)
        self.assertEqual(plan.entity, "permit_knowledge")
        self.assertEqual(plan.semantic_intent.value_origin, "comparison")
        self.assertEqual(plan.satisfies, package.required_outputs)

    def test_package_projection_contains_only_assigned_outputs(self):
        contract = self.contract()
        packages = evidence_work_packages(contract)

        projected = package_contract(contract, packages[0])

        self.assertEqual(projected.required_outputs, ["switch count"])
        self.assertEqual(projected.work_packages, [])

    def test_typed_outputs_resolve_terminal_specialists(self):
        cases = [
            (EvidenceWorkPackage(
                package_id="quantity", objective="Count objects",
                required_outputs=["count"],
                output_specs=[OutputSpec(key="count", kind="count")],
            ), "quantity"),
            (EvidenceWorkPackage(
                package_id="relations", objective="Check connectivity",
                required_outputs=["coverage"],
                output_specs=[OutputSpec(key="coverage", kind="relationship_coverage")],
            ), "relationship"),
            (EvidenceWorkPackage(
                package_id="geometry", objective="Calculate geometry",
                required_outputs=["area"], route_hint="geometry",
            ), "geometry"),
            (EvidenceWorkPackage(
                package_id="requirements", objective="Check requirements",
                required_outputs=["compliance"],
                output_specs=[OutputSpec(key="compliance", kind="compliance")],
            ), "requirements"),
        ]
        for package, expected in cases:
            with self.subTest(package=package.package_id):
                self.assertEqual(resolve_package_specialist(package), expected)

    def test_dependency_policy_keeps_requirements_availability_independent(self):
        requirements = EvidenceWorkPackage(
            package_id="requirements", objective="Find applicable requirements",
            required_outputs=["applicable requirements"], route_hint="requirements",
            depends_on=["facts"],
        )
        facts = EvidenceWorkPackage(
            package_id="facts", objective="Resolve model facts", required_outputs=["facts"],
        )

        self.assertEqual(_dependency_policy(requirements), "independent")
        self.assertEqual(_failed_dependencies(requirements, {
            "facts": SimpleNamespace(status="error"),
        }), [])
        self.assertEqual(_dependency_policy(facts), "all_success")

    def test_compliance_dependency_policy_always_requires_success(self):
        comparison = EvidenceWorkPackage(
            package_id="comparison", objective="Compare facts with requirements",
            required_outputs=["compliance"],
            output_specs=[OutputSpec(key="compliance", kind="compliance")],
            dependency_policy="independent", depends_on=["facts", "requirements"],
        )

        self.assertEqual(_dependency_policy(comparison), "all_success")
        self.assertEqual(
            _failed_dependencies(comparison, {
                "facts": SimpleNamespace(status="query_completed"),
                "requirements": SimpleNamespace(status="partial_completed"),
            }),
            ["requirements"],
        )

    def test_compliance_without_both_typed_inputs_is_blocked(self):
        comparison = EvidenceWorkPackage(
            package_id="comparison", objective="Compare facts with requirements",
            required_outputs=["compliance"],
            output_specs=[OutputSpec(key="compliance", kind="compliance")],
        )

        self.assertEqual(
            _failed_dependencies(comparison, {}),
            ["model-facts input", "requirements input"],
        )

    def test_parallel_branch_budgets_never_exceed_root_remainder(self):
        allocations = [_branch_limit(10, 2, 4, slot) for slot in range(4)]
        self.assertEqual(allocations, [2, 2, 2, 2])
        self.assertEqual(sum(allocations), 8)

    def test_explicit_specialist_cannot_override_typed_job(self):
        package = EvidenceWorkPackage(
            package_id="relations", objective="Check connectivity",
            required_outputs=["coverage"], specialist="quantity",
            output_specs=[OutputSpec(key="coverage", kind="relationship_coverage")],
        )
        with self.assertRaisesRegex(ValueError, "requires 'relationship'"):
            resolve_package_specialist(package)

    def test_package_projection_does_not_leak_sibling_floor_constraint(self):
        contract = BimTaskContract(
            goal="Count all and ground-floor units", operation="count_distinct",
            entity_concept="apartments",
            constraints=[TaskConstraint(concept="floor", requested_value="ground floor")],
            required_outputs=["all units", "ground-floor units"],
            work_packages=[
                EvidenceWorkPackage(
                    package_id="all", objective="Count all units", required_outputs=["all units"],
                    constraints=[],
                ),
                EvidenceWorkPackage(
                    package_id="ground", objective="Count ground-floor units",
                    required_outputs=["ground-floor units"],
                    constraints=[TaskConstraint(concept="floor", requested_value="ground floor")],
                ),
            ],
            success_criteria=["Both counts"],
        )
        self.assertEqual(package_contract(contract, contract.work_packages[0]).constraints, [])
        self.assertEqual(
            package_contract(contract, contract.work_packages[1]).constraints[0].requested_value,
            "ground floor",
        )

    def test_exact_named_geometry_route_precedes_schema_discovery(self):
        root = BimRunContext(
            bim=SimpleNamespace(ontology=SimpleNamespace(bim_query_knowledge={
                "massing": {
                    "calculation": "section_heights",
                    "route_terms": ["various section heights"],
                    "semantics": "Project massing tops.",
                }
            })),
            scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
            question="What are the various section heights?",
            task_contract=BimTaskContract(
                goal="Report section heights", operation="list", entity_concept="sections",
                required_outputs=["section heights"], success_criteria=["Measured heights"],
            ),
        )
        package = EvidenceWorkPackage(
            package_id="heights", objective="Report various section heights",
            required_outputs=["section heights"],
        )
        handoff = trusted_geometry_handoff(root, package)
        self.assertEqual(handoff.route, "geometry")
        self.assertEqual(handoff.calculation, "section_heights")

    def test_exact_geometry_route_overrides_scout_live_schema_hint(self):
        root = BimRunContext(
            bim=SimpleNamespace(ontology=SimpleNamespace(bim_query_knowledge={
                "endpoints": {
                    "calculation": "electrical_endpoints_by_level",
                    "route_terms": ["נקודות קצה"],
                }
            })),
            scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
            question="כמה נקודות קצה מתוכננות בקומה?",
            task_contract=BimTaskContract(
                goal="Count endpoints", operation="group_count",
                entity_concept="endpoints", required_outputs=["endpoint counts"],
                success_criteria=["Governed calculation"],
            ),
        )
        package = EvidenceWorkPackage(
            package_id="endpoints", objective="Investigate the live endpoint schema",
            required_outputs=["endpoint counts"], route_hint="live_schema",
        )

        handoff = trusted_geometry_handoff(root, package)

        self.assertEqual(handoff.calculation, "electrical_endpoints_by_level")

    def test_support_only_governed_calculation_does_not_claim_connectivity_output(self):
        output = "physical connection or continuity to the feeding electrical panel"
        contract = BimTaskContract(
            goal="Assess electrical connectivity", operation="relationship_coverage",
            entity_concept="electrical components", required_outputs=[output],
            output_specs=[OutputSpec(key=output, kind="relationship_coverage")],
            success_criteria=["Keep logical assignment separate from physical continuity."],
        )
        knowledge = {
            "panel_coverage": {
                "calculation": "electrical_panel_assignment_coverage",
                "recipe": "property_coverage",
                "support_only": True,
                "output_kinds": ["count"],
                "semantic_intent": {
                    "value_origin": "planned",
                    "absence_semantics": "property_missing",
                },
                "route_terms": ["feeding electrical panel"],
            }
        }
        bim = SimpleNamespace(ontology=SimpleNamespace(bim_query_knowledge=knowledge))
        root = BimRunContext(
            bim=bim, scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(), question=contract.goal,
            task_contract=contract,
        )
        branch = BimRunContext(
            bim=bim, scope=root.scope, graph_contract=root.graph_contract,
            question=root.question, task_contract=contract,
        )
        package = EvidenceWorkPackage(
            package_id="connectivity", objective=output,
            required_outputs=[output], output_specs=contract.output_specs,
        )

        handoff = trusted_geometry_handoff(root, package)
        plan = _deterministic_geometry_plan(branch, handoff, package)

        self.assertEqual(handoff.calculation, "electrical_panel_assignment_coverage")
        self.assertEqual(plan.role, "supporting")
        self.assertFalse(plan.include_in_answer)
        self.assertEqual(plan.satisfies, [])

    def test_governed_mapping_can_override_incorrect_contract_route_hint(self):
        knowledge = {"trays": {
            "entity_concept": "cable tray segments",
            "aliases": ["מגשי הכבלים"],
            "exact_family_values": ["Wire Mesh Cable Tray"],
        }}
        contract = BimTaskContract(
            goal="סכם מגשי הכבלים", operation="count",
            entity_concept="מגשי הכבלים", required_outputs=["count"],
            success_criteria=["Governed mapping"],
        )
        bim = SimpleNamespace(ontology=SimpleNamespace(bim_query_knowledge=knowledge))
        root = BimRunContext(
            bim=bim, scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(), question=contract.goal,
            task_contract=contract,
        )
        branch = BimRunContext(
            bim=bim, scope=root.scope, graph_contract=root.graph_contract,
            question=root.question, task_contract=contract,
        )
        package = EvidenceWorkPackage(
            package_id="trays", objective=contract.goal,
            required_outputs=contract.required_outputs, route_hint="contract",
        )
        registered = self._governed_mapping(entity="cable tray segments")
        with patch(
            "bim_agents.orchestration.activate_project_knowledge_mapping",
            return_value=registered.model_dump_json(),
        ):
            handoff = trusted_governed_mapping_handoff(root, branch, contract, package)

        self.assertEqual(handoff.mapping_id, registered.mapping_id)

    def test_multilingual_broad_inventory_and_endpoint_intents_route_to_calculations(self):
        knowledge = {
            "broad_switch_inventory": {
                "calculation": "broad_switch_inventory",
                "route_terms": ["סוגי מפסקים בפרויקט"],
                "semantics": "Broad governed switch scope.",
            },
            "electrical_endpoints_by_level": {
                "calculation": "electrical_endpoints_by_level",
                "route_terms": ["נקודות קצה"],
                "semantics": "Governed electrical endpoints grouped by floor.",
            },
        }
        for question, expected in (
            ("אילו סוגי מפסקים בפרויקט?", "broad_switch_inventory"),
            ("כמה נקודות קצה מתוכננות בקומה?", "electrical_endpoints_by_level"),
        ):
            contract = BimTaskContract(
                goal=question,
                operation="list",
                entity_concept="electrical elements",
                required_outputs=["answer"],
                success_criteria=["Governed calculation"],
            )
            root = BimRunContext(
                bim=SimpleNamespace(ontology=SimpleNamespace(bim_query_knowledge=knowledge)),
                scope=ProjectScope(client_id="c", project_id="p"),
                graph_contract=load_graph_contract(),
                question=question,
                task_contract=contract,
            )
            package = EvidenceWorkPackage(
                package_id="electrical-answer",
                objective=question,
                required_outputs=["answer"],
            )

            handoff = trusted_geometry_handoff(root, package)

            self.assertEqual(handoff.calculation, expected)

    def test_unicode_hebrew_endpoint_intent_routes_to_calculation(self):
        question = "כמה נקודות קצה מתוכננות בקומה?"
        contract = BimTaskContract(
            goal=question, operation="group_count", entity_concept="electrical endpoints",
            required_outputs=["endpoint count by floor"],
            success_criteria=["Governed calculation"],
        )
        root = BimRunContext(
            bim=SimpleNamespace(ontology=SimpleNamespace(bim_query_knowledge={
                "endpoints": {
                    "calculation": "electrical_endpoints_by_level",
                    "route_terms": ["נקודות קצה"],
                }
            })),
            scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(), question=question,
            task_contract=contract,
        )
        package = EvidenceWorkPackage(
            package_id="endpoints", objective=question,
            required_outputs=contract.required_outputs,
        )

        handoff = trusted_geometry_handoff(root, package)

        self.assertEqual(handoff.calculation, "electrical_endpoints_by_level")

    def test_function_area_request_prefers_governed_project_mapping(self):
        knowledge = {
            "functional_spaces": {
                "entity_concept": "functional spaces",
                "aliases": ["functions on the ground floor", "function types"],
                "exact_family_values": ["parking", "retail"],
            },
        }
        contract = BimTaskContract(
            goal="Report function types and their areas on the ground floor",
            operation="group_summary",
            entity_concept="functional spaces",
            constraints=[TaskConstraint(
                concept="floor", requested_value="ground floor",
            )],
            required_outputs=["function types, counts, and areas"],
            success_criteria=["Replayable grouped summary"],
        )
        bim = SimpleNamespace(
            ontology=SimpleNamespace(bim_query_knowledge=knowledge),
            query=lambda *_args, **_kwargs: [{"value": "00 begane grond"}],
        )
        root = BimRunContext(
            bim=bim,
            scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(),
            question=contract.goal,
            task_contract=contract,
        )
        branch = BimRunContext(
            bim=root.bim,
            scope=root.scope,
            graph_contract=root.graph_contract,
            question=root.question,
            task_contract=contract,
        )
        package = EvidenceWorkPackage(
            package_id="functions",
            objective=contract.goal,
            required_outputs=contract.required_outputs,
            output_specs=[OutputSpec(
                key="function types, counts, and areas",
                kind="grouped_summary",
                metric="area",
                grouping_dimensions=["function type"],
            )],
        )
        mapping = self._governed_mapping(entity="functional spaces")
        mapping.proposal = mapping.proposal.model_copy(update={
            "label": "IfcSpace",
            "fields": [
                SchemaFieldMapping(
                    semantic_name="function_type", property="canonical_type",
                ),
                SchemaFieldMapping(
                    semantic_name="level", property="canonical_level", ontology_kind="level",
                ),
                SchemaFieldMapping(
                    semantic_name="area_m2", property="canonical_area_m2",
                    data_type="number", unit="m²",
                ),
            ],
        })
        branch.schema_mappings[mapping.mapping_id] = mapping

        handoff = _checkpoint_handoff(root, branch, package)

        self.assertEqual(handoff.operation, "group_summary")
        self.assertEqual(handoff.metric, "area_m2")
        self.assertEqual(handoff.grouping_fields, ["function_type"])

    def test_registered_mapping_checkpoint_recovers_scout_turn_limit(self):
        contract = self.contract()
        root = BimRunContext(
            bim=object(), scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(), question="Count switches",
            task_contract=contract,
        )
        branch = BimRunContext(
            bim=object(), scope=root.scope, graph_contract=root.graph_contract,
            question=root.question, task_contract=package_contract(
                contract, contract.work_packages[0]
            ),
        )
        mapping = RegisteredSchemaMapping(
            mapping_id="mapping-switches",
            proposal=SchemaMappingProposal(
                entity_name="switches", label="IfcProduct",
                identity_property="GlobalID", source_property="source",
                fields=[SchemaFieldMapping(
                    semantic_name="type", property="family_and_type",
                )],
                counting_unit="physical switch",
                counting_unit_evidence="GlobalID is unique per switch.",
                reasoning_summary="Validated live mapping.",
            ),
            node_count=21, populated_identity_count=21, distinct_identity_count=21,
        )
        branch.schema_mappings[mapping.mapping_id] = mapping

        handoff = _checkpoint_handoff(root, branch, contract.work_packages[0])

        self.assertEqual(handoff.route, "live_mapping")
        self.assertEqual(handoff.mapping_id, mapping.mapping_id)
        self.assertIn("type", handoff.evidence_summary)

    def _governed_mapping(self, mapping_id="mapping-governed", entity="elements"):
        return RegisteredSchemaMapping(
            mapping_id=mapping_id,
            proposal=SchemaMappingProposal(
                entity_name=entity, label="IfcFlowSegment",
                identity_property="GlobalID", source_property="source",
                fields=[SchemaFieldMapping(
                    semantic_name="length", property="Length", data_type="number", unit="m",
                )],
                counting_unit="physical segment",
                counting_unit_evidence="GlobalID is unique per physical segment.",
                reasoning_summary="Live-validated governed mapping.",
            ),
            node_count=8, populated_identity_count=8, distinct_identity_count=8,
        )

    def _routing_context(self, knowledge, question, entity_concept="elements"):
        contract = BimTaskContract(
            goal="Resolve governed elements", operation="sum",
            entity_concept=entity_concept, required_outputs=["total length"],
            success_criteria=["Replayable total"],
        )
        root = BimRunContext(
            bim=SimpleNamespace(ontology=SimpleNamespace(bim_query_knowledge=knowledge)),
            scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(), question=question, task_contract=contract,
        )
        branch = BimRunContext(
            bim=root.bim, scope=root.scope, graph_contract=root.graph_contract,
            question=question, task_contract=contract,
        )
        package = EvidenceWorkPackage(
            package_id="governed", objective=question, required_outputs=["total length"],
        )
        return root, branch, contract, package

    def test_governed_conduit_retrieval_term_routes_without_scout(self):
        knowledge = {
            "nonmetallic_conduits": {
                "entity_concept": "nonmetallic electrical conduit segments",
                "aliases": ["electrical conduit"],
                "retrieval_terms": ["HDPE"],
                "exact_family_values": ["Electrical Nonmetallic Conduit (HDPE)"],
            }
        }
        root, branch, contract, package = self._routing_context(
            knowledge, "What is the total HDPE length?",
        )
        registered = self._governed_mapping(entity="nonmetallic_conduit_segments")

        with patch(
            "bim_agents.orchestration.activate_project_knowledge_mapping",
            return_value=registered.model_dump_json(),
        ) as activate:
            handoff = trusted_governed_mapping_handoff(
                root, branch, contract, package,
            )

        activate.assert_called_once()
        self.assertEqual(activate.call_args.args[1], "nonmetallic_conduits")
        self.assertEqual(handoff.route, "live_mapping")
        self.assertEqual(handoff.mapping_id, registered.mapping_id)
        self.assertEqual(handoff.metric, "length")

    def test_governed_apartment_floor_route_precedes_schema_scout(self):
        knowledge = {
            "apartment_spaces": {
                "entity_concept": "apartment spaces",
                "aliases": ["apartments"],
                "retrieval_terms": ["apartments on the ground floor"],
                "exact_family_values": ["apartment"],
            }
        }
        root, branch, contract, package = self._routing_context(
            knowledge,
            "Are there apartments on the ground floor?",
            entity_concept="apartments",
        )
        registered = self._governed_mapping(entity="apartment_spaces")

        with patch(
            "bim_agents.orchestration.activate_project_knowledge_mapping",
            return_value=registered.model_dump_json(),
        ) as activate:
            handoff = trusted_governed_mapping_handoff(
                root, branch, contract, package,
            )

        activate.assert_called_once()
        self.assertEqual(activate.call_args.args[1], "apartment_spaces")
        self.assertEqual(handoff.route, "live_mapping")

    def test_governed_handoff_resolves_requested_floor_to_exact_live_value(self):
        knowledge = {
            "apartment_spaces": {
                "entity_concept": "apartment spaces",
                "aliases": ["apartments"],
                "exact_family_values": ["apartment"],
            }
        }
        bim = SimpleNamespace(
            ontology=SimpleNamespace(bim_query_knowledge=knowledge),
            query=lambda _cypher, _parameters: [
                {"value": "00 begane grond"},
                {"value": "01 eerste verdieping"},
                {"value": "07 zevende verdieping"},
            ],
        )
        contract = BimTaskContract(
            goal="Count ground-floor apartments",
            operation="count",
            entity_concept="apartments",
            constraints=[TaskConstraint(concept="floor", requested_value="ground floor")],
            required_outputs=["ground-floor apartment count"],
            success_criteria=["Use the exact live floor value."],
        )
        root = BimRunContext(
            bim=bim,
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
            question="Are there apartments on the ground floor?",
            task_contract=contract,
        )
        branch = BimRunContext(
            bim=bim,
            scope=root.scope,
            graph_contract=root.graph_contract,
            question=root.question,
            task_contract=contract,
        )
        package = EvidenceWorkPackage(
            package_id="ground-apartments",
            objective="Count ground-floor apartments",
            required_outputs=["ground-floor apartment count"],
            constraints=[TaskConstraint(concept="floor", requested_value="ground floor")],
        )
        registered = RegisteredSchemaMapping(
            mapping_id="mapping-apartments",
            proposal=SchemaMappingProposal(
                entity_name="apartment_spaces",
                label="IfcSpace",
                identity_property="GlobalID",
                source_property="source",
                fields=[SchemaFieldMapping(
                    semantic_name="level",
                    property="canonical_level",
                    ontology_kind="level",
                )],
                counting_unit="physical apartment-space record",
                counting_unit_evidence="GlobalID is unique per space.",
                reasoning_summary="Live-validated governed mapping.",
            ),
            node_count=10,
            populated_identity_count=10,
            distinct_identity_count=10,
        )

        with patch(
            "bim_agents.orchestration.activate_project_knowledge_mapping",
            return_value=registered.model_dump_json(),
        ):
            handoff = trusted_governed_mapping_handoff(
                root, branch, contract, package,
            )

        floor = next(
            item for item in handoff.constraint_bindings
            if item.semantic_field == "level"
        )
        self.assertEqual(floor.exact_values, ["00 begane grond"])
        self.assertEqual(floor.package_id, "ground-apartments")
        self.assertEqual(
            next(item for item in handoff.constraints if item.semantic_field == "level").exact_values,
            ["00 begane grond"],
        )

    def test_typed_grouped_summary_handoff_derives_generic_dimensions_and_metric(self):
        knowledge = {
            "segments": {
                "entity_concept": "distribution segments",
                "aliases": ["segments"],
                "exact_family_values": ["Segment A"],
            }
        }
        contract = BimTaskContract(
            goal="Summarize segment length by type, width, and material",
            operation="group_summary",
            entity_concept="segments",
            required_outputs=["length breakdown"],
            output_specs=[OutputSpec(
                key="length breakdown",
                kind="grouped_summary",
                metric="total length",
                grouping_dimensions=["type", "width", "material"],
                required_unit="m",
            )],
            success_criteria=["Every grouping dimension and metric is executable."],
        )
        bim = SimpleNamespace(ontology=SimpleNamespace(bim_query_knowledge=knowledge))
        root = BimRunContext(
            bim=bim, scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(), question=contract.goal,
            task_contract=contract,
        )
        branch = BimRunContext(
            bim=bim, scope=root.scope, graph_contract=root.graph_contract,
            question=root.question, task_contract=contract,
        )
        package = EvidenceWorkPackage(
            package_id="segment-summary", objective=contract.goal,
            required_outputs=contract.required_outputs,
            output_specs=contract.output_specs,
        )
        registered = RegisteredSchemaMapping(
            mapping_id="mapping-segments",
            proposal=SchemaMappingProposal(
                entity_name="distribution_segments", label="IfcFlowSegment",
                identity_property="GlobalID", source_property="source",
                fields=[
                    SchemaFieldMapping(semantic_name="type", property="Type"),
                    SchemaFieldMapping(semantic_name="width", property="Width", data_type="number", unit="mm"),
                    SchemaFieldMapping(semantic_name="material", property="Material"),
                    SchemaFieldMapping(semantic_name="length", property="Length", data_type="number", unit="m"),
                ],
                counting_unit="physical segment",
                counting_unit_evidence="GlobalID is unique per segment.",
                reasoning_summary="Live-validated governed mapping.",
            ),
            node_count=10, populated_identity_count=10, distinct_identity_count=10,
        )

        with patch(
            "bim_agents.orchestration.activate_project_knowledge_mapping",
            return_value=registered.model_dump_json(),
        ):
            handoff = trusted_governed_mapping_handoff(root, branch, contract, package)

        self.assertEqual(handoff.operation, "multi_group_summary")
        self.assertEqual(handoff.metric, "length")
        self.assertEqual(handoff.grouping_fields, ["type", "width", "material"])

    def test_governed_query_profile_adds_domain_summary_without_result_values(self):
        knowledge = {"functional_spaces": {
            "entity_concept": "functional spaces",
            "aliases": ["function types"],
            "exact_family_values": ["parking", "retail"],
            "query_profile": {
                "apply_when_operations": ["list", "distinct"],
                "operation": "group_summary",
                "metric": "area_m2",
                "grouping_fields": ["function_type"],
            },
        }}
        contract = BimTaskContract(
            goal="List ground floor function types", operation="list",
            entity_concept="functional spaces",
            required_outputs=["function types"], success_criteria=["List function types"],
        )
        bim = SimpleNamespace(ontology=SimpleNamespace(bim_query_knowledge=knowledge))
        root = BimRunContext(
            bim=bim, scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(), question=contract.goal,
            task_contract=contract,
        )
        branch = BimRunContext(
            bim=bim, scope=root.scope, graph_contract=root.graph_contract,
            question=root.question, task_contract=contract,
        )
        package = EvidenceWorkPackage(
            package_id="functions", objective=contract.goal,
            required_outputs=contract.required_outputs,
        )
        registered = RegisteredSchemaMapping(
            mapping_id="mapping-functions",
            proposal=SchemaMappingProposal(
                entity_name="functional_spaces", label="IfcSpace",
                identity_property="GlobalID", source_property="source",
                fields=[
                    SchemaFieldMapping(semantic_name="function_type", property="canonical_type"),
                    SchemaFieldMapping(
                        semantic_name="area_m2", property="canonical_area_m2",
                        data_type="number", unit="m²",
                    ),
                ],
                counting_unit="functional spaces",
                counting_unit_evidence="GlobalID is unique per functional space.",
                reasoning_summary="Live-validated governed mapping.",
            ),
            node_count=10, populated_identity_count=10, distinct_identity_count=10,
        )
        with patch(
            "bim_agents.orchestration.activate_project_knowledge_mapping",
            return_value=registered.model_dump_json(),
        ):
            handoff = trusted_governed_mapping_handoff(root, branch, contract, package)

        self.assertEqual(handoff.operation, "group_summary")
        self.assertEqual(handoff.metric, "area_m2")
        self.assertEqual(handoff.grouping_fields, ["function_type"])
        self.assertNotIn("parking", handoff.evidence_summary)

    def test_typed_mapping_compiles_multiple_outputs_without_query_agent(self):
        contract = BimTaskContract(
            goal="Report function coverage and areas",
            operation="list",
            entity_concept="functional spaces",
            required_outputs=["function assignment coverage", "function types and areas"],
            output_specs=[
                OutputSpec(key="function_assignment_coverage", kind="coverage"),
                OutputSpec(key="function_type_area_summary", kind="list"),
            ],
            success_criteria=["Both outputs are replayable"],
        )
        mapping = RegisteredSchemaMapping(
            mapping_id="mapping-functions",
            proposal=SchemaMappingProposal(
                entity_name="functional_spaces", label="IfcSpace",
                identity_property="GlobalID", source_property="source",
                fields=[
                    SchemaFieldMapping(semantic_name="function_type", property="canonical_type"),
                    SchemaFieldMapping(
                        semantic_name="area_m2", property="canonical_area_m2",
                        data_type="number", unit="m²",
                    ),
                ],
                value_bindings=[],
                counting_unit="functional spaces",
                counting_unit_evidence="GlobalID is unique.",
                reasoning_summary="Governed mapping.",
            ),
            node_count=10, populated_identity_count=10, distinct_identity_count=10,
        )
        branch = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(), task_contract=contract,
            workstream_id="functions",
            schema_mappings={mapping.mapping_id: mapping},
        )
        package = EvidenceWorkPackage(
            package_id="functions", objective=contract.goal,
            required_outputs=contract.required_outputs,
            output_specs=contract.output_specs,
        )
        handoff = EvidenceHandoff(
            package_id="functions", status="ready_for_query", route="live_mapping",
            entity="functional_spaces", mapping_id=mapping.mapping_id,
            operation="group_summary", metric="area_m2",
            grouping_fields=["function_type"],
        )

        plans = _deterministic_mapping_plans(branch, handoff, package)

        self.assertEqual([plan.operation for plan in plans], ["coverage", "group_summary"])
        self.assertEqual(plans[0].coverage_field, "function_type")
        self.assertEqual(plans[1].group_by, "function_type")
        self.assertEqual(plans[1].metric, "area_m2")
        self.assertEqual(plans[0].satisfies, ["function assignment coverage"])
        self.assertEqual(plans[0].answer_key, "functions:function_assignment_coverage")

    def test_governed_geometry_compiles_without_query_agent_when_kinds_fit(self):
        knowledge = {"inventory": {
            "calculation": "broad_switch_inventory",
            "output_kinds": ["list", "count", "grouped_summary"],
        }}
        contract = BimTaskContract(
            goal="Report all switch types", operation="list",
            entity_concept="switches",
            required_outputs=["switch inventory"],
            output_specs=[OutputSpec(key="switch inventory", kind="list")],
            success_criteria=["Governed inventory"],
        )
        branch = BimRunContext(
            bim=SimpleNamespace(ontology=SimpleNamespace(bim_query_knowledge=knowledge)),
            scope=ProjectScope(client_id="c", project_id="p"),
            graph_contract=load_graph_contract(), task_contract=contract,
        )
        package = EvidenceWorkPackage(
            package_id="switches", objective=contract.goal,
            required_outputs=contract.required_outputs,
            output_specs=contract.output_specs,
        )
        handoff = EvidenceHandoff(
            package_id="switches", status="ready_for_query", route="geometry",
            calculation="broad_switch_inventory",
        )

        plan = _deterministic_geometry_plan(branch, handoff, package)

        self.assertEqual(plan.calculation, "broad_switch_inventory")
        self.assertEqual(plan.satisfies, ["switch inventory"])

    def test_governed_floor_area_route_precedes_schema_scout(self):
        knowledge = {
            "area_plan": {
                "entity_concept": "residential floor plates",
                "aliases": ["floor area"],
                "retrieval_terms": ["sixth floor area"],
                "exact_family_values": ["residential_zone"],
            }
        }
        root, branch, contract, package = self._routing_context(
            knowledge,
            "What is the sixth floor area?",
            entity_concept="floor area",
        )
        registered = self._governed_mapping(entity="residential_floor_plates")

        with patch(
            "bim_agents.orchestration.activate_project_knowledge_mapping",
            return_value=registered.model_dump_json(),
        ) as activate:
            handoff = trusted_governed_mapping_handoff(
                root, branch, contract, package,
            )

        activate.assert_called_once()
        self.assertEqual(activate.call_args.args[1], "area_plan")
        self.assertEqual(handoff.route, "live_mapping")

    def test_governed_socket_multilingual_alias_routes_without_scout(self):
        knowledge = {
            "socket_box": {
                "entity_concept": "socket box for 6 modules",
                "aliases": ["קופסת שקעים ל-6 מודולים"],
                "exact_family_values": ["D-20"],
            }
        }
        question = "מה האורך עבור קופסת שקעים ל-6 מודולים?"
        root, branch, contract, package = self._routing_context(knowledge, question)
        registered = self._governed_mapping(entity="socket_box_for_6_modules")

        with patch(
            "bim_agents.orchestration.activate_project_knowledge_mapping",
            return_value=registered.model_dump_json(),
        ) as activate:
            handoff = trusted_governed_mapping_handoff(
                root, branch, contract, package,
            )

        activate.assert_called_once()
        self.assertEqual(activate.call_args.args[1], "socket_box")
        self.assertEqual(handoff.mapping_id, registered.mapping_id)

    def test_governed_mapping_route_fails_closed_on_ambiguity(self):
        knowledge = {
            "lighting_switches": {
                "entity_concept": "lighting switches", "aliases": ["switches"],
                "exact_family_values": ["lighting"],
            },
            "door_switches": {
                "entity_concept": "door switches", "aliases": ["switches"],
                "exact_family_values": ["door"],
            },
        }
        root, branch, contract, package = self._routing_context(
            knowledge, "Count switches", entity_concept="switches",
        )

        with patch(
            "bim_agents.orchestration.activate_project_knowledge_mapping"
        ) as activate:
            handoff = trusted_governed_mapping_handoff(
                root, branch, contract, package,
            )

        self.assertIsNone(handoff)
        activate.assert_not_called()

    async def test_governed_route_precedes_schema_scout_in_workstream(self):
        knowledge = {
            "conduits": {
                "entity_concept": "electrical conduits", "aliases": ["conduits"],
                "exact_family_values": ["HDPE"],
            }
        }
        root, _, contract, _ = self._routing_context(
            knowledge, "What is the total length of conduits?", entity_concept="conduits",
        )
        contract.work_packages = [EvidenceWorkPackage(
            package_id="conduit", objective="Total conduit length",
            required_outputs=["total length"],
        )]
        root.task_contract = contract
        settings = Settings("bolt://test", "u", "p", "neo4j", "c", "p")
        registered = self._governed_mapping(entity="electrical_conduits")
        called_agents = []

        async def fake_run(agent, input_text, *, context, **kwargs):
            called_agents.append(agent.name)
            evidence_id = "query-conduit"
            context.add_evidence(Evidence(
                evidence_id=evidence_id, kind="query", summary="total length",
                payload=json.dumps({
                    "plan": {
                        "role": "answer_producing", "include_in_answer": True,
                        "satisfies": ["total length"], "answer_key": "conduit-length",
                    },
                    "claim": {"value": 8, "unit": "m"},
                }),
            ))
            return SimpleNamespace(final_output=EvidenceWorkstreamResult(
                status="query_completed", package_id="conduit",
                evidence_ids=[evidence_id],
            ))

        with patch("bim_agents.orchestration.BimContext", return_value=FakeBimContext()), patch(
            "bim_agents.orchestration.activate_project_knowledge_mapping",
            return_value=registered.model_dump_json(),
        ), patch("bim_agents.orchestration.Runner.run", side_effect=fake_run):
            outcomes = await run_evidence_workstreams(
                settings=settings, root=root, registry=build_agent_registry(),
                events=PipelineEvents(),
                merge=lambda authoritative, branch: _merge_isolated_workstream(
                    authoritative, branch, workstream_id=branch.workstream_id,
                ),
            )

        self.assertEqual(called_agents, ["BIM Quantity Specialist"])
        self.assertEqual(outcomes[0].status, "query_completed")

    async def test_specialists_receive_fresh_package_local_contexts_and_compact_inputs(self):
        contract = self.contract()
        root = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(),
            question="ROOT_GLOBAL_SENTINEL Count and classify switches",
            task_contract=contract,
        )
        settings = Settings("bolt://test", "u", "p", "neo4j", "c", "p")
        registry = build_agent_registry()
        instances = [FakeBimContext(), FakeBimContext()]
        seen_contexts = {}
        seen_discovery = {}
        seen_context_objects = []
        seen_inputs = {}

        async def fake_run(agent, input_text, *, context, **kwargs):
            seen_context_objects.append(context)
            seen_contexts.setdefault(context.workstream_id, []).append(id(context))
            seen_discovery.setdefault(context.workstream_id, []).append(dict(context.schema_discovery))
            seen_inputs.setdefault(context.workstream_id, []).append(input_text)
            if agent.name == "BIM Schema Mapping Specialist":
                context.schema_discovery["raw_transcript"] = "RAW_MAPPING_TRANSCRIPT_SENTINEL"
                return SimpleNamespace(final_output=EvidenceHandoff(
                    package_id=context.workstream_id,
                    status="ready_for_query", route="contract",
                    entity="elements", operation="count",
                    evidence_summary="Trusted contract route.",
                ))
            evidence_id = "query-" + context.workstream_id
            output = context.task_contract.required_outputs[0]
            context.add_evidence(Evidence(
                evidence_id=evidence_id, kind="query", summary=output,
                payload=json.dumps({
                    "plan": {
                        "role": "answer_producing", "include_in_answer": True,
                        "satisfies": [output], "answer_key": context.workstream_id,
                    },
                    "claim": {"value": 1},
                }),
            ))
            return SimpleNamespace(final_output=EvidenceWorkstreamResult(
                status="query_completed", package_id=context.workstream_id,
                evidence_ids=[evidence_id],
            ))

        with patch("bim_agents.orchestration.BimContext", side_effect=instances), patch(
            "bim_agents.orchestration.Runner.run", side_effect=fake_run
        ):
            outcomes = await run_evidence_workstreams(
                settings=settings, root=root, registry=registry,
                events=PipelineEvents(),
                merge=lambda authoritative, branch: _merge_isolated_workstream(
                    authoritative, branch, workstream_id=branch.workstream_id
                ),
            )

        self.assertEqual({item.status for item in outcomes}, {"query_completed"})
        self.assertEqual(set(root.evidence), {"query-count", "query-types"})
        self.assertTrue(all(len(set(values)) == 2 for values in seen_contexts.values()))
        self.assertNotEqual(seen_contexts["count"][0], seen_contexts["types"][0])
        self.assertTrue(all(item.connected and item.closed for item in instances))
        for input_text in seen_inputs["count"]:
            self.assertNotIn("ROOT_GLOBAL_SENTINEL", input_text)
            self.assertNotIn("switch types", input_text)
        for input_text in seen_inputs["types"]:
            self.assertNotIn("ROOT_GLOBAL_SENTINEL", input_text)
            self.assertNotIn("switch count", input_text)
        self.assertTrue(all(
            "raw_transcript" not in snapshot
            for snapshots in seen_discovery.values()
            for snapshot in snapshots[1:]
        ))
        self.assertEqual(root.evidence["query-count"].workstream_id, "count")

    async def test_invalid_typed_result_recovers_complete_durable_evidence_without_retry(self):
        package = EvidenceWorkPackage(
            package_id="count", objective="Count switches", required_outputs=["switch count"],
        )
        root = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(), question="Count switches",
            task_contract=BimTaskContract(
                goal="Count switches", operation="count", entity_concept="switches",
                required_outputs=["switch count"], work_packages=[package],
                success_criteria=["Produce a replayable switch count."],
            ),
        )
        settings = Settings("bolt://test", "u", "p", "neo4j", "c", "p")
        calls = []
        captured_events = []

        async def fake_run(agent, input_text, *, context, **kwargs):
            calls.append(agent.name)
            if agent.name == "BIM Schema Mapping Specialist":
                return SimpleNamespace(final_output=EvidenceHandoff(
                    package_id="count", status="ready_for_query", route="contract",
                    entity="elements", operation="count", evidence_summary="Mapped.",
                ))
            context.add_evidence(Evidence(
                evidence_id="query-count", kind="query", summary="21 switches",
                payload=json.dumps({
                    "plan": {
                        "role": "answer_producing", "include_in_answer": True,
                        "satisfies": ["switch count"], "answer_key": "count",
                    },
                    "claim": {"value": 21},
                }),
            ))
            return SimpleNamespace(final_output={"status": "query_completed"})

        with patch("bim_agents.orchestration.BimContext", return_value=FakeBimContext()), patch(
            "bim_agents.orchestration.Runner.run", side_effect=fake_run,
        ):
            outcomes = await run_evidence_workstreams(
                settings=settings, root=root, registry=build_agent_registry(),
                events=PipelineEvents(event_sink=captured_events.append),
                merge=lambda authoritative, branch: _merge_isolated_workstream(
                    authoritative, branch, workstream_id=branch.workstream_id,
                ),
            )

        self.assertEqual(calls.count("BIM Quantity Specialist"), 1)
        self.assertEqual(outcomes[0].status, "query_completed")
        self.assertEqual(outcomes[0].typed_output_failures, 1)
        self.assertEqual(
            outcomes[0].recovery_strategy, "deterministic_evidence_checkpoint",
        )
        self.assertIn("query-count", root.evidence)
        self.assertTrue(any(
            event.get("stage") == "workstream_typed_output_recovered"
            for event in captured_events
        ))

    async def test_typed_output_retry_keeps_first_attempt_evidence_and_completes_missing_output(self):
        package = EvidenceWorkPackage(
            package_id="inventory", objective="Inventory switches",
            required_outputs=["switch count", "switch types"],
        )
        root = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(), question="Inventory switches",
            task_contract=BimTaskContract(
                goal="Inventory switches", operation="group_count", entity_concept="switches",
                required_outputs=list(package.required_outputs), work_packages=[package],
                success_criteria=["Produce replayable switch count and type evidence."],
            ),
        )
        settings = Settings("bolt://test", "u", "p", "neo4j", "c", "p")
        specialist_calls = 0

        async def fake_run(agent, input_text, *, context, **kwargs):
            nonlocal specialist_calls
            if agent.name == "BIM Schema Mapping Specialist":
                return SimpleNamespace(final_output=EvidenceHandoff(
                    package_id="inventory", status="ready_for_query", route="contract",
                    entity="elements", operation="group_count", evidence_summary="Mapped.",
                ))
            specialist_calls += 1
            if specialist_calls == 1:
                context.add_evidence(Evidence(
                    evidence_id="query-count", kind="query", summary="count",
                    payload=json.dumps({"plan": {
                        "role": "answer_producing", "satisfies": ["switch count"],
                        "answer_key": "count",
                    }}),
                ))
                return SimpleNamespace(final_output={"status": "not-a-valid-status"})
            self.assertIn('"missing_outputs": ["switch types"]', input_text)
            context.add_evidence(Evidence(
                evidence_id="query-types", kind="query", summary="types",
                payload=json.dumps({"plan": {
                    "role": "answer_producing", "satisfies": ["switch types"],
                    "answer_key": "types",
                }}),
            ))
            return SimpleNamespace(final_output=EvidenceWorkstreamResult(
                status="query_completed", package_id="inventory",
                evidence_ids=["query-count", "query-types"],
            ))

        with patch("bim_agents.orchestration.BimContext", return_value=FakeBimContext()), patch(
            "bim_agents.orchestration.Runner.run", side_effect=fake_run,
        ):
            outcomes = await run_evidence_workstreams(
                settings=settings, root=root, registry=build_agent_registry(),
                events=PipelineEvents(),
                merge=lambda authoritative, branch: _merge_isolated_workstream(
                    authoritative, branch, workstream_id=branch.workstream_id,
                ),
            )

        self.assertEqual(specialist_calls, 2)
        self.assertEqual(outcomes[0].status, "query_completed")
        self.assertEqual(outcomes[0].specialist_attempts, 2)
        self.assertEqual(outcomes[0].typed_output_failures, 1)
        self.assertEqual(set(root.evidence), {"query-count", "query-types"})

    async def test_exhausted_typed_repair_retains_partial_evidence_without_unblocking_package(self):
        package = EvidenceWorkPackage(
            package_id="inventory", objective="Inventory switches",
            required_outputs=["switch count", "switch types"],
        )
        root = BimRunContext(
            bim=object(),
            scope=ProjectScope(client_id="c", project_id="p", allowed_sources=["model.ifc"]),
            graph_contract=load_graph_contract(), question="Inventory switches",
            task_contract=BimTaskContract(
                goal="Inventory switches", operation="group_count", entity_concept="switches",
                required_outputs=list(package.required_outputs), work_packages=[package],
                success_criteria=["Produce replayable switch count and type evidence."],
            ),
        )
        settings = Settings("bolt://test", "u", "p", "neo4j", "c", "p")
        specialist_calls = 0

        async def fake_run(agent, input_text, *, context, **kwargs):
            nonlocal specialist_calls
            if agent.name == "BIM Schema Mapping Specialist":
                return SimpleNamespace(final_output=EvidenceHandoff(
                    package_id="inventory", status="ready_for_query", route="contract",
                    entity="elements", operation="group_count", evidence_summary="Mapped.",
                ))
            specialist_calls += 1
            if specialist_calls == 1:
                context.add_evidence(Evidence(
                    evidence_id="query-count", kind="query", summary="count",
                    payload=json.dumps({"plan": {
                        "role": "answer_producing", "satisfies": ["switch count"],
                        "answer_key": "count",
                    }}),
                ))
            return SimpleNamespace(final_output={"status": "invalid"})

        with patch("bim_agents.orchestration.BimContext", return_value=FakeBimContext()), patch(
            "bim_agents.orchestration.Runner.run", side_effect=fake_run,
        ):
            outcomes = await run_evidence_workstreams(
                settings=settings, root=root, registry=build_agent_registry(),
                events=PipelineEvents(),
                merge=lambda authoritative, branch: _merge_isolated_workstream(
                    authoritative, branch, workstream_id=branch.workstream_id,
                ),
            )

        self.assertEqual(specialist_calls, 2)
        self.assertEqual(outcomes[0].status, "partial_completed")
        self.assertEqual(outcomes[0].satisfied_outputs, ["switch count"])
        self.assertIn("switch types", outcomes[0].limitations[0])
        self.assertIn("query-count", root.evidence)


if __name__ == "__main__":
    unittest.main()
