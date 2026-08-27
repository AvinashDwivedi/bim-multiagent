from __future__ import annotations

from pathlib import Path

from bim_agent.dataset import ProjectDataset
from bim_agent.executor import DeterministicExecutor
from bim_agent.ifc_graph import IfcGraph
from bim_agent.models import PropertyFilter, QueryBranch, QueryPlan
from bim_agent.planner import apply_domain_guardrails
from bim_agent.planner import SemanticPlanner
from bim_agent.verifier import EvidenceVerifier


def test_grouped_measurement_and_missing_property_projection(sample_data: Path) -> None:
    dataset = ProjectDataset(sample_data)
    plan = QueryPlan(
        intent="sum",
        categories=["Pipes"],
        group_by=["type_name"],
        measure_property="Dimensions.Length",
        select_properties=["Materials and Finishes.Material"],
        output_unit="m",
        required_facets=["population", "measure", "grouping", "property_values"],
    )

    evidence = DeterministicExecutor(dataset).execute(plan)

    assert evidence.operation_value == 5.5
    assert evidence.grouped_measurements[0]["sum"] == 5.5
    assert evidence.property_summaries[0]["missing"] == 2
    assert EvidenceVerifier(dataset).verify(plan, evidence, "length by type and material").status == "verified"


def test_population_branches_form_a_deduplicated_union(sample_data: Path) -> None:
    dataset = ProjectDataset(sample_data)
    plan = QueryPlan(population_branches=[
        QueryBranch(label="switches", categories=["Lighting Devices"]),
        QueryBranch(label="door switch", categories=["Electrical Fixtures"]),
    ])

    evidence = DeterministicExecutor(dataset).execute(plan)

    assert evidence.distinct_identity_count == 4
    assert {item["category"] for item in evidence.category_counts} == {
        "Lighting Devices", "Electrical Fixtures"
    }


def test_specific_description_requires_an_executable_constraint(sample_data: Path) -> None:
    dataset = ProjectDataset(sample_data)
    plan = QueryPlan(categories=["Electrical Fixtures"], select_properties=["Constraints.Level"])
    evidence = DeterministicExecutor(dataset).execute(plan)

    verification = EvidenceVerifier(dataset).verify(
        plan, evidence, "what is the height of the specific description?"
    )

    assert verification.status == "limited"
    assert not next(
        check for check in verification.checks if check["name"] == "specific_scope_constraint"
    )["passed"]


def test_specific_multilingual_concept_removes_conflicting_identity_filters() -> None:
    profile = {
        "categories": [{"category": "Electrical Fixtures", "family_types": []}],
        "property_keys": ["Identity Data.Description", "Constraints.Default Elevation"],
        "property_stats": [],
    }
    plan = QueryPlan(
        categories=["Electrical Fixtures"],
        families=["EF_D17"],
        types=["4h+2t"],
        filters=[PropertyFilter("Identity Data.Description", "contains", "Socket box for 6 modules")],
        match_terms=["Socket box for 6 modules"],
    )

    guarded = apply_domain_guardrails(
        "מה מרחק שקע מסוג Socket box for 6 modules, מהרצפה?", profile, plan
    )

    assert guarded.categories == ["Electrical Fixtures"]
    assert guarded.families == []
    assert guarded.types == []
    assert guarded.filters == []
    assert guarded.match_terms[0] == "Socket box for 6 modules"
    assert "IFC.Placement Height Above Storey" in guarded.select_properties


def test_real_project_specific_socket_population_is_cross_language() -> None:
    data_dir = Path(__file__).parents[1] / "test-project-data"
    dataset = ProjectDataset(data_dir)
    question = "מה מרחק שקע מסוג Socket box for 6 modules, מהרצפה?"
    profile = dataset.profile(focus_terms=[question])
    plan = QueryPlan(
        categories=["Electrical Fixtures"],
        families=["EF_D17"],
        filters=[PropertyFilter("Identity Data.Description", "contains", "Socket box for 6 modules")],
        match_terms=["Socket box for 6 modules"],
    )
    plan = apply_domain_guardrails(question, profile, plan)

    evidence = DeterministicExecutor(dataset).execute(plan)

    assert evidence.distinct_identity_count == 61
    intended = next(
        item for item in evidence.property_summaries
        if item["field"] == "BIM.Intended Height From Description"
    )
    assert {item["value"]: item["count"] for item in intended["values"]} == {
        "145 cm": 44, "40 cm": 17,
    }
    placement = next(
        item for item in evidence.property_summaries
        if item["field"] == "IFC.Placement Height Above Storey"
    )
    assert placement["present"] == 61
    assert EvidenceVerifier(dataset).verify(plan, evidence, question).status == "verified"


def test_switch_questions_require_related_scope_audit() -> None:
    profile = {
        "categories": [
            {"category": "Lighting Devices", "family_types": [{"family": "A Switch", "type": "Single"}]},
            {"category": "Electrical Equipment", "family_types": [{"family": "Switchboard-Siemens", "type": "Panel"}]},
        ],
        "property_keys": [],
        "property_stats": [],
    }
    plan = apply_domain_guardrails(
        "אילו סוגי מפסקים בפרוייקט?", profile,
        QueryPlan(categories=["Lighting Devices"], include_related=False),
    )

    assert plan.include_related
    assert "related_scope" in plan.required_facets
    assert plan.group_by == ["family", "type_name"]
    assert "Switchboard-Siemens" in plan.search_terms


def test_real_project_connectivity_separates_logical_and_physical_failures() -> None:
    data_dir = Path(__file__).parents[1] / "test-project-data"
    dataset = ProjectDataset(data_dir)
    question = "האם ישנם רכיבי חשמל המתוכננים ללא חיבור/המשכיות אל לוח החשמל המזין?"
    plan = SemanticPlanner(model="unused", reasoning_effort="low", use_llm=False).plan(
        question, dataset.profile(focus_terms=[question])
    )

    evidence = DeterministicExecutor(dataset).execute(plan)
    paths = evidence.connectivity["ifc"]["panel_path_analysis"]

    assert evidence.distinct_identity_count == 488
    assert paths["logical_missing_panel"] == 305
    assert paths["assigned_panel_resolved"] == 183
    assert paths["assigned_panel_unresolved"] == 0
    assert paths["physical_path_to_any_panel"] == 0
    assert paths["no_path_to_assigned_panel"] == 183
    assert paths["both_logical_and_physical_failure"] == 305
    assert EvidenceVerifier(dataset).verify(plan, evidence, question).status == "verified"


def test_generic_unseen_question_capabilities() -> None:
    dataset = ProjectDataset(Path(__file__).parents[1] / "test-project-data")
    planner = SemanticPlanner(model="unused", reasoning_effort="low", use_llm=False)
    executor = DeterministicExecutor(dataset)
    verifier = EvidenceVerifier(dataset)

    def run(question: str):
        plan = planner.plan(question, dataset.profile(focus_terms=[question]))
        evidence = executor.execute(plan)
        return plan, evidence, verifier.verify(plan, evidence, question)

    average_plan, average, average_check = run("What is the average length of cable trays?")
    assert average_plan.calculation == "average"
    assert average_plan.measure_property == "Dimensions.Length"
    assert round(float(average.operation_value), 6) == round(556.98472 / 106, 6)
    assert average_check.status == "verified"

    top_plan, top, top_check = run("Which level has the most lighting fixtures?")
    assert top_plan.group_by == ["level"]
    assert top_plan.sort_by == "count" and top_plan.limit == 1
    assert top.groups == [{"level": "GF", "count": 49}]
    assert top_check.status == "verified"

    threshold_plan, threshold, threshold_check = run("How many cable trays are longer than 5 m?")
    assert any(item.operator == "gt" and item.value == "5 m" for item in threshold_plan.filters)
    assert threshold.distinct_identity_count == 53
    assert threshold_check.status == "verified"

    compare_plan, compare, compare_check = run("Compare the number of sockets on GF and B1.")
    level_filter = next(item for item in compare_plan.filters if item.field == "level")
    assert level_filter.operator == "in" and level_filter.value == ["GF", "B1"]
    assert compare_plan.group_by == ["level"]
    assert {item["level"] for item in compare.groups} == {"GF", "B1"}
    assert compare_check.status == "verified"

    lookup_plan, lookup, lookup_check = run("Where is the element with Mark 33?")
    assert any("Mark" in item.field and item.value == "33" for item in lookup_plan.filters)
    assert lookup.distinct_identity_count == 3
    assert lookup_check.status == "verified"

    duplicate_plan, duplicate, duplicate_check = run("Are there duplicate Mark values in the project?")
    assert duplicate_plan.minimum_group_count == 2
    assert duplicate.groups and all(item["count"] >= 2 for item in duplicate.groups)
    assert duplicate_check.status == "verified"

    unsupported_plan, _, unsupported_check = run("Why are the sockets located there?")
    assert unsupported_plan.unsupported_requirements
    assert unsupported_check.status == "limited"

    dropped_average = QueryPlan(categories=["Cable Trays"], calculation="count")
    dropped_average_evidence = executor.execute(dropped_average)
    dropped_average_check = verifier.verify(
        dropped_average, dropped_average_evidence, "What is the average length of cable trays?"
    )
    assert dropped_average_check.status == "limited"
    assert not next(
        item for item in dropped_average_check.checks if item["name"] == "average_requirement"
    )["passed"]

    dropped_threshold = QueryPlan(categories=["Cable Trays"])
    dropped_threshold_evidence = executor.execute(dropped_threshold)
    dropped_threshold_check = verifier.verify(
        dropped_threshold, dropped_threshold_evidence, "How many cable trays are longer than 5 m?"
    )
    assert dropped_threshold_check.status == "limited"
    assert not next(
        item for item in dropped_threshold_check.checks if item["name"] == "threshold_requirement"
    )["passed"]


def test_step_ifc_graph_extracts_containment_ports_and_systems(tmp_path: Path) -> None:
    path = tmp_path / "model.ifc"
    path.write_text(
        """ISO-10303-21;
HEADER;
ENDSEC;
DATA;
#1=IFCBUILDINGSTOREY('0000000000000000000001',$,'GF',$,$,$,$,$,$,$);
#2=IFCFLOWTERMINAL('0000000000000000000002',$,'Socket',$,$,$,$,$,$);
#3=IFCDISTRIBUTIONPORT('0000000000000000000003',$,$,$,$,$,$,$,$,$);
#4=IFCDISTRIBUTIONPORT('0000000000000000000004',$,$,$,$,$,$,$,$,$);
#5=IFCRELCONTAINEDINSPATIALSTRUCTURE('0000000000000000000005',$,$,$,(#2),#1);
#6=IFCRELCONNECTSPORTTOELEMENT('0000000000000000000006',$,$,$,#3,#2);
#7=IFCRELCONNECTSPORTTOELEMENT('0000000000000000000007',$,$,$,#4,#2);
#8=IFCRELCONNECTSPORTS('0000000000000000000008',$,$,$,#3,#4,$);
#9=IFCSYSTEM('0000000000000000000009',$,'Circuit A',$,$);
#10=IFCRELASSIGNSTOGROUP('0000000000000000000010',$,$,$,(#2),$,#9);
ENDSEC;
END-ISO-10303-21;
""",
        encoding="utf-8",
    )

    graph = IfcGraph.load(path)

    assert graph is not None
    assert graph.storey_for_guid("0000000000000000000002") == "GF"
    assert graph.systems_for_guid("0000000000000000000002") == ["Circuit A"]
    assert graph.connectivity_summary()["port_connections"] == 1
