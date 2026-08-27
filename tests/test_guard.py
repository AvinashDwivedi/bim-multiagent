import pytest

from bim_agents.graph import CypherGuard, UnsafeCypherError


def test_accepts_scoped_read_query():
    query = """
    MATCH (n:BIMElement {client_id: $client_id, project_id: $project_id})
    WHERE n.canonical_type = $kind
    RETURN count(n) AS count
    """
    assert CypherGuard().validate(query).strip().startswith("MATCH")


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n:BIMElement {client_id:$client_id, project_id:$project_id}) DELETE n RETURN n",
        "MATCH (n:BIMElement) RETURN n",
        "CALL dbms.components() YIELD name RETURN name, $client_id, $project_id",
        "MATCH (n:BIMElement {client_id:$client_id, project_id:$project_id}) RETURN n; MATCH (m) RETURN m",
        "MATCH (n) WITH n, $client_id AS client_id, $project_id AS project_id RETURN n",
    ],
)
def test_rejects_unsafe_queries(query):
    with pytest.raises(UnsafeCypherError):
        CypherGuard().validate(query)


def test_forbidden_word_inside_parameter_value_is_not_part_of_query():
    query = """
    MATCH (n:BIMElement {client_id: $client_id, project_id: $project_id})
    WHERE n.name = $name RETURN n.id AS id LIMIT 1
    """
    assert CypherGuard().validate(query)
