from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping
from typing import Any

from neo4j import GraphDatabase, Query, READ_ACCESS

from ..config import Settings
from ..contracts import EvidenceArtifact
from ..evidence import EvidenceStore
from .guard import CypherGuard


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (int, bool)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 20_000 else value[:20_000] + "…[truncated]"
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        output = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 250:
                output["__truncated_properties__"] = len(value) - 250
                break
            name = str(key)
            if "embedding" in name.casefold():
                output[name] = f"[vector omitted: {len(item) if hasattr(item, '__len__') else 'unknown'} values]"
            else:
                output[name] = _json_value(item)
        return output
    if isinstance(value, (list, tuple, set)):
        sequence = list(value)
        output = [_json_value(item) for item in sequence[:500]]
        if len(sequence) > 500:
            output.append(f"[truncated {len(sequence) - 500} items]")
        return output
    if hasattr(value, "items"):
        return {str(key): _json_value(item) for key, item in value.items()}
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _lucene_query(value: str) -> str:
    """Turn user/model text into a safe literal token query for Neo4j full-text indexes."""
    cleaned = re.sub(r"[+\-!(){}\[\]^\"~*?:\\/]|&&|\|\|", " ", value)
    tokens = [token for token in cleaned.split() if token]
    if not tokens:
        raise ValueError("Search text does not contain searchable literal tokens.")
    return " ".join(tokens[:24])


class BIMGraph:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        self.guard = CypherGuard()
        self._schema_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}

    def close(self) -> None:
        self.driver.close()

    def verify_connectivity(self) -> None:
        self.driver.verify_connectivity()

    def _run(self, query: str, parameters: dict[str, Any] | None = None) -> tuple[list[str], list[dict[str, Any]], float]:
        started = time.perf_counter()
        with self.driver.session(
            database=self.settings.neo4j_database,
            default_access_mode=READ_ACCESS,
        ) as session:
            result = session.run(
                Query(query, timeout=self.settings.query_timeout_seconds),
                parameters or {},
            )
            columns = list(result.keys())
            records = result.fetch(self.settings.max_result_rows + 1)
            result.consume()
        elapsed_ms = (time.perf_counter() - started) * 1000
        return columns, [_json_value(record.data()) for record in records], elapsed_ms

    def scope_summary(self, client_id: str, project_id: str) -> dict[str, int]:
        _, rows, _ = self._run(
            """
            MATCH (n:BIMElement {client_id: $client_id, project_id: $project_id})
            RETURN count(n) AS element_count,
                   count(DISTINCT n.source) AS source_count
            """,
            {"client_id": client_id, "project_id": project_id},
        )
        return rows[0] if rows else {"element_count": 0, "source_count": 0}

    def schema_snapshot(self, client_id: str, project_id: str) -> dict[str, Any]:
        cache_key = (client_id, project_id)
        cached = self._schema_cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < 300:
            return cached[1]

        params = {"client_id": client_id, "project_id": project_id}
        _, labels, _ = self._run(
            """
            MATCH (n:BIMElement {client_id: $client_id, project_id: $project_id})
            UNWIND labels(n) AS label
            RETURN label, count(*) AS count ORDER BY count DESC LIMIT 80
            """,
            params,
        )
        _, relationships, _ = self._run(
            """
            MATCH (a:BIMElement {client_id: $client_id, project_id: $project_id})-[r]-(b)
            RETURN type(r) AS relationship, count(*) AS count
            ORDER BY count DESC LIMIT 50
            """,
            params,
        )
        _, directed_relationships, _ = self._run(
            """
            MATCH (a:BIMElement {client_id: $client_id, project_id: $project_id})-[r]->
                  (b:BIMElement {client_id: $client_id, project_id: $project_id})
            RETURN type(r) AS relationship, labels(a) AS start_labels,
                   labels(b) AS end_labels, count(*) AS count
            ORDER BY count DESC LIMIT 100
            """,
            params,
        )
        _, properties, _ = self._run(
            """
            MATCH (n:BIMElement {client_id: $client_id, project_id: $project_id})
            UNWIND keys(n) AS property
            WITH property, count(*) AS populated
            RETURN property, populated ORDER BY populated DESC LIMIT 140
            """,
            params,
        )
        _, semantic_properties, _ = self._run(
            """
            MATCH (n:BIMElement {client_id: $client_id, project_id: $project_id})
            UNWIND keys(n) AS property
            WITH property, count(*) AS populated
            WHERE any(token IN ['area','volume','length','height','width','elevation','level',
                                 'floor','material','type','category','system','unit']
                      WHERE toLower(property) CONTAINS token)
            RETURN property, populated ORDER BY populated DESC LIMIT 180
            """,
            params,
        )
        _, identity_profile, _ = self._run(
            """
            MATCH (n:BIMElement {client_id: $client_id, project_id: $project_id})
            RETURN count(n) AS records,
                   count(DISTINCT n.GlobalID) AS distinct_global_ids,
                   count(DISTINCT n.object_id) AS distinct_object_ids,
                   count(CASE WHEN n.GlobalID IS NULL OR trim(toString(n.GlobalID)) = '' THEN 1 END)
                       AS missing_global_ids,
                   count(CASE WHEN n.object_id IS NULL OR trim(toString(n.object_id)) = '' THEN 1 END)
                       AS missing_object_ids
            """,
            params,
        )
        _, source_profile, _ = self._run(
            """
            MATCH (n:BIMElement {client_id: $client_id, project_id: $project_id})
            RETURN coalesce(toString(n.source), '<missing>') AS source, count(*) AS records,
                   count(DISTINCT n.GlobalID) AS distinct_global_ids
            ORDER BY records DESC LIMIT 40
            """,
            params,
        )
        semantic_names = [row["property"] for row in semantic_properties[:60]]
        _, semantic_value_profile, _ = self._run(
            """
            UNWIND $properties AS property
            MATCH (n:BIMElement {client_id: $client_id, project_id: $project_id})
            WHERE n[property] IS NOT NULL AND trim(toString(n[property])) <> ''
            WITH property, toString(n[property]) AS value, count(*) AS records
            ORDER BY property, records DESC
            WITH property, sum(records) AS populated,
                 count(*) AS distinct_values,
                 collect({value: value, records: records})[0..8] AS examples
            RETURN property, populated, distinct_values, examples
            ORDER BY populated DESC LIMIT 60
            """,
            {**params, "properties": semantic_names},
        )
        _, source_overlap, _ = self._run(
            """
            MATCH (n:BIMElement {client_id: $client_id, project_id: $project_id})
            WHERE n.GlobalID IS NOT NULL AND trim(toString(n.GlobalID)) <> ''
            WITH n.GlobalID AS global_id, collect(DISTINCT n.source) AS sources, count(*) AS records
            WHERE size(sources) > 1 OR records > 1
            RETURN count(*) AS overlapping_global_ids,
                   sum(records) AS overlapping_records,
                   collect({global_id: global_id, sources: sources, records: records})[0..12] AS examples
            """,
            params,
        )
        _, related_labels, _ = self._run(
            """
            MATCH (p:Project {id: $project_id, client_id: $client_id})
            MATCH (p)-[*1..5]->(n)
            UNWIND labels(n) AS label
            RETURN label, count(DISTINCT n) AS count
            ORDER BY count DESC LIMIT 50
            """,
            params,
        )
        summary = self.scope_summary(client_id, project_id)
        snapshot = {
            **summary,
            "labels": labels,
            "relationships": relationships,
            "directed_relationship_endpoints": directed_relationships,
            "common_properties": properties,
            "semantic_properties": semantic_properties,
            "identity_profile": identity_profile[0] if identity_profile else {},
            "source_profile": source_profile,
            "source_overlap": source_overlap[0] if source_overlap else {},
            "semantic_value_profile": semantic_value_profile,
            "project_related_labels": related_labels,
            "scope_rule": (
                "Every BIMElement match must constrain client_id=$client_id and "
                "project_id=$project_id. Other nodes must be reached from that scoped population "
                "or from a Project constrained by id=$project_id and client_id=$client_id."
            ),
        }
        self._schema_cache[cache_key] = (time.monotonic(), snapshot)
        return snapshot

    def inventory_documents(
        self, *, client_id: str, project_id: str, store: EvidenceStore, purpose: str
    ) -> EvidenceArtifact:
        query = """
        MATCH (project:Project {id: $project_id, client_id: $client_id})
        OPTIONAL MATCH (project)-[*1..5]->(node)
        WHERE any(label IN labels(node) WHERE toLower(label) CONTAINS 'document'
              OR toLower(label) CONTAINS 'chunk')
        RETURN labels(node) AS labels, node.fileName AS fileName,
               node.source_label AS source_label, count(DISTINCT node) AS record_count
        ORDER BY record_count DESC, fileName
        """
        columns, rows, elapsed = self._run(
            query, {"client_id": client_id, "project_id": project_id}
        )
        return store.add(
            tool="document-inventory", purpose=purpose, query=query.strip(), columns=columns,
            rows=rows, row_count=len(rows), truncated=False, elapsed_ms=elapsed,
            population="All document-like nodes reachable from the authorized Project",
            entity_role="curated_record", spatial_scope="Authorized project",
            measurement_basis="Graph record inventory", aggregation="Distinct records by source",
            unit="records", identity_key="Neo4j node identity",
            inclusion_rules=["Document/chunk labels"], exclusion_rules=[],
        )

    def search_documents(
        self,
        *,
        client_id: str,
        project_id: str,
        search_text: str,
        limit: int,
        store: EvidenceStore,
        purpose: str,
    ) -> EvidenceArtifact:
        safe_limit = min(max(limit, 1), 30)
        query = """
        CALL db.index.fulltext.queryNodes('doc_keyword_enriched', $search_text)
        YIELD node, score
        MATCH (project:Project {id: $project_id, client_id: $client_id})-[*1..5]->(node)
        RETURN DISTINCT node.id AS id, node.chunk_key AS chunk_key,
               node.fileName AS fileName, node.source_label AS source_label,
               node.section_title AS section_title, node.page_start AS page_start,
               node.text AS text, node.contextual_summary AS contextual_summary, score
        ORDER BY score DESC
        LIMIT $limit
        """
        columns, rows, elapsed = self._run(
            query,
            {
                "client_id": client_id,
                "project_id": project_id,
                "search_text": _lucene_query(search_text),
                "limit": safe_limit,
            },
        )
        return store.add(
            tool="document-search",
            purpose=purpose,
            query=query.strip(),
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=False,
            elapsed_ms=elapsed,
        )

    def search_elements(
        self,
        *,
        client_id: str,
        project_id: str,
        search_text: str,
        limit: int,
        store: EvidenceStore,
        purpose: str,
    ) -> EvidenceArtifact:
        safe_limit = min(max(limit, 1), 50)
        query = """
        CALL db.index.fulltext.queryNodes('bim_keyword', $search_text)
        YIELD node, score
        WHERE node.client_id = $client_id AND node.project_id = $project_id
        RETURN node.id AS id, node.GlobalID AS GlobalID, node.object_id AS object_id,
               labels(node) AS labels, node.name AS name, node.elementType AS elementType,
               node.raw_type AS raw_type, node.canonical_type AS canonical_type,
               node.canonical_level AS canonical_level, node.Level AS Level,
               node.source AS source, score,
               [key IN keys(node) WHERE NOT key IN ['embedding', 'embedding_text', 'psets_json']][0..80]
                   AS available_properties
        ORDER BY score DESC
        LIMIT $limit
        """
        columns, rows, elapsed = self._run(
            query,
            {
                "client_id": client_id,
                "project_id": project_id,
                "search_text": _lucene_query(search_text),
                "limit": safe_limit,
            },
        )
        return store.add(
            tool="element-search",
            purpose=purpose,
            query=query.strip(),
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=False,
            elapsed_ms=elapsed,
        )

    def profile_properties(
        self,
        *,
        client_id: str,
        project_id: str,
        properties: list[str],
        store: EvidenceStore,
        purpose: str,
    ) -> EvidenceArtifact:
        selected = [item for item in dict.fromkeys(properties) if item][:12]
        if not selected:
            raise ValueError("At least one property name is required.")
        query = """
        UNWIND $properties AS property
        MATCH (n:BIMElement {client_id: $client_id, project_id: $project_id})
        WHERE n[property] IS NOT NULL AND trim(toString(n[property])) <> ''
        WITH property, toString(n[property]) AS value, count(*) AS count
        RETURN property, value, count
        ORDER BY property, count DESC
        LIMIT $result_limit
        """
        columns, rows, elapsed = self._run(
            query,
            {
                "client_id": client_id,
                "project_id": project_id,
                "properties": selected,
                "result_limit": self.settings.max_result_rows,
            },
        )
        return store.add(
            tool="property-profile",
            purpose=purpose,
            query=query.strip(),
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=len(rows) >= self.settings.max_result_rows,
            elapsed_ms=elapsed,
        )

    def run_agent_cypher(
        self,
        *,
        client_id: str,
        project_id: str,
        query: str,
        parameters: dict[str, Any],
        store: EvidenceStore,
        purpose: str,
        provenance: dict[str, Any],
    ) -> EvidenceArtifact:
        query = self.guard.validate(query)
        scoped_parameters = {
            **parameters,
            "client_id": client_id,
            "project_id": project_id,
        }
        # EXPLAIN catches syntax, missing parameters, and invalid schema references before execution.
        self._run(f"EXPLAIN {query}", scoped_parameters)
        columns, fetched_rows, elapsed = self._run(query, scoped_parameters)
        truncated = len(fetched_rows) > self.settings.max_result_rows
        rows = fetched_rows[: self.settings.max_result_rows]
        return store.add(
            tool="cypher-query",
            purpose=purpose,
            query=query,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            elapsed_ms=elapsed,
            **provenance,
        )
