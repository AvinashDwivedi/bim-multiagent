from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase, Query, RoutingControl

from ontology.concept_ontology import ConceptOntology, load_ontology


@dataclass(frozen=True)
class Settings:
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str
    client_id: str
    project_id: str
    query_timeout_seconds: float = 20.0

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()

        required = {
            "NEO4J_URI": os.getenv("NEO4J_URI"),
            "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME"),
            "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
            "BIM_CLIENT_ID": os.getenv("BIM_CLIENT_ID"),
            "BIM_PROJECT_ID": os.getenv("BIM_PROJECT_ID"),
        }

        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        return cls(
            neo4j_uri=required["NEO4J_URI"],
            neo4j_username=required["NEO4J_USERNAME"],
            neo4j_password=required["NEO4J_PASSWORD"],
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            client_id=required["BIM_CLIENT_ID"],
            project_id=required["BIM_PROJECT_ID"],
            query_timeout_seconds=float(os.getenv("BIM_QUERY_TIMEOUT_SECONDS", "20")),
        )


class BimContext:
    """Neo4j connection plus project-scoped BIM and ontology context."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.driver: Driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(
                settings.neo4j_username,
                settings.neo4j_password,
            ),
            max_connection_pool_size=50,
            max_connection_lifetime=200,
            connection_timeout=min(settings.query_timeout_seconds, 20.0),
        )

        # Composes:
        # concept_ontology.yaml
        #   → ontology/clients/<client_id>.yaml
        #   → ontology/projects/<project_id>.yaml
        self.ontology: ConceptOntology = load_ontology(
            client_id=settings.client_id,
            project_id=settings.project_id,
        )

    def connect(self) -> None:
        """Fail immediately if Neo4j is unavailable or credentials are invalid."""
        self.driver.verify_connectivity()

    def close(self) -> None:
        self.driver.close()

    def query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        records, _, _ = self.driver.execute_query(
            Query(cypher, timeout=self.settings.query_timeout_seconds),
            parameters_=parameters or {},
            database_=self.settings.neo4j_database,
            routing_=RoutingControl.READ,
        )
        return [record.data() for record in records]

    def resolve_allowed_sources(self) -> list[str]:
        """
        Resolve BIM sources through the authorized client/project hierarchy.

        Returns an empty list on missing or invalid scope, so callers fail closed.
        """
        cypher = """
        MATCH (c:Client {id: $client_id})
              -[:HAS_PROJECT]->
              (p:Project {id: $project_id})
        MATCH (p)-[:HAS_BIM]->
              (:BIMHub)-[:CONTAINS]->
              (ip:IfcProject)
        WHERE ip.source IS NOT NULL
        RETURN DISTINCT ip.source AS source
        ORDER BY source
        """

        rows = self.query(
            cypher,
            {
                "client_id": self.settings.client_id,
                "project_id": self.settings.project_id,
            },
        )
        return [row["source"] for row in rows if row.get("source")]

    def get_project_summary(self) -> dict[str, Any]:
        """Return a small, safely scoped BIM-model summary."""
        allowed_sources = self.resolve_allowed_sources()

        if not allowed_sources:
            return {
                "connected": True,
                "authorized": False,
                "allowed_sources": [],
                "element_count": 0,
            }

        rows = self.query(
            """
            MATCH (b:BIMElement)
            WHERE b.source IN $allowed_sources
            RETURN
                count(DISTINCT b.object_id) AS element_count,
                count(DISTINCT b.canonical_type) AS canonical_type_count,
                count(DISTINCT b.canonical_level) AS level_count
            """,
            {"allowed_sources": allowed_sources},
        )

        counts = rows[0] if rows else {}
        return {
            "connected": True,
            "authorized": True,
            "allowed_sources": allowed_sources,
            **counts,
        }

    def resolve_term(self, term: str) -> dict[str, Any]:
        """Demonstrate the ontology resolution helpers."""
        return {
            "input": term,
            "space_function": self.ontology.resolve_space_function(term),
            "ifc_classes": self.ontology.resolve_element_classes(term),
            "retrieval_terms": self.ontology.retrieval_terms_for(term),
            "absence_notes": self.ontology.absence_notes_for_query(term),
        }

    def __enter__(self) -> "BimContext":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def main() -> None:
    settings = Settings.from_env()

    with BimContext(settings) as bim:
        print("Neo4j connection: OK")
        print(f"Ontology version: {bim.ontology.version}")
        print(f"Loaded concepts: {len(bim.ontology.concepts)}")
        print(f"Loaded permit measures: {len(bim.ontology.permit_measures)}")

        print("\nProject summary:")
        print(bim.get_project_summary())

        print("\nExample ontology resolutions:")
        for term in ("apartment", "elevator", "door", "wood"):
            print(bim.resolve_term(term))


if __name__ == "__main__":
    main()
