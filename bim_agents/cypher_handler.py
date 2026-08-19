from __future__ import annotations

import re
from typing import Any, Callable

from pydantic import BaseModel, Field


class CypherExecution(BaseModel):
    """One auditable, parameterized read-only calculation executed against Neo4j."""

    statement: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    row_count: int = Field(ge=0)


class CypherQueryHandler:
    """Validates and executes compiler-produced Cypher calculation queries."""

    _WRITE_KEYWORDS = re.compile(
        r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|ALTER|LOAD\s+CSV|FOREACH)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        query: Callable[[str, dict[str, Any]], list[dict[str, Any]]],
        allowed_sources: list[str],
    ) -> None:
        self._query = query
        self._allowed_sources = list(allowed_sources)
        self.executions: list[CypherExecution] = []

    def execute(self, statement: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute scoped calculation Cypher after enforcing the read-only contract."""
        if self._WRITE_KEYWORDS.search(statement):
            raise PermissionError("The Cypher Query Handler accepts read-only calculations only.")
        if "$allowed_sources" not in statement:
            raise PermissionError("Calculation Cypher must be restricted to authorized sources.")
        scoped_parameters = dict(parameters)
        scoped_parameters["allowed_sources"] = self._allowed_sources
        rows = self._query(statement, scoped_parameters)
        self.executions.append(CypherExecution(
            statement=statement,
            parameters=scoped_parameters,
            row_count=len(rows),
        ))
        return rows

    def audit_log(self) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in self.executions]
