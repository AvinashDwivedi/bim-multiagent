from __future__ import annotations

import re


class UnsafeCypherError(ValueError):
    pass


class CypherGuard:
    """Conservative static guard in front of a read-only Neo4j account."""

    _forbidden = re.compile(
        r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|ALTER|RENAME|GRANT|DENY|REVOKE|"
        r"LOAD\s+CSV|FOREACH|USE|SHOW|TERMINATE|START\s+DATABASE|STOP\s+DATABASE)\b",
        re.IGNORECASE,
    )
    _call = re.compile(r"\bCALL\s+([^\s({]+)", re.IGNORECASE)
    _allowed_calls = (
        "db.index.fulltext.querynodes",
        "db.index.vector.querynodes",
    )

    @staticmethod
    def _mask_strings(query: str) -> str:
        return re.sub(r"'(?:\\.|''|[^'])*'|\"(?:\\.|\"\"|[^\"])*\"", "''", query)

    def validate(self, query: str) -> str:
        query = query.strip()
        if not query:
            raise UnsafeCypherError("Cypher query is empty.")
        if len(query) > 20_000:
            raise UnsafeCypherError("Cypher query exceeds the size limit.")
        if ";" in query or "//" in query or "/*" in query:
            raise UnsafeCypherError("Comments and multiple Cypher statements are not allowed.")

        masked = self._mask_strings(query)
        if self._forbidden.search(masked):
            raise UnsafeCypherError("Only read-only Cypher is allowed.")
        if re.search(r"\b(apoc|dbms)\.", masked, re.IGNORECASE):
            raise UnsafeCypherError("APOC and DBMS procedures are not available to the agent.")
        for match in self._call.finditer(masked):
            procedure = match.group(1).casefold()
            if procedure not in self._allowed_calls:
                raise UnsafeCypherError(f"Procedure is not allowlisted: {procedure}")
        if "$client_id" not in query or "$project_id" not in query:
            raise UnsafeCypherError(
                "Every agent query must use both $client_id and $project_id parameters."
            )
        client_scope = re.search(
            r"(?:\.\s*client_id\b|\bclient_id\s*:)\s*(?:=\s*)?\$client_id\b",
            masked,
            re.IGNORECASE,
        )
        project_scope = re.search(
            r"(?:\.\s*project_id\b|\bproject_id\s*:)\s*(?:=\s*)?\$project_id\b",
            masked,
            re.IGNORECASE,
        )
        project_node_scope = re.search(
            r"(?:\.\s*id\b|\bid\s*:)\s*(?:=\s*)?\$project_id\b",
            masked,
            re.IGNORECASE,
        )
        if not client_scope or not (project_scope or project_node_scope):
            raise UnsafeCypherError(
                "The query does not explicitly constrain the authorized client and project."
            )
        if not re.search(r"\b(RETURN|YIELD)\b", masked, re.IGNORECASE):
            raise UnsafeCypherError("A read query must return evidence.")
        return query
