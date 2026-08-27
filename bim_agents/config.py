from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class Settings:
    llm_provider: str
    openai_api_key: str | None
    anthropic_api_key: str | None
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str | None
    agent_model: str
    worker_model: str
    reasoning_effort: str
    high_reasoning_effort: str
    low_reasoning_effort: str
    max_agent_turns: int
    investigator_max_turns: int
    auditor_max_turns: int
    repair_max_turns: int
    repair_auditor_max_turns: int
    max_artifacts: int
    investigator_artifact_budget: int
    auditor_artifact_budget: int
    repair_artifact_budget: int
    repair_auditor_artifact_budget: int
    max_cypher_repairs: int
    max_repairs: int
    max_model_calls: int
    trace_enabled: bool
    trace_dir: str
    max_result_rows: int
    query_timeout_seconds: float
    api_host: str
    api_port: int

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Settings":
        load_dotenv(env_file or ROOT / ".env", override=False)

        def required(name: str) -> str:
            value = os.getenv(name, "").strip()
            if not value:
                raise RuntimeError(f"Missing required environment variable: {name}")
            return value

        provider = os.getenv("BIM_LLM_PROVIDER", "openai").strip().casefold()
        if provider not in {"openai", "anthropic"}:
            raise RuntimeError("BIM_LLM_PROVIDER must be either 'openai' or 'anthropic'.")
        openai_key = os.getenv("OPENAI_API_KEY", "").strip() or None
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip() or None
        if provider == "openai" and not openai_key:
            raise RuntimeError("Missing required environment variable: OPENAI_API_KEY")
        if provider == "anthropic" and not anthropic_key:
            raise RuntimeError("Missing required environment variable: ANTHROPIC_API_KEY")
        if provider == "openai":
            agent_model = os.getenv("BIM_OPENAI_AGENT_MODEL", "gpt-5.6-sol").strip()
            worker_model = os.getenv("BIM_OPENAI_WORKER_MODEL", "gpt-5.6-sol").strip()
        else:
            agent_model = os.getenv("BIM_ANTHROPIC_AGENT_MODEL", "claude-sonnet-4-6").strip()
            worker_model = os.getenv("BIM_ANTHROPIC_WORKER_MODEL", "claude-sonnet-4-6").strip()

        trace_value = os.getenv("BIM_TRACE_ENABLED", "true").strip().casefold()
        if trace_value not in {"true", "false", "1", "0", "yes", "no"}:
            raise RuntimeError("BIM_TRACE_ENABLED must be a boolean value.")
        trace_path = Path(os.getenv("BIM_TRACE_DIR", "logs/agent-traces").strip())
        if not trace_path.is_absolute():
            trace_path = ROOT / trace_path
        allowed_efforts = {"none", "low", "medium", "high", "xhigh", "max"}
        default_effort = os.getenv("BIM_REASONING_EFFORT", "medium").strip().casefold()
        high_effort = os.getenv("BIM_REASONING_EFFORT_HIGH", "high").strip().casefold()
        low_effort = os.getenv("BIM_REASONING_EFFORT_LOW", "low").strip().casefold()
        for name, value in {
            "BIM_REASONING_EFFORT": default_effort,
            "BIM_REASONING_EFFORT_HIGH": high_effort,
            "BIM_REASONING_EFFORT_LOW": low_effort,
        }.items():
            if value not in allowed_efforts:
                raise RuntimeError(
                    f"{name} must be one of: {', '.join(sorted(allowed_efforts))}."
                )
        max_artifacts = int(os.getenv("BIM_AGENT_MAX_ARTIFACTS", "18"))
        phase_budgets = {
            "investigator": int(os.getenv("BIM_INVESTIGATOR_ARTIFACT_BUDGET", "8")),
            "auditor": int(os.getenv("BIM_AUDITOR_ARTIFACT_BUDGET", "4")),
            "repair": int(os.getenv("BIM_REPAIR_ARTIFACT_BUDGET", "4")),
            "repair_auditor": int(os.getenv("BIM_REPAIR_AUDITOR_ARTIFACT_BUDGET", "2")),
        }
        if any(value < 0 for value in phase_budgets.values()):
            raise RuntimeError("Evidence phase budgets cannot be negative.")
        if sum(phase_budgets.values()) > max_artifacts:
            raise RuntimeError(
                "Evidence phase budgets cannot exceed BIM_AGENT_MAX_ARTIFACTS."
            )
        max_agent_turns = int(os.getenv("BIM_AGENT_MAX_TURNS", "10"))
        phase_turns = {
            "investigator": int(os.getenv("BIM_INVESTIGATOR_MAX_TURNS", "6")),
            "auditor": int(os.getenv("BIM_AUDITOR_MAX_TURNS", "3")),
            "repair": int(os.getenv("BIM_REPAIR_MAX_TURNS", "3")),
            "repair_auditor": int(os.getenv("BIM_REPAIR_AUDITOR_MAX_TURNS", "2")),
        }
        if any(value < 1 or value > max_agent_turns for value in phase_turns.values()):
            raise RuntimeError(
                "Phase turn budgets must be between 1 and BIM_AGENT_MAX_TURNS."
            )

        return cls(
            llm_provider=provider,
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            neo4j_uri=required("NEO4J_URI"),
            neo4j_username=required("NEO4J_USERNAME"),
            neo4j_password=required("NEO4J_PASSWORD"),
            neo4j_database=os.getenv("NEO4J_DATABASE", "").strip() or None,
            agent_model=agent_model,
            worker_model=worker_model,
            reasoning_effort=default_effort,
            high_reasoning_effort=high_effort,
            low_reasoning_effort=low_effort,
            max_agent_turns=max_agent_turns,
            investigator_max_turns=phase_turns["investigator"],
            auditor_max_turns=phase_turns["auditor"],
            repair_max_turns=phase_turns["repair"],
            repair_auditor_max_turns=phase_turns["repair_auditor"],
            max_artifacts=max_artifacts,
            investigator_artifact_budget=phase_budgets["investigator"],
            auditor_artifact_budget=phase_budgets["auditor"],
            repair_artifact_budget=phase_budgets["repair"],
            repair_auditor_artifact_budget=phase_budgets["repair_auditor"],
            max_cypher_repairs=int(os.getenv("BIM_AGENT_MAX_CYPHER_REPAIRS", "2")),
            max_repairs=int(os.getenv("BIM_AGENT_MAX_REPAIRS", "1")),
            max_model_calls=int(os.getenv("BIM_AGENT_MAX_MODEL_CALLS", "24")),
            trace_enabled=trace_value in {"true", "1", "yes"},
            trace_dir=str(trace_path.resolve()),
            max_result_rows=int(os.getenv("BIM_AGENT_MAX_RESULT_ROWS", "250")),
            query_timeout_seconds=float(os.getenv("BIM_QUERY_TIMEOUT_SECONDS", "30")),
            api_host=os.getenv("BIM_API_HOST", "127.0.0.1"),
            api_port=int(os.getenv("BIM_API_PORT", "8000")),
        )
