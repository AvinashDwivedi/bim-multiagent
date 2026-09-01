# BIM Built-in Tools Agent

An OpenAI Responses API agent for answering questions over one three-file BIM export. The runtime has no custom
function tools, MCP servers, planner tools, reviewer tools, skills, subagents, or project-specific routing.

## Tool policy

The active runtime exposes only:

- `shell`: the OpenAI equivalent of the reference Claude SDK agent's `Read`, `Grep`, `Glob`, and `Bash` tools.
- `web_search`: the OpenAI equivalent of `WebSearch` and `WebFetch`; the hosted tool searches and opens pages.

Like the Claude Agent SDK reference, `shell` executes native Bash with the selected project directory as its
working directory. On Windows, Git Bash is discovered automatically; set `BIM_BASH_PATH` only when Bash is in a
custom location. Docker is not installed, started, or used. The agent instruction limits Bash to read-only project
analysis and disables dedicated write tools, matching the reference runtime's permission model.

Project facts must come from the selected project's three artifacts. Web evidence is reserved for standards,
regulations, product references, or current public information and cannot substitute for project evidence.

`GET /api/tools` reports the effective policy and its mapping to the Claude Agent SDK reference tools.

## Project contract

The selected project directory must contain exactly one file for each role:

```text
project-data/
|-- <name>.ifc
|-- <name>-properties.json
`-- <name>-tree.json
```

The JSON files may also be named `properties.json` / `tree.json` or use an underscore before the role.

For the evaluator and web application, place project folders under `bim-data/<client_id>/<project_id>/`. The service discovers
that collection on every `GET /api/projects` request, reports incomplete folders separately, and resolves each chat
request by its selected `client_id` and `project_id`. Set `BIM_PROJECTS_ROOT` only to use a different collection directory.
Ambiguous or incomplete directories are rejected.

## Setup

```powershell
python -m pip install -e ".[test]"
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Windows requires Git for Windows, which supplies the same native Bash environment
used by Claude Code. The application automatically maps `python`, `python3`, and `py` to the active interpreter,
so Microsoft Store aliases do not interfere. API keys and token/password environment variables are removed from
the Bash subprocess environment.

## CLI and server

```powershell
python -m bim_agent --data-dir test-project-data inspect --json
python -m bim_agent --data-dir test-project-data ask "How many doors are in the model?"
python -m bim_agent --data-dir test-project-data serve --host 127.0.0.1 --port 8000
# Multi-project server used by bim-evaluator (loads ./bim-data automatically):
python -m bim_agents.webapp
```

Core endpoints:

- `GET /api/health`
- `GET /api/tools`
- `GET /api/inspect`
- `POST /api/ask` with `{"question":"..."}`
- Evaluator compatibility: `GET /api/projects`, `POST /api/chat`, and `POST /api/chat/stream`

Reports include artifact hashes, model response IDs, built-in tool traces, token/cost accounting, limitations, and
a request-scoped JSONL audit trace. Web-search fees are reported separately from token-price estimates.

Every run prints a compact terminal timeline for model turns, requested commands, execution backend, result
previews, elapsed time, estimated cost, and final status. Set `BIM_PRETTY_LOGS=false` to disable it or `NO_COLOR=1`
to keep the layout without color. Private model reasoning is never printed.

## Tests

```powershell
pytest
```

The policy tests assert that only `shell` and `web_search` are sent to the Responses API and that native Bash—not
Docker or a partial shell emulator—executes project analysis.
