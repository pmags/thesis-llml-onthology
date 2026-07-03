# Codex Instructions

These instructions apply to the entire repository.

## Source Of Truth

- Keep the existing GitHub Copilot setup in `.github/` intact. Do not rename, delete, or rewrite those files as part of Codex setup work.
- Before making project changes, read `.github/copilot-instructions.md` for the ontology algorithm, architecture, command rules, and project constraints.
- For Python changes, also read `.github/instructions/python.instructions.md`.
- Treat feature specifications and plans under `.github/features/<feature_id>/` as the task source of truth when working on feature-loop tasks.

## Project Guardrails

- Do not use embedding-based semantic similarity. Semantic similarity and relationship classification must be performed through LLM prompts that return structured values.
- Require structured JSON from LLM prompts and parse responses defensively.
- Cache pairwise similarity results with sorted term-pair keys to avoid redundant LLM calls.
- Use `rdflib` for RDF graph construction and serialization.
- Prefer `RDFS.subClassOf` for class hierarchy and `RDF.type` for instance classification.
- Mock LLM calls in tests to avoid API costs.

## Working With The Existing Agent Loop

The Copilot agent prompts in `.github/agents/` are reusable role specifications for Codex:

- For planning requests, read `.github/agents/planner.agent.md` and follow its plan-only protocol.
- For build requests that target the next planned task, read `.github/agents/builder.agent.md` and execute exactly the task named in `## NEXT_ACTION`.
- For review requests, read `.github/agents/reviewer.agent.md` and use a review/evaluation stance.

For ordinary coding requests that do not explicitly invoke the planner, builder, or reviewer loop, use this `AGENTS.md` together with the project instructions in `.github/` and implement the requested change directly.

## Commands And Environment

- Use `pytest tests/ -v` for the full test suite unless a narrower test command is justified by the task.
- Do not run `python -m pytest` or `python3 -m pytest`; the project instructions explain that those can resolve to the wrong Python in the devcontainer.
- When an explicit Python interpreter is needed in the devcontainer, use `/opt/conda/bin/python`.
- Do not install dependencies unless the user explicitly asks or a required verification step cannot run without them.

## Change Discipline

- Keep edits scoped to the requested task and existing architecture.
- Preserve completed plan work and user-authored changes.
- Update the relevant `.github/features/<feature_id>/plan.md` only when operating in the documented planner/builder/reviewer loop.
- Record progress and findings in the plan file for feature-loop tasks; do not create parallel status files.
