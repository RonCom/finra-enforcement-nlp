# finra-enforcement-nlp

Predicts which rules a FINRA disciplinary action charged from the case facts, and tests whether a RAG system over the FINRA Rulebook retrieves and cites the same rules. The plan and every expectation are in [docs/spec.md](docs/spec.md), written before any data was pulled.

## Setup

```powershell
uv sync
uv run pytest
```

## Cases and labels

```powershell
# 1. Find the monthly report links; check the count covers 2016-2026 before going on
uv run python -m finra_nlp.monthly discover

# 2. Download and split reports into one row per case (raw.monthly_cases)
uv run python -m finra_nlp.monthly parse

# 3. Confirm the Disciplinary Actions Online page yields a document link for one case
uv run python -m finra_nlp.dao probe --case 2023077018401

# 4. Pull rule labels from the full case documents (raw.case_document_labels)
uv run python -m finra_nlp.dao labels --limit 50
uv run python -m finra_nlp.dao labels
```

Data and the HTTP cache go in `data/`, which git ignores.
