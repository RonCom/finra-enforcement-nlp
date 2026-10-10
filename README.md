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

# 5. Cases whose document gives no rule fall back to the summary's citations; OCR the rest (needs Tesseract)
uv run python scripts/dao_unlabeled.py           # why each document gave no rule
uv run python -m finra_nlp.ocr --limit 3         # check Tesseract and the timing
uv run python -m finra_nlp.ocr                   # raw.case_ocr_labels
```

Data and the HTTP cache go in `data/`, which git ignores.

## Label checks

```powershell
uv run python -m finra_nlp.labels sample    # 100 cases to data/label_handcheck.csv; correct true_rules, set checked to Y
uv run python -m finra_nlp.labels score     # precision and recall against the 0.98 gate
uv run python -m finra_nlp.labels profile   # reports/label_profile.md
uv run python -m finra_nlp.labels export    # data/label_handcheck_docs.json: each sampled document's citing and violation sentences
uv run python -m finra_nlp.masking          # exits 1 if any masked summary still has a numbered rule reference
```

## Classifier

```powershell
uv run python -m finra_nlp.dataset          # model.dataset: series labels, 2010 dropped, splits by action date
uv run python -m finra_nlp.baseline         # TF-IDF + logistic regression; reports/baseline_validation.md (validation only)
uv run python -m finra_nlp.zeroshot --limit 5   # local model through Ollama: check answers and time per case
uv run python -m finra_nlp.zeroshot         # all validation cases (resumes); reports/zeroshot_validation.md
```

## Rulebook

```powershell
uv run python -m finra_nlp.rulebook terms   # reports/rulebook_terms.md: terms-of-use passages and robots.txt
uv run python -m finra_nlp.rulebook terms --accept "what you read and decided"
uv run python -m finra_nlp.rulebook download --limit 5
uv run python -m finra_nlp.rulebook download
```

The 20 out-of-scope questions for H4 are in [docs/out_of_scope_questions.csv](docs/out_of_scope_questions.csv); mark `keep` Y or N after review.
