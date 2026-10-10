# FINRA enforcement actions: rule classification and cited RAG

Written before any data is pulled or code runs. Results go in `docs/results.md`, scored against each expectation below. Changes after data is seen are logged at the bottom with a date and reason.

## Questions

1. From the facts of a FINRA disciplinary action, with rule citations removed, can a model predict which rule series were violated?
2. Can a RAG system over the FINRA Rulebook retrieve and cite the rules FINRA actually charged?

FINRA's own citations are the labels for both questions, so neither needs hand-labeled training data.

## Data

| Source | Use | Pre-check |
| --- | --- | --- |
| Monthly "Disciplinary and Other FINRA Actions" reports, 2016–2026 (PDF) | One case summary per action: facts, rules cited, sanctions | Text extracts cleanly from 3 sampled years |
| FINRA Disciplinary Actions Online | Full AWCs and decisions for a 300-case sample | Downloadable as text PDFs at a polite request rate |
| FINRA Rulebook (finra.org) | RAG corpus, chunked by rule and subsection | Terms of use allow bulk download for non-commercial use |

The corpus starts in 2016 so that every case cites the consolidated FINRA rule numbers, avoiding NASD rules that were renumbered during consolidation.

## Labels

1. **Source:** labels come from the violation sentences of each case's full document (AWC, complaint or decision) in Disciplinary Actions Online, matched to the monthly summary by FINRA case number. Recent monthly summaries omit rule numbers for most cases. Where a summary does cite rules, its citations are kept as a second label set and their agreement with the full-document labels is reported. Where the document gives no rule, the summary's citations are the label, then rules from an OCR pass over the document (see change log, 2026-10-10); each case's label source is kept.
2. **Extraction:** regex pulls every cited rule: "FINRA Rule(s) NNNN", "NASD Rule NNNN", "MSRB Rule G-NN", "Exchange Act Rule NN…" and statute sections.
3. **Hand-check:** check 100 cases by hand. The gate is citation precision and recall ≥ 0.98 before going further.
4. **Classifier labels:** the rule series, as numbered in the Rulebook (2000, 3000, 4000, 5000, 6000, 7000, 8000 and so on). Rule 2010 is dropped as a label because FINRA cites it alongside nearly every other violation; its frequency is reported.
5. **Masking:** numbered rule references are replaced with a token in each summary. The build fails if any numbered reference remains. Rule titles stay in, because they describe the conduct (for example, "outside business activities"); the share of test cases whose summary contains the title of a cited rule is reported.
6. **Rare series:** series with fewer than 30 training cases are merged into "other".

## Classifier

| Model | Detail |
| --- | --- |
| Baseline | TF-IDF, one-vs-rest logistic regression |
| Fine-tuned transformer | Small encoder (ModernBERT-base or DeBERTa-v3-base), multi-label head |
| Zero-shot | Local model (Ollama) given each series' title and scope text from the Rulebook |

- **Splits by action date:** train 2016–2022, validation 2023, test 2024–2026.
- **Metrics:** macro-F1 and per-series F1, Brier score per series, and a reliability plot.

## RAG

- **Retrieval:** hybrid retrieval (BM25 plus embeddings) over rule chunks, reusing the policy-rag pipeline.
- **Input and output:** the input is a masked case summary; the system answers with the rules it believes apply and cites each section.
- **Retrieval scoring:** for each test case, check whether any rule FINRA cited, other than 2010, appears among the top 5 retrieved chunks.
- **Out-of-scope questions:** 20 questions answered only by SEC rules or state law. The correct response is to decline.
- **Judge check:** an LLM judge scores answer correctness against FINRA's citations and is checked against hand labels on 100 answers.

## Expectations

| ID | Test | Expectation | If it fails |
| --- | --- | --- | --- |
| H1 | Fine-tuned transformer vs. TF-IDF baseline, test macro-F1 | Gain ≥ 0.05 | Report; compare per-series F1 to find where the baseline matches it |
| H2 | Zero-shot local model vs. fine-tuned transformer, test macro-F1 | Zero-shot lower by ≥ 0.10 | If within 0.10, report as a finding: labeled data adds little here |
| H3 | Retrieval: a cited rule in the top 5 chunks, test cases | ≥ 70% | Break down by series and summary length |
| H4 | Out-of-scope questions declined | ≥ 18 of 20 | List the answered ones and the retrieved chunks |
| H5 | LLM judge vs. hand labels, Cohen's κ | ≥ 0.60 | Report judge scores as unreliable; use hand labels for the 100 |
| H6 | Full AWC vs. monthly summary as input, 300-case sample, transformer macro-F1 | Gain ≥ 0.03 | Report; summaries are enough for triage |

## Limits stated in advance

- **Mixed authorship:** summaries are written by FINRA staff who know the charges, so the wording leans toward the cited rules even after masking. Test scores are an upper bound for classifying raw tips or complaints.
- **Current rulebook:** the RAG corpus is today's Rulebook. Rules amended since a case was decided may retrieve correctly but read differently than they did when charged.
- **Settlements only:** the cited rules reflect what was charged and settled, which can differ from all rules the conduct touched.

## Shared code

The extraction module (local model, fixed schema, hand-check harness) is shared with `insider-screen`, which uses it on SEC litigation releases.

## Build

uv, DuckDB, dbt for the case and label tables, MLflow for runs, GitHub Actions running the masking check and unit tests on every push.

## Milestones

1. Monthly reports parsed; citations extracted and hand-checked; masking check passing.
2. Baseline and fine-tuned classifiers; validation results.
3. RAG over the Rulebook; retrieval, decline and judge evaluation.
4. Zero-shot comparison and AWC sample (H2, H6).
5. Frozen test-period run; `results.md`; write-up.

## Change log

| Date | Change | Reason |
| --- | --- | --- |
| 2026-10-08 | Labels from full case documents; summary citations kept as a second label set | The July 2026 monthly report cites rule numbers in a minority of its cases; the October 2023 report still cited them. Seen while writing the parser, before any data pull. |
| 2026-10-08 | Rule titles no longer masked; their presence is reported | Titles name the conduct, so removing them removes facts the classifier needs |
| 2026-10-10 | Cases whose document gives no rule take the monthly summary's citations; cases with neither go through OCR (Tesseract) and the same violation-sentence rules. A `label_source` column records document, summary or OCR. The 100-case hand-check stays on document labels; summary labels are checked through their agreement with document labels where both exist. | 213 of 6,198 cases had a document but no rule: 133 old scans whose OCR text garbles rule numbers, 53 with no text layer, about 20 whose fonts extract as shifted characters, and a few short or non-AWC documents. Dropping them would thin 2016–2017 training data. |
| 2026-10-10 | A five-digit FINRA or NASD rule number outside the numbers that exist (FINRA 11000–14999, NASD 10000–11999) keeps its first four digits | Footnote markers print against rule numbers in case documents ("NASD Rule 30401" is Rule 3040 with footnote 1; "FINRA Rules 8210 and 20101"). A spot-check found 59 cases with such numbers (58 from document text, 1 from OCR). In 50 the four-digit rule was already in the labels from another sentence; in 9 it wasn't, and the fix recovers the right rule in 6 (the other 3 become FINRA 3111, 2015 and a FINRA 3030 that should be NASD, which leave the labels' series unchanged except one). |
| 2026-10-10 | Label extraction: OCR-misread or glued rulebook names ("FlNRA", "F?NRA", "ofNASD", "NASO") are read as FINRA or NASD; sentences about earlier cases (relevant disciplinary history, prior AWCs, actions against others) are skipped; Rules 9216, 9143 and 9144 (AWC procedure and waiver of rights) are never labels; only the verb "waive" marks a waiver sentence ("sales charge waivers" is conduct); subsection lists ("3110(a) and (b) and 2010") stay in one citation list; Regulation NMS ("Exchange Act Rule 606"), Regulation M and exchange rules ("Nasdaq Rule 4613") get their own families. The gate is re-measured on a fresh 100-case sample (seed 7). | The first hand-check (seed 42) failed: precision 0.881, recall 0.966. Of 34 false positives, 15 came from misread or glued names, 16 from disciplinary history or other cases, and 3 from a procedural rule or another person's case; the 9 misses came from misread names, a rule number split by OCR ("831 1"), the waiver filter, and two regulations with no pattern. Re-extracting that sample from its exported sentences gives 0.981 and 0.985, but the rules were written from it, so it no longer measures them. |
