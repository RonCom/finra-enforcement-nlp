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
| 2026-10-10 | Label extraction, second round: a rule number that exists in only one rulebook moves to it (NASD 2110, 2510, 3010-3013, 3030, 3040, 3050; FINRA 2010, 2111, 3270, 3280, 4511); the whole Code of Procedure (FINRA 9000-9999) and Exchange Act Sections 19 and 15A are never labels; more prior-case phrasings are skipped ("entered into an AWC", "Prior Matter", another person's consent); "constituted" alone no longer marks a violation sentence; OCR-split numbers after a FINRA or NASD citation are joined ("451 1", "20 I 0"); "§ 17(a)", "(MSRB) Rule G-17", bare "Rule 606" (Regulation NMS) and "Rule 204" (Regulation SHO), Regulation Crowdfunding, FINRA Funding Portal Rules and "IM-12000 of FINRA's Code" are read. Masking adds a pass that masks any remaining "Rule/Section <number>", "§", "G-<n>" or SEC-style number, and the leak check no longer counts exam names ("Series 7"). The gate is re-measured on a third sample (seed 11). | The second hand-check (seed 7, after the first round) failed: precision 0.907, recall 0.944. False positives came mostly from procedure rules and statute sections in decisions, rules from earlier cases, and bare rules that took the wrong rulebook; misses from "§" sections, OCR-split numbers and regulations with no pattern. Re-extracting the two checked samples from their exported sentences gives 0.985/0.992 (seed 42) and 0.992/0.985 (seed 7), but both samples shaped the rules. Masking: 300 of 6,198 summaries had a leak; 118 of 491 leaks were exam names and the rest MSRB, Regulation NMS and statute references the patterns missed. |
| 2026-10-10 | Three more prior-case phrasings are skipped ("was censured and fined", "included fines of", "in that proceeding"). The gate is re-measured on a fourth sample (seed 23). | The third hand-check (seed 11, after the second round) failed narrowly: recall 1.000, precision 0.973, or 0.952 if six rules that appear in no exported sentence turn out wrong. Of 8 confirmed false positives, 5 came from those phrasings and 3 from rules described or cited from other cases. Masking passed: no numbered reference left in 6,198 summaries. |
| 2026-10-10 | Rule numbers with a letter suffix ("6380A", "7230A") stay in their citation list and map to the base rule; "consented to findings that" marks an earlier case. | The fourth hand-check (seed 23) failed: precision 0.936, recall 0.981. Rule-level precision by sample: 0.881, 0.907, 0.973, 0.936; each sample brought new one-off phrasings for earlier cases and passing mentions (an earlier consent, a staff letter, case citations, a third party's violation). At the series level the classifier uses (Rule 2010 dropped), seeds 11 and 23 give precision 0.978 and 0.962, recall 1.000 and 0.993, with 3 and 5 of 100 cases carrying a wrong series. |
| 2026-10-10 | Label gate missed and accepted by the study owner. The 0.98 gate on rule-level precision and recall is not met: the last two hand-checks scored precision 0.973 and 0.936, recall 1.000 and 0.981. The study goes ahead on these labels, with label noise reported: at the series level the classifier uses, 3 and 5 of 100 checked cases (3-5%) carry a wrong series, mostly an extra series from a rule cited about an earlier case or another person. Per-series F1 and Brier scores are read against that noise floor, and results.md reports it next to every classifier metric. The four checked samples and their corrections are kept (seeds 42, 7, 11, 23). | Four rounds moved precision from 0.881 to 0.936-0.973 without converging: each sample brought new one-off phrasings for earlier cases and passing mentions, and three documents held 11 of the last sample's 18 false positives. Restricting extraction to AWC charging sections would not cover SEC and NAC decisions, which have no such headings. |
| 2026-10-10 | Series labels use each NASD rule's FINRA successor (NASD 3010 -> FINRA 3110, 3040 -> 3280, 3030 -> 3270, 2510 -> 3260, 2310 -> 2111, 2110 -> 2010, which is then dropped, and so on; the map is `labels.NASD_SUCCESSOR`). NASD rules with no successor in the map keep their own series. The FINRA 0100 series (Rule 0140) joins the Code of Procedure as never a label. Raw extracted rules are unchanged. | Decided after seeing the first validation baseline (macro-F1 0.698). The NASD series scored F1 0.24-0.53 and NASD:3000 was predicted 96 times for 23 true cases: FINRA charges the NASD or the FINRA number by when the conduct happened, so one obligation split into two classes that the summary text can't separate, and the 2023 validation year has less pre-2015 conduct than the 2016-2022 training years. The test split has not been scored. |
| 2026-10-10 | Successor map extended with the NASD rules left after the first pass (2120 and 2020 -> FINRA 2020, 2330 -> 2150, 2230 -> 2232, 3360 -> 4560, IM-2110-2 -> 5320, 2211/2212 -> 2210/3230, IM-2310(-2) -> 2111, 1050 -> 1220, and others; scripts/nasd_rules.py lists what's left). Macro-F1 is also reported over series with at least 10 validation cases, next to macro-F1 over all series. | After the first map, NASD:1000 and NASD:2000 still had 30+ training cases (106 rule citations) but 2 and 1 validation cases. Series with 1-2 validation cases move macro-F1 by about 0.06 per case, which would drown a comparison between models. |
| 2026-10-10 | Zero-shot classifier: each series is described by its Rulebook title and the rules most often charged in it, written by hand (`zeroshot.SERIES`), instead of scope text taken from the downloaded Rulebook. The model gives a probability per series; scored with the baseline's metrics at threshold 0.5. | The Rulebook download waits on the terms-of-use check, and the zero-shot run can go ahead with the local model now. The descriptions name rule numbers, so the model sees what each series covers; the summaries it reads stay masked. |
| 2026-10-10 | Zero-shot prompt v2: two sentences on how FINRA charges. Misconduct with no specific rule (conversion, forgery, false statements) is usually charged under Rule 2010 alone, so no series; an Exchange Act or Securities Act section is usually charged with the rule under it (17(a) with 17a-3/17a-4, 15(c) with 15c3-x, 10(b) with 10b-5). v1's answers stay stored; each version has its own report. The version scored on the test split is the one with the better validation macro-F1 over 10+ series. | Tuning on validation, as for the other models. v1 scored 0.673 macro-F1 (0.672 over 10+ series): FINRA:2000 was predicted 123 times for 61 cases and FINRA:3000 208 for 157, and SEC_SECTION 13 times for 44 (F1 0.14). Speed settings changed between v1 and v2 (flash attention, q8 KV cache); a 20-case recheck under the new settings moved 2 of 300 probabilities across 0.5. |
