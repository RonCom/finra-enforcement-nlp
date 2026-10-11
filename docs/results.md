# Results

Scored against the expectations in [spec.md](spec.md). The classifiers were tuned on the validation split (2023) and scored once on the test split (2024-2026, 1,143 cases) with code tagged `freeze-2` (commit b947368); the full output is `reports/test_results.md`. Every rule changed after data was seen is in the spec's change log, dated, with the reason. The full-document comparison (H6) was scored once on 300 test cases with code tagged `freeze-3` (commit 96a1419); the output is `reports/h6_results.md`. The retrieval part (H3, H4, H5) was scored once with code tagged `freeze-4` (commit b459136); the outputs are `reports/retrieval_test.md`, `reports/h4_results.md` and `reports/h5_results.md`.

## Expectations

| ID | Test | Expectation | Result | Verdict |
| --- | --- | --- | --- | --- |
| H1 | Fine-tuned transformer vs. TF-IDF baseline, test macro-F1 | Gain ≥ 0.05 | +0.000 (0.694 vs 0.693) | Failed |
| H2 | Zero-shot local model vs. fine-tuned transformer, test macro-F1 | Zero-shot lower by ≥ 0.10 | Lower by 0.030 (0.664 vs 0.694) | Failed: labeled data adds little here |
| H3 | Retrieval: a charged rule among the top 5 rules (BM25) | ≥ 70% | 71.3% of 950 test cases | Met |
| H4 | Out-of-scope questions declined | ≥ 18 of 20 | 20 of 20 | Met, from a system that also declines 43% of in-scope cases |
| H5 | LLM judge vs. hand labels, Cohen's κ | ≥ 0.60 | 0.741 (agreement 0.87) | Met; labels drafted by an assistant model and reviewed by hand |
| H6 | Full AWC vs. monthly summary as input | Gain ≥ 0.03 | -0.067 (0.614 vs 0.681), 95% interval -0.114 to -0.016 | Failed: summaries are enough for triage |

## Test split

| Model | Macro-F1, all 15 series | Macro-F1, 14 series with 10+ test cases |
| --- | --- | --- |
| TF-IDF + logistic regression | 0.693 | 0.714 |
| ModernBERT-base, fine-tuned, average of 3 seeds | 0.694 | 0.715 |
| gemma4:26b zero-shot, prompt v2 | 0.664 | 0.691 |

H1 and H2 are judged on macro-F1 over all series, as pre-registered. The figure over series with 10+ cases was added during validation (spec change log) because series with one or two cases swing the all-series average by about 0.06 per case; it tells the same story here.

The hand-checks put 3-5% of cases with a wrong series label, so differences under about 0.02 can't be read as a model being better. Paired bootstrap over test cases (2,000 resamples, `scripts/test_bootstrap.py`, added after the test run and descriptive only) gives these 95% intervals:

| | Value | 95% interval | Resamples above 0 |
| --- | --- | --- | --- |
| Baseline macro-F1 | 0.693 | 0.649 to 0.725 | |
| Fine-tuned macro-F1 | 0.694 | 0.650 to 0.727 | |
| Zero-shot macro-F1 | 0.664 | 0.622 to 0.699 | |
| H1: fine-tuned minus baseline | +0.000 | -0.023 to +0.025 | 53% |
| H2: fine-tuned minus zero-shot | +0.030 | -0.006 to +0.073 | 94% |

H1's interval excludes the 0.05 gain it called for. H2's excludes the 0.10 gap; the fine-tuned model is probably ahead of the zero-shot model, but the interval doesn't rule out a tie.

### What each model does well

| Series | Test cases | Baseline | Fine-tuned | Zero-shot |
| --- | --- | --- | --- | --- |
| FINRA:3000 supervision, outside activities | 509 | 0.952 | 0.952 | 0.897 |
| FINRA:8000 failing to cooperate (8210) | 286 | 0.947 | 0.953 | 0.959 |
| FINRA:4000 books and records, net capital | 182 | 0.869 | 0.852 | 0.850 |
| FINRA:2000 suitability, fraud, communications | 176 | 0.729 | 0.747 | 0.623 |
| SEC_RULE Exchange Act rules | 194 | 0.581 | 0.657 | 0.680 |
| SEC_SECTION Exchange Act sections | 130 | 0.687 | 0.612 | 0.494 |
| FINRA:1000 registration, Form U4 | 56 | 0.673 | 0.690 | 0.452 |
| MSRB | 45 | 0.829 | 0.867 | 0.905 |
| SA_SECTION Securities Act sections | 29 | 0.571 | 0.533 | 0.515 |

- **The two trained models tie on F1 but differ by series.** They agree within 0.02 on FINRA:3000, 8000, 4000 and 2000; on the statutes they split, the fine-tuned model ahead on SEC_RULE (+0.076) and the baseline ahead on SEC_SECTION (+0.075). The fine-tuned model's probabilities are better calibrated: lower Brier score on 10 of 15 series and tied on 3 (FINRA:3000 0.033 vs 0.042, FINRA:2000 0.060 vs 0.074).
- **The zero-shot model is close where a series has its own vocabulary:** 8210 cooperation cases (0.959), MSRB (0.905), Regulation NMS (0.950), trade reporting (FINRA:6000, 0.905). It loses where the label depends on FINRA's charging habits rather than on the conduct. It predicts FINRA:1000 for 143 cases against 56 true ones (any mention of a Form U4 or registration), and it splits statute sections from their rules differently from FINRA's drafting (SEC_SECTION 0.494).
- **Statutes are hard for every model** (SEC_RULE 0.58-0.68, SEC_SECTION 0.49-0.69, SA_SECTION 0.51-0.57). Whether an AWC charges Section 17(a) next to Rule 17a-4, or only the rule, is a drafting choice the summary rarely shows.
- **"other" (12 test cases, rare series merged) scores 0 for both trained models** and 0.214 for the zero-shot model, which costs the trained models about 0.05 of all-series macro-F1.

### Calibration

The spec's reliability plot, drawn after the frozen run from the stored test probabilities (`scripts/reliability.py`, `reports/reliability_test.md`): every case-and-series probability pooled, ten bins.

| Predicted | Baseline: observed | Fine-tuned: observed | Zero-shot: observed |
| --- | --- | --- | --- |
| 0.1-0.2 | 0.035 (mean 0.14) | 0.117 (0.15) | 0.055 (0.10) |
| 0.3-0.4 | 0.183 (0.34) | 0.223 (0.35) | 0.161 (0.30) |
| 0.5-0.6 | 0.446 (0.55) | 0.402 (0.55) | 0.286 (0.50, 7 points) |
| 0.7-0.8 | 0.696 (0.75) | 0.617 (0.76) | 0.311 (0.70) |
| 0.8-0.9 | 0.878 (0.85) | 0.797 (0.85) | 0.464 (0.81) |
| 0.9-1.0 | 0.974 (0.97) | 0.955 (0.99) | 0.833 (0.95) |
| Expected calibration error | 0.040 | 0.013 | 0.037 |

- **The fine-tuned model has the lowest ECE because it commits.** 86% of its points are under 0.1 and 7% over 0.9, where it's close to right; it leaves fewer points in the middle than the baseline (1,261 between 0.1 and 0.9 against 3,733). In its 0.5-0.6 and 0.7-0.8 bins it overstates by 0.14-0.15.
- **The baseline overstates the low-middle range.** Its balanced class weights push probabilities up: of the points it scores 0.1-0.2, 3.5% are true labels; of 0.3-0.4, 18%. From 0.6 up it's within 0.06.
- **The zero-shot model is overconfident and answers in round numbers.** Its probabilities come in steps of 0.1, most at 0, 0.1, 0.8 and 0.9-1.0; of the points it puts at 0.8-0.9, 46% are true labels, and at 0.9-1.0, 83%. Its macro-F1 is close to the others' at the 0.5 threshold, but its probabilities don't rank cases for review.
- 73-86% of each model's points fall under 0.1, where every model is close to right, so ECE understates the gaps in the middle; the bins above show them.

### Validation to test

Both trained models lost 0.07-0.08 from validation to test (baseline 0.774 to 0.693, fine-tuned 0.766 to 0.694, all series); the zero-shot model held steady (0.667 to 0.664). The test years bring more cases in series that were rare in training (FINRA:5000 48 test cases vs 8 in validation, FINRA:6000 42 vs 7, "other" 12 vs 2). A model with no training labels doesn't depend on which series the training years happened to contain.

## Why H1 and H2 failed

Monthly summaries are short, written by FINRA staff, and name the conduct in the words the rules use: "failed to supervise", "outside business activity", "refused to appear for on-the-record testimony". With rule numbers masked, those phrases carry most of the signal, and TF-IDF over word pairs captures them as well as a fine-tuned encoder does. The spec anticipated this in its limits: the summaries' authors know the charges, so the wording leans toward them.

For H2, the zero-shot model came within 0.03 of the trained models with no labeled cases, given only a one-line description of each series. Labeled data bought 0.03 macro-F1 and better-calibrated probabilities; it also bought the ability to learn FINRA's charging conventions, which is where the zero-shot model's errors concentrate.

## H6: facts section vs. summary

The labels come from the documents' violation sentences, so the document input is its facts section with every sentence that states a violation, explains or quotes a rule, or discusses the sanction removed, and rule numbers masked (spec change log, 2026-10-11). A second ModernBERT was trained on 4,007 training cases' facts sections the same way as the summary model (three seeds averaged, 1,024 tokens) and compared on 300 test cases with a facts section, drawn at random.

| Input | Macro-F1, all series | Macro-F1, series with 10+ cases in the sample |
| --- | --- | --- |
| Monthly summary | 0.681 | 0.835 |
| Facts section | 0.614 | 0.773 |

The facts model is behind by 0.067 (paired bootstrap 95% interval -0.114 to -0.016; no resample above 0), so H6 fails in the other direction: the summary is the better input.

| Series | Sample cases | Summary | Facts |
| --- | --- | --- | --- |
| FINRA:3000 | 133 | 0.962 | 0.930 |
| FINRA:8000 | 72 | 0.972 | 0.959 |
| SEC_RULE | 43 | 0.750 | 0.627 |
| FINRA:4000 | 42 | 0.916 | 0.861 |
| FINRA:2000 | 36 | 0.737 | 0.747 |
| SEC_SECTION | 29 | 0.731 | 0.571 |
| FINRA:1000 | 17 | 0.857 | 0.600 |
| MSRB | 13 | 0.963 | 1.000 |

- **The losses are where the charge depends on drafting.** The statutes (SEC_RULE -0.12, SEC_SECTION -0.16) lose the most: the sentences that separate Section 17(a) from Rule 17a-4 explain the statute, and those were removed so the facts wouldn't name the charge. A summary keeps a phrase like "books and records requirements of the Exchange Act".
- **FINRA:1000 is over-predicted from facts** (33 predicted against 17 true). Facts sections mention Form U4, registration and statutory disqualification as background in many cases; the summary mentions them only when they were charged.
- **Where the conduct names the series, the two inputs tie:** 8210 cooperation, supervision, MSRB, trade reporting (FINRA:6000, 0.800 each).
- **The comparison isn't neutral to the document.** The facts input is what remains after removing every sentence that would give the label away, and those removals take some conduct with them. A model reading the whole document would score higher by reading the answer. What H6 shows is that FINRA's one-paragraph summary carries at least as much usable signal as the stripped narrative it was written from.

## Retrieval and answers (H3, H4, H5)

The index holds the 657 FINRA rules with text of their own (2,399 chunks of 250 words); case queries search the conduct rules (0100-8000), the only part any validation case was charged under. The query is the masked summary without its sanction and procedure sentences. A local model (gemma4:26b) answers from the top 5 rules' best chunks, citing rules or declining. Every design choice was made on validation and logged before the test run: the query cleanup, one chunk per rule, the conduct-rule filter, BM25 over hybrid, and answer prompt v2 of three.

### H3: retrieval, test split

| Retriever | A charged rule in the top 5 | MRR |
| --- | --- | --- |
| BM25 (judged) | **0.713** | 0.512 |
| Hybrid (BM25 + nomic-embed-text, reciprocal rank fusion) | 0.672 | 0.483 |
| Dense (nomic-embed-text) | 0.463 | 0.301 |

950 of 1,143 test cases were charged with a FINRA Rulebook rule other than 2010 and are scored. H3 is met by 1.3 points; validation was 0.644, so the margin is thin. By series (hybrid): trade reporting 0.91, offerings and trading 0.90, supervision 0.80, suitability and communications 0.77, investigations (8210) 0.62, books and records and net capital 0.54, registration 0.52. Four test charges were under the Uniform Practice and arbitration codes, which a conduct-rule search can't return.

- **Word matching beat embeddings at every step.** Summaries and rules share their key words ("supervise", "outside business activity"), and fusing in the weaker dense ranking lowered BM25's result. A larger embedding model (bge-m3) raised dense slightly and lowered hybrid.
- **The summary's sanction sentence was the biggest single problem.** The first validation run retrieved FINRA's rules on sanctions (8310, 8320, BrokerCheck disclosure) because every summary opens with "censured and fined"; removing those sentences took hybrid from 0.27 to 0.51 on validation.

### H4: out-of-scope questions

All 20 out-of-scope questions were declined (pass line 18), including the four near misses where retrieval surfaced a related FINRA rule (2241, 5123, the arbitration codes). The same system declines 43 of 100 in-scope test cases, so H4 shows the decline works when it should; it doesn't show the system answers when it should.

### H5: the LLM judge

| | Judge: correct | Judge: incorrect |
| --- | --- | --- |
| Hand: correct | 38 | 13 |
| Hand: incorrect | 0 | 49 |

Cohen's κ 0.741, agreement 0.87, on 100 random test answers. Every disagreement goes one way: the judge called 13 answers incorrect that the hand labels called correct, and never the reverse, so the judge's correct rate (0.38) understates the hand-labeled rate (0.51).

**How the hand labels were made.** An assistant model (Claude, not the local judge) drafted all 100 labels blind to the judge's labels, with a written reason per row; the author reviewed every row against the case summary and changed none. Six close calls were flagged for review and kept as drafted: a cited rule of the right kind for a different product (6140 for Treasury spoofing, 2220 for promissory-note communications, 6622 for TRACE reporting) or a related rule (5122, 5310, 3280) counted as related, not unrelated. On a strict reading those six would be incorrect. So H5 measures agreement between the local judge and a second model's labels confirmed by a person; a person labeling from scratch could land differently.

### What the answers are worth

On the 100 test cases, 43 answers declined and 55 cited a charged rule. Of the 57 that answered, 51 cited a charged rule and nothing unrelated. Retrieval put a charged rule in front of the model for 71% of cases, so the answer step loses about a quarter of the cases retrieval got right, nearly all to declines: the model declines when no excerpt covers every part of the conduct, and three prompt versions didn't change that (spec change log).

## Labels

- 6,111 of 6,198 cases have labels: violation sentences of the full case document, then the monthly summary's citations (20 cases), then OCR of scanned or garbled PDFs (114 cases).
- The 0.98 hand-check gate on rule-level labels was missed and accepted (spec change log): four 100-case checks scored precision 0.881, 0.907, 0.973, 0.936 and recall 0.966, 0.944, 1.000, 0.981, each round's errors turned into extractor rules. At the series level the classifiers use, 3-5% of checked cases carry a wrong series.
- NASD rules are mapped to their FINRA successors before series labels are built (NASD 3010 to FINRA 3110, and so on). Without the map, one obligation split into two classes by conduct date, and the NASD classes scored F1 0.24-0.53 on validation.

## Limits

- **Label noise** of 3-5% of cases at the series level is about the size of the differences between models.
- **Summaries only.** The classifiers read the monthly-report summary, written after the case by staff who know the charges; scores are an upper bound for classifying a complaint or tip (stated in advance). H6 tested the document's facts section instead and found it worse, not better.
- **Facts extraction is rule-based.** Five extraction rounds, each checked on printed samples before training; about 3 of 5,493 facts texts keep sanction boilerplate the OCR misread, and 204 cases were left out as too short.
- **One zero-shot model and prompt.** Series descriptions were written by hand from the Rulebook's table of contents (the Rulebook download is pending); prompt v2 was chosen on validation over v1.
- **One fine-tuned architecture,** three seeds averaged; no hyperparameter search beyond positive-class weights and epoch count.
