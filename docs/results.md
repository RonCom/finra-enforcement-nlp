# Results

Scored against the expectations in [spec.md](spec.md). The classifiers were tuned on the validation split (2023) and scored once on the test split (2024-2026, 1,143 cases) with code tagged `freeze-2` (commit b947368); the full output is `reports/test_results.md`. Every rule changed after data was seen is in the spec's change log, dated, with the reason. The retrieval part (H3, H4, H5) and the full-document comparison (H6) are not run yet.

## Expectations

| ID | Test | Expectation | Result | Verdict |
| --- | --- | --- | --- | --- |
| H1 | Fine-tuned transformer vs. TF-IDF baseline, test macro-F1 | Gain ≥ 0.05 | +0.000 (0.694 vs 0.693) | Failed |
| H2 | Zero-shot local model vs. fine-tuned transformer, test macro-F1 | Zero-shot lower by ≥ 0.10 | Lower by 0.030 (0.664 vs 0.694) | Failed: labeled data adds little here |
| H3 | Retrieval: a cited rule in the top 5 chunks | ≥ 70% | | Not run yet |
| H4 | Out-of-scope questions declined | ≥ 18 of 20 | | Not run yet |
| H5 | LLM judge vs. hand labels, Cohen's κ | ≥ 0.60 | | Not run yet |
| H6 | Full AWC vs. monthly summary as input | Gain ≥ 0.03 | | Not run yet |

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

### Validation to test

Both trained models lost 0.07-0.08 from validation to test (baseline 0.774 to 0.693, fine-tuned 0.766 to 0.694, all series); the zero-shot model held steady (0.667 to 0.664). The test years bring more cases in series that were rare in training (FINRA:5000 48 test cases vs 8 in validation, FINRA:6000 42 vs 7, "other" 12 vs 2). A model with no training labels doesn't depend on which series the training years happened to contain.

## Why H1 and H2 failed

Monthly summaries are short, written by FINRA staff, and name the conduct in the words the rules use: "failed to supervise", "outside business activity", "refused to appear for on-the-record testimony". With rule numbers masked, those phrases carry most of the signal, and TF-IDF over word pairs captures them as well as a fine-tuned encoder does. The spec anticipated this in its limits: the summaries' authors know the charges, so the wording leans toward them.

For H2, the zero-shot model came within 0.03 of the trained models with no labeled cases, given only a one-line description of each series. Labeled data bought 0.03 macro-F1 and better-calibrated probabilities; it also bought the ability to learn FINRA's charging conventions, which is where the zero-shot model's errors concentrate.

## Labels

- 6,111 of 6,198 cases have labels: violation sentences of the full case document, then the monthly summary's citations (20 cases), then OCR of scanned or garbled PDFs (114 cases).
- The 0.98 hand-check gate on rule-level labels was missed and accepted (spec change log): four 100-case checks scored precision 0.881, 0.907, 0.973, 0.936 and recall 0.966, 0.944, 1.000, 0.981, each round's errors turned into extractor rules. At the series level the classifiers use, 3-5% of checked cases carry a wrong series.
- NASD rules are mapped to their FINRA successors before series labels are built (NASD 3010 to FINRA 3110, and so on). Without the map, one obligation split into two classes by conduct date, and the NASD classes scored F1 0.24-0.53 on validation.

## Limits

- **Label noise** of 3-5% of cases at the series level is about the size of the differences between models.
- **Summaries only.** The classifiers read the monthly-report summary, written after the case by staff who know the charges; scores are an upper bound for classifying a complaint or tip (stated in advance).
- **One zero-shot model and prompt.** Series descriptions were written by hand from the Rulebook's table of contents (the Rulebook download is pending); prompt v2 was chosen on validation over v1.
- **One fine-tuned architecture,** three seeds averaged; no hyperparameter search beyond positive-class weights and epoch count.
