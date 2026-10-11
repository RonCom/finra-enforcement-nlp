# Test split results (2024-2026)

Code: freeze-2 at b947368. Test cases: 1143. Threshold 0.5.

Labels: the hand-checks put 3-5% of cases with a wrong series (spec change log); read differences under about 0.02 against that.

|                                   |   macro-F1, all series |   macro-F1, series with 10+ test cases |
|:----------------------------------|-----------------------:|---------------------------------------:|
| TF-IDF baseline                   |                  0.693 |                                  0.714 |
| Fine-tuned (average of 3 seeds)   |                  0.694 |                                  0.715 |
| Zero-shot (gemma4:26b, prompt v2) |                  0.664 |                                  0.691 |

| ID | Test | Expectation | Result | Verdict |
| --- | --- | --- | --- | --- |
| H1 | Fine-tuned vs. baseline, test macro-F1 | Gain >= 0.05 | +0.000 (0.694 vs 0.693) | Failed |
| H2 | Zero-shot vs. fine-tuned, test macro-F1 | Zero-shot lower by >= 0.1 | +0.030 (0.664 vs 0.694) | Failed: within 0.10, labeled data adds little |

Zero-shot prompt chosen on validation (macro-F1 over 10+ series): v1 0.672, v2 0.707.

## TF-IDF baseline

| series      |   support |   predicted |    f1 |   brier |
|:------------|----------:|------------:|------:|--------:|
| FINRA:1000  |        56 |          45 | 0.673 |   0.026 |
| FINRA:2000  |       176 |         208 | 0.729 |   0.074 |
| FINRA:3000  |       509 |         518 | 0.952 |   0.042 |
| FINRA:4000  |       182 |         168 | 0.869 |   0.043 |
| FINRA:5000  |        48 |          37 | 0.753 |   0.02  |
| FINRA:6000  |        42 |          32 | 0.865 |   0.008 |
| FINRA:7000  |         2 |           8 | 0.4   |   0.006 |
| FINRA:8000  |       286 |         261 | 0.947 |   0.021 |
| MSRB        |        45 |          37 | 0.829 |   0.014 |
| REG_NMS     |        21 |          20 | 0.878 |   0.004 |
| REG_SHO     |        10 |           5 | 0.667 |   0.006 |
| SA_SECTION  |        29 |          20 | 0.571 |   0.015 |
| SEC_RULE    |       194 |         195 | 0.581 |   0.095 |
| SEC_SECTION |       130 |         138 | 0.687 |   0.06  |
| other       |        12 |           0 | 0     |   0.009 |

## Fine-tuned (average of 3 seeds)

| series      |   support |   predicted |    f1 |   brier |
|:------------|----------:|------------:|------:|--------:|
| FINRA:1000  |        56 |          60 | 0.69  |   0.035 |
| FINRA:2000  |       176 |         188 | 0.747 |   0.06  |
| FINRA:3000  |       509 |         499 | 0.952 |   0.033 |
| FINRA:4000  |       182 |         163 | 0.852 |   0.034 |
| FINRA:5000  |        48 |          31 | 0.709 |   0.016 |
| FINRA:6000  |        42 |          31 | 0.849 |   0.008 |
| FINRA:7000  |         2 |           8 | 0.4   |   0.004 |
| FINRA:8000  |       286 |         272 | 0.953 |   0.02  |
| MSRB        |        45 |          38 | 0.867 |   0.009 |
| REG_NMS     |        21 |          20 | 0.878 |   0.004 |
| REG_SHO     |        10 |           7 | 0.706 |   0.004 |
| SA_SECTION  |        29 |          16 | 0.533 |   0.015 |
| SEC_RULE    |       194 |         153 | 0.657 |   0.087 |
| SEC_SECTION |       130 |         102 | 0.612 |   0.057 |
| other       |        12 |           0 | 0     |   0.01  |

## Zero-shot (gemma4:26b, prompt v2)

| series      |   support |   predicted |    f1 |   brier |
|:------------|----------:|------------:|------:|--------:|
| FINRA:1000  |        56 |         143 | 0.452 |   0.072 |
| FINRA:2000  |       176 |         283 | 0.623 |   0.12  |
| FINRA:3000  |       509 |         597 | 0.897 |   0.075 |
| FINRA:4000  |       182 |         185 | 0.85  |   0.039 |
| FINRA:5000  |        48 |          29 | 0.519 |   0.026 |
| FINRA:6000  |        42 |          42 | 0.905 |   0.007 |
| FINRA:7000  |         2 |           5 | 0.286 |   0.004 |
| FINRA:8000  |       286 |         273 | 0.959 |   0.02  |
| MSRB        |        45 |          39 | 0.905 |   0.007 |
| REG_NMS     |        21 |          19 | 0.95  |   0.002 |
| REG_SHO     |        10 |           7 | 0.706 |   0.004 |
| SA_SECTION  |        29 |          37 | 0.515 |   0.02  |
| SEC_RULE    |       194 |         256 | 0.68  |   0.104 |
| SEC_SECTION |       130 |         125 | 0.494 |   0.098 |
| other       |        12 |          16 | 0.214 |   0.017 |
