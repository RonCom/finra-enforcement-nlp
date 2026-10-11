# Baseline: TF-IDF + one-vs-rest logistic regression, validation split

Train cases: 4568. Validation cases: 414. Threshold 0.5.

Macro-F1: 0.774 (all 15 series)

Macro-F1, series with 10+ validation cases: 0.782 (9 series)

| series      |   support |   predicted |    f1 |   brier |
|:------------|----------:|------------:|------:|--------:|
| FINRA:1000  |        40 |          35 | 0.773 |   0.027 |
| FINRA:2000  |        61 |          63 | 0.806 |   0.05  |
| FINRA:3000  |       157 |         166 | 0.935 |   0.049 |
| FINRA:4000  |        70 |          65 | 0.904 |   0.033 |
| FINRA:5000  |         8 |           7 | 0.8   |   0.011 |
| FINRA:6000  |         7 |           5 | 0.833 |   0.005 |
| FINRA:7000  |         2 |           2 | 1     |   0.004 |
| FINRA:8000  |       109 |         105 | 0.981 |   0.011 |
| MSRB        |        11 |           8 | 0.737 |   0.012 |
| REG_NMS     |         8 |           7 | 0.933 |   0.005 |
| REG_SHO     |         4 |           4 | 1     |   0.003 |
| SA_SECTION  |        10 |           6 | 0.625 |   0.014 |
| SEC_RULE    |        52 |          50 | 0.667 |   0.062 |
| SEC_SECTION |        44 |          35 | 0.608 |   0.056 |
| other       |         2 |           0 | 0     |   0.005 |
