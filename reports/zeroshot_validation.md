# Zero-shot: local model through Ollama, validation split

Model gemma4:26b, prompt v1, reasoning off, context 4096. Validation cases: 414. Median 8.5 s per case. Threshold 0.5.

Macro-F1: 0.673 (all 15 series)

Macro-F1, series with 10+ validation cases: 0.672 (9 series)

| series      |   support |   predicted |    f1 |   brier |
|:------------|----------:|------------:|------:|--------:|
| FINRA:1000  |        40 |          50 | 0.844 |   0.029 |
| FINRA:2000  |        61 |         123 | 0.576 |   0.138 |
| FINRA:3000  |       157 |         208 | 0.833 |   0.116 |
| FINRA:4000  |        70 |          75 | 0.897 |   0.029 |
| FINRA:5000  |         8 |           8 | 0.625 |   0.011 |
| FINRA:6000  |         7 |          13 | 0.7   |   0.014 |
| FINRA:7000  |         2 |           2 | 1     |   0     |
| FINRA:8000  |       109 |         112 | 0.959 |   0.022 |
| MSRB        |        11 |           6 | 0.706 |   0.012 |
| REG_NMS     |         8 |           6 | 0.857 |   0.005 |
| REG_SHO     |         4 |           3 | 0.857 |   0.002 |
| SA_SECTION  |        10 |          10 | 0.4   |   0.024 |
| SEC_RULE    |        52 |          46 | 0.694 |   0.068 |
| SEC_SECTION |        44 |          13 | 0.14  |   0.115 |
| other       |         2 |           8 | 0     |   0.023 |
