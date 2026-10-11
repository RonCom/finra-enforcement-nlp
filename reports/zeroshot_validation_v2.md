# Zero-shot: local model through Ollama, validation split

Model gemma4:26b, prompt v2, reasoning off, context 4096. Validation cases: 414. Median 8.9 s per case. Threshold 0.5.

Macro-F1: 0.667 (all 15 series)

Macro-F1, series with 10+ validation cases: 0.707 (9 series)

| series      |   support |   predicted |    f1 |   brier |
|:------------|----------:|------------:|------:|--------:|
| FINRA:1000  |        40 |          47 | 0.874 |   0.021 |
| FINRA:2000  |        61 |          87 | 0.635 |   0.105 |
| FINRA:3000  |       157 |         204 | 0.853 |   0.093 |
| FINRA:4000  |        70 |          76 | 0.904 |   0.028 |
| FINRA:5000  |         8 |          10 | 0.556 |   0.014 |
| FINRA:6000  |         7 |          13 | 0.7   |   0.012 |
| FINRA:7000  |         2 |           1 | 0.667 |   0.003 |
| FINRA:8000  |       109 |         113 | 0.955 |   0.021 |
| MSRB        |        11 |           7 | 0.778 |   0.01  |
| REG_NMS     |         8 |           6 | 0.857 |   0.005 |
| REG_SHO     |         4 |           3 | 0.857 |   0.002 |
| SA_SECTION  |        10 |          14 | 0.417 |   0.027 |
| SEC_RULE    |        52 |          84 | 0.574 |   0.107 |
| SEC_SECTION |        44 |          52 | 0.375 |   0.119 |
| other       |         2 |           2 | 0     |   0.01  |
