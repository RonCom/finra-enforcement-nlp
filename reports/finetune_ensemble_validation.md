# Fine-tuned transformer, average of 3 seeds, validation split

Macro-F1: 0.766 (all 15 series)

Macro-F1, series with 10+ validation cases: 0.813 (9 series)

| series      |   support |   predicted |    f1 |   brier |
|:------------|----------:|------------:|------:|--------:|
| FINRA:1000  |        40 |          34 | 0.784 |   0.025 |
| FINRA:2000  |        61 |          56 | 0.872 |   0.029 |
| FINRA:3000  |       157 |         160 | 0.934 |   0.043 |
| FINRA:4000  |        70 |          65 | 0.889 |   0.033 |
| FINRA:5000  |         8 |           5 | 0.615 |   0.007 |
| FINRA:6000  |         7 |           5 | 0.833 |   0.003 |
| FINRA:7000  |         2 |           3 | 0.8   |   0.003 |
| FINRA:8000  |       109 |         113 | 0.964 |   0.012 |
| MSRB        |        11 |           8 | 0.842 |   0.007 |
| REG_NMS     |         8 |           7 | 0.933 |   0.005 |
| REG_SHO     |         4 |           4 | 1     |   0.001 |
| SA_SECTION  |        10 |           4 | 0.571 |   0.014 |
| SEC_RULE    |        52 |          58 | 0.782 |   0.047 |
| SEC_SECTION |        44 |          36 | 0.675 |   0.055 |
| other       |         2 |           0 | 0     |   0.006 |

## Each seed and the average

| model                      |   macro_f1 |   macro_f1_10plus |
|:---------------------------|-----------:|------------------:|
| data/models/finetune       |     0.7458 |            0.787  |
| data/models/finetune_seed1 |     0.7277 |            0.815  |
| data/models/finetune_seed2 |     0.6827 |            0.7894 |
| average                    |     0.7663 |            0.8125 |
