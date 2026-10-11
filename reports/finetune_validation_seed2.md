# Fine-tuned transformer, validation split

Model answerdotai/ModernBERT-base, 6 epochs (best: epoch 4), batch 16, learning rate 5e-05, positive weights sqrt(neg/pos), capped at 10, max 512 tokens (7.7% of training summaries cut), seed 2, device cuda. Train cases: 4568. Validation cases: 414. Threshold 0.5.

Macro-F1: 0.683 (all 15 series)

Macro-F1, series with 10+ validation cases: 0.789 (9 series)

| series      |   support |   predicted |    f1 |   brier |
|:------------|----------:|------------:|------:|--------:|
| FINRA:1000  |        40 |          35 | 0.827 |   0.022 |
| FINRA:2000  |        61 |          50 | 0.847 |   0.034 |
| FINRA:3000  |       157 |         156 | 0.92  |   0.052 |
| FINRA:4000  |        70 |          70 | 0.871 |   0.036 |
| FINRA:5000  |         8 |           6 | 0.714 |   0.008 |
| FINRA:6000  |         7 |           5 | 0.833 |   0.004 |
| FINRA:7000  |         2 |           0 | 0     |   0.005 |
| FINRA:8000  |       109 |         115 | 0.955 |   0.02  |
| MSRB        |        11 |           8 | 0.842 |   0.007 |
| REG_NMS     |         8 |          12 | 0.7   |   0.01  |
| REG_SHO     |         4 |           5 | 0.889 |   0.003 |
| SA_SECTION  |        10 |           9 | 0.526 |   0.018 |
| SEC_RULE    |        52 |          62 | 0.702 |   0.067 |
| SEC_SECTION |        44 |          44 | 0.614 |   0.066 |
| other       |         2 |           0 | 0     |   0.006 |

## By epoch

|   epoch |   train_loss |   macro_f1 |   macro_f1_10plus |
|--------:|-------------:|-----------:|------------------:|
|       1 |       0.395  |     0.4558 |            0.6275 |
|       2 |       0.1953 |     0.6292 |            0.7263 |
|       3 |       0.1096 |     0.6313 |            0.7328 |
|       4 |       0.0482 |     0.6827 |            0.7894 |
|       5 |       0.0193 |     0.6767 |            0.7746 |
|       6 |       0.0067 |     0.6954 |            0.7809 |
