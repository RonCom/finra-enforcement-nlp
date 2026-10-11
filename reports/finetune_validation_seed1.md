# Fine-tuned transformer, validation split

Model answerdotai/ModernBERT-base, 6 epochs (best: epoch 4), batch 16, learning rate 5e-05, positive weights sqrt(neg/pos), capped at 10, max 512 tokens (7.2% of training summaries cut), seed 1, device cuda. Train cases: 4568. Validation cases: 414. Threshold 0.5.

Macro-F1: 0.728 (all 15 series)

Macro-F1, series with 10+ validation cases: 0.815 (9 series)

| series      |   support |   predicted |    f1 |   brier |
|:------------|----------:|------------:|------:|--------:|
| FINRA:1000  |        40 |          39 | 0.785 |   0.034 |
| FINRA:2000  |        61 |          70 | 0.824 |   0.039 |
| FINRA:3000  |       157 |         170 | 0.911 |   0.053 |
| FINRA:4000  |        70 |          73 | 0.839 |   0.044 |
| FINRA:5000  |         8 |           7 | 0.8   |   0.008 |
| FINRA:6000  |         7 |           5 | 0.833 |   0.004 |
| FINRA:7000  |         2 |           5 | 0.571 |   0.004 |
| FINRA:8000  |       109 |         109 | 0.972 |   0.011 |
| MSRB        |        11 |           8 | 0.842 |   0.008 |
| REG_NMS     |         8 |           8 | 0.625 |   0.009 |
| REG_SHO     |         4 |           4 | 0.75  |   0.002 |
| SA_SECTION  |        10 |           5 | 0.667 |   0.012 |
| SEC_RULE    |        52 |          52 | 0.788 |   0.045 |
| SEC_SECTION |        44 |          41 | 0.706 |   0.052 |
| other       |         2 |           0 | 0     |   0.005 |

## By epoch

|   epoch |   train_loss |   macro_f1 |   macro_f1_10plus |
|--------:|-------------:|-----------:|------------------:|
|       1 |       0.3905 |     0.5494 |            0.6741 |
|       2 |       0.1871 |     0.6201 |            0.7562 |
|       3 |       0.1006 |     0.6801 |            0.7605 |
|       4 |       0.039  |     0.7277 |            0.815  |
|       5 |       0.013  |     0.7328 |            0.8043 |
|       6 |       0.0047 |     0.7167 |            0.8039 |
