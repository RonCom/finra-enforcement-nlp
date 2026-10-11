# Fine-tuned transformer, validation split

Model answerdotai/ModernBERT-base, 6 epochs (best: epoch 5), batch 16, learning rate 5e-05, positive weights sqrt(neg/pos), capped at 10, max 512 tokens (6.6% of training summaries cut), seed 0, device cuda. Train cases: 4568. Validation cases: 414. Threshold 0.5.

Macro-F1: 0.746 (all 15 series)

Macro-F1, series with 10+ validation cases: 0.787 (9 series)

| series      |   support |   predicted |    f1 |   brier |
|:------------|----------:|------------:|------:|--------:|
| FINRA:1000  |        40 |          33 | 0.795 |   0.028 |
| FINRA:2000  |        61 |          51 | 0.839 |   0.037 |
| FINRA:3000  |       157 |         161 | 0.931 |   0.042 |
| FINRA:4000  |        70 |          61 | 0.87  |   0.035 |
| FINRA:5000  |         8 |           4 | 0.667 |   0.01  |
| FINRA:6000  |         7 |           6 | 0.923 |   0.002 |
| FINRA:7000  |         2 |           3 | 0.8   |   0.002 |
| FINRA:8000  |       109 |         113 | 0.964 |   0.014 |
| MSRB        |        11 |           8 | 0.842 |   0.007 |
| REG_NMS     |         8 |           6 | 0.857 |   0.004 |
| REG_SHO     |         4 |           3 | 0.857 |   0.001 |
| SA_SECTION  |        10 |           4 | 0.571 |   0.015 |
| SEC_RULE    |        52 |          53 | 0.743 |   0.056 |
| SEC_SECTION |        44 |          28 | 0.528 |   0.072 |
| other       |         2 |           2 | 0     |   0.007 |

## By epoch

|   epoch |   train_loss |   macro_f1 |   macro_f1_10plus |
|--------:|-------------:|-----------:|------------------:|
|       1 |       0.3927 |     0.5114 |            0.6668 |
|       2 |       0.1907 |     0.6451 |            0.7259 |
|       3 |       0.1077 |     0.7191 |            0.76   |
|       4 |       0.047  |     0.7271 |            0.7836 |
|       5 |       0.0194 |     0.7458 |            0.787  |
|       6 |       0.008  |     0.725  |            0.7857 |
