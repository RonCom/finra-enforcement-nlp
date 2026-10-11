# Fine-tuned transformer, validation split, input: facts

Model answerdotai/ModernBERT-base, 6 epochs (best: epoch 3), batch 16, learning rate 5e-05, positive weights sqrt(neg/pos), capped at 10, max 1024 tokens (13.6% of training summaries cut), seed 1, device cuda. Train cases: 4007. Validation cases: 392. Threshold 0.5.

Macro-F1: 0.634 (all 15 series)

Macro-F1, series with 10+ validation cases: 0.737 (9 series)

| series      |   support |   predicted |    f1 |   brier |
|:------------|----------:|------------:|------:|--------:|
| FINRA:1000  |        40 |          45 | 0.706 |   0.056 |
| FINRA:2000  |        60 |          95 | 0.697 |   0.094 |
| FINRA:3000  |       152 |         128 | 0.85  |   0.084 |
| FINRA:4000  |        68 |          77 | 0.828 |   0.043 |
| FINRA:5000  |         6 |           6 | 0.667 |   0.01  |
| FINRA:6000  |         7 |           7 | 0.714 |   0.006 |
| FINRA:7000  |         2 |           2 | 0.5   |   0.004 |
| FINRA:8000  |        96 |         101 | 0.954 |   0.02  |
| MSRB        |        11 |           9 | 0.9   |   0.006 |
| REG_NMS     |         8 |          12 | 0.5   |   0.018 |
| REG_SHO     |         3 |           9 | 0.5   |   0.011 |
| SA_SECTION  |        10 |           7 | 0.471 |   0.021 |
| SEC_RULE    |        51 |          70 | 0.645 |   0.08  |
| SEC_SECTION |        43 |          53 | 0.583 |   0.071 |
| other       |         2 |           0 | 0     |   0.005 |

## By epoch

|   epoch |   train_loss |   macro_f1 |   macro_f1_10plus |
|--------:|-------------:|-----------:|------------------:|
|       1 |       0.4076 |     0.4927 |            0.564  |
|       2 |       0.2307 |     0.5623 |            0.6686 |
|       3 |       0.1348 |     0.6343 |            0.737  |
|       4 |       0.0596 |     0.6438 |            0.7    |
|       5 |       0.0223 |     0.6326 |            0.7105 |
|       6 |       0.0089 |     0.6239 |            0.7085 |
