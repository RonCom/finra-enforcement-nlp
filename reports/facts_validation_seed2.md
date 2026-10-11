# Fine-tuned transformer, validation split, input: facts

Model answerdotai/ModernBERT-base, 6 epochs (best: epoch 3), batch 16, learning rate 5e-05, positive weights sqrt(neg/pos), capped at 10, max 1024 tokens (14.3% of training summaries cut), seed 2, device cuda. Train cases: 4007. Validation cases: 392. Threshold 0.5.

Macro-F1: 0.653 (all 15 series)

Macro-F1, series with 10+ validation cases: 0.725 (9 series)

| series      |   support |   predicted |    f1 |   brier |
|:------------|----------:|------------:|------:|--------:|
| FINRA:1000  |        40 |          43 | 0.675 |   0.052 |
| FINRA:2000  |        60 |          62 | 0.787 |   0.053 |
| FINRA:3000  |       152 |         159 | 0.894 |   0.066 |
| FINRA:4000  |        68 |          63 | 0.855 |   0.037 |
| FINRA:5000  |         6 |           6 | 0.667 |   0.01  |
| FINRA:6000  |         7 |           3 | 0.6   |   0.007 |
| FINRA:7000  |         2 |           3 | 0.4   |   0.005 |
| FINRA:8000  |        96 |         103 | 0.955 |   0.022 |
| MSRB        |        11 |           8 | 0.842 |   0.007 |
| REG_NMS     |         8 |          12 | 0.6   |   0.018 |
| REG_SHO     |         3 |           3 | 1     |   0     |
| SA_SECTION  |        10 |           7 | 0.471 |   0.018 |
| SEC_RULE    |        51 |          33 | 0.571 |   0.07  |
| SEC_SECTION |        43 |          28 | 0.479 |   0.073 |
| other       |         2 |           0 | 0     |   0.005 |

## By epoch

|   epoch |   train_loss |   macro_f1 |   macro_f1_10plus |
|--------:|-------------:|-----------:|------------------:|
|       1 |       0.4176 |     0.4278 |            0.5734 |
|       2 |       0.2178 |     0.6336 |            0.7211 |
|       3 |       0.1292 |     0.653  |            0.7254 |
|       4 |       0.0599 |     0.6328 |            0.7156 |
|       5 |       0.0218 |     0.6474 |            0.6875 |
|       6 |       0.0079 |     0.6249 |            0.6896 |
