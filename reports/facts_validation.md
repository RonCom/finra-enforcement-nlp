# Fine-tuned transformer, validation split, input: facts

Model answerdotai/ModernBERT-base, 6 epochs (best: epoch 6), batch 16, learning rate 5e-05, positive weights sqrt(neg/pos), capped at 10, max 1024 tokens (12.7% of training summaries cut), seed 0, device cuda. Train cases: 4007. Validation cases: 392. Threshold 0.5.

Macro-F1: 0.687 (all 15 series)

Macro-F1, series with 10+ validation cases: 0.719 (9 series)

| series      |   support |   predicted |    f1 |   brier |
|:------------|----------:|------------:|------:|--------:|
| FINRA:1000  |        40 |          34 | 0.73  |   0.041 |
| FINRA:2000  |        60 |          58 | 0.814 |   0.047 |
| FINRA:3000  |       152 |         159 | 0.894 |   0.079 |
| FINRA:4000  |        68 |          64 | 0.924 |   0.024 |
| FINRA:5000  |         6 |           3 | 0.444 |   0.008 |
| FINRA:6000  |         7 |           6 | 0.923 |   0.003 |
| FINRA:7000  |         2 |           4 | 0.667 |   0.005 |
| FINRA:8000  |        96 |          96 | 0.969 |   0.014 |
| MSRB        |        11 |           8 | 0.842 |   0.007 |
| REG_NMS     |         8 |           7 | 0.8   |   0.007 |
| REG_SHO     |         3 |           3 | 1     |   0     |
| SA_SECTION  |        10 |           1 | 0.182 |   0.022 |
| SEC_RULE    |        51 |          44 | 0.695 |   0.066 |
| SEC_SECTION |        43 |          24 | 0.418 |   0.078 |
| other       |         2 |           0 | 0     |   0.005 |

## By epoch

|   epoch |   train_loss |   macro_f1 |   macro_f1_10plus |
|--------:|-------------:|-----------:|------------------:|
|       1 |       0.4202 |     0.495  |            0.6063 |
|       2 |       0.225  |     0.5859 |            0.6661 |
|       3 |       0.1314 |     0.6866 |            0.6969 |
|       4 |       0.0594 |     0.6856 |            0.7067 |
|       5 |       0.0231 |     0.6739 |            0.7089 |
|       6 |       0.0091 |     0.6867 |            0.7185 |
