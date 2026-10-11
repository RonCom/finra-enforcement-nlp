# H6: facts section vs. monthly summary as input (test split)

Code: freeze-3 at 96a1419. 300 test cases with a facts section, drawn at random (seed 6) from 1094. Both models are ModernBERT-base, fine-tuned the same way, average of 3 seeds; one read summaries (stored predictions from the test run), one read facts sections (max 1,024 tokens). Threshold 0.5.

| Input | Macro-F1, all series | Macro-F1, series with 10+ cases in the sample |
| --- | --- | --- |
| Monthly summary | 0.681 | 0.835 |
| Facts section | 0.614 | 0.773 |

| ID | Test | Expectation | Result | Verdict |
| --- | --- | --- | --- | --- |
| H6 | Facts section vs. summary, macro-F1 | Gain >= 0.03 | -0.067 (95% interval -0.114 to -0.016, 0% of resamples above 0) | Failed: summaries are enough for triage |

## Summary input

| series      |   support |   predicted |    f1 |   brier |
|:------------|----------:|------------:|------:|--------:|
| FINRA:1000  |        17 |          18 | 0.857 |   0.026 |
| FINRA:2000  |        36 |          40 | 0.737 |   0.048 |
| FINRA:3000  |       133 |         131 | 0.962 |   0.029 |
| FINRA:4000  |        42 |          41 | 0.916 |   0.021 |
| FINRA:5000  |        14 |           7 | 0.667 |   0.019 |
| FINRA:6000  |        12 |           8 | 0.8   |   0.009 |
| FINRA:7000  |         0 |           3 | 0     |   0.006 |
| FINRA:8000  |        72 |          70 | 0.972 |   0.012 |
| MSRB        |        13 |          14 | 0.963 |   0.004 |
| REG_NMS     |         6 |           6 | 0.833 |   0.004 |
| REG_SHO     |         4 |           2 | 0.667 |   0.007 |
| SA_SECTION  |         9 |           2 | 0.364 |   0.022 |
| SEC_RULE    |        43 |          37 | 0.75  |   0.058 |
| SEC_SECTION |        29 |          23 | 0.731 |   0.041 |
| other       |         4 |           0 | 0     |   0.012 |

## Facts input

| series      |   support |   predicted |    f1 |   brier |
|:------------|----------:|------------:|------:|--------:|
| FINRA:1000  |        17 |          33 | 0.6   |   0.044 |
| FINRA:2000  |        36 |          47 | 0.747 |   0.064 |
| FINRA:3000  |       133 |         125 | 0.93  |   0.048 |
| FINRA:4000  |        42 |          37 | 0.861 |   0.037 |
| FINRA:5000  |        14 |           8 | 0.636 |   0.024 |
| FINRA:6000  |        12 |           8 | 0.8   |   0.01  |
| FINRA:7000  |         0 |           2 | 0     |   0.004 |
| FINRA:8000  |        72 |          74 | 0.959 |   0.016 |
| MSRB        |        13 |          13 | 1     |   0.001 |
| REG_NMS     |         6 |           7 | 0.615 |   0.01  |
| REG_SHO     |         4 |           4 | 0.5   |   0.012 |
| SA_SECTION  |         9 |           2 | 0.364 |   0.027 |
| SEC_RULE    |        43 |          40 | 0.627 |   0.07  |
| SEC_SECTION |        29 |          20 | 0.571 |   0.051 |
| other       |         4 |           0 | 0     |   0.013 |
