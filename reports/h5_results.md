# H5: LLM judge vs. hand labels

100 test answers (random, seed 5), labeled by hand blind to the judge. Judge: gemma4:26b, prompt v1.

| ID | Test | Expectation | Result | Verdict |
| --- | --- | --- | --- | --- |
| H5 | LLM judge vs. hand labels, Cohen's kappa | >= 0.6 | 0.741 (agreement 0.870) | Met |

Hand labels: 0.510 correct. Judge: 0.380 correct.

| hand      |   correct |   incorrect |
|:----------|----------:|------------:|
| correct   |        38 |          13 |
| incorrect |         0 |          49 |
