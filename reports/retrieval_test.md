# Retrieval (H3), test split

1143 cases; 950 charged at least one FINRA Rulebook rule other than 2010 and are scored (193 charged only SEC, MSRB or other rules and are left out). 0 scored cases have no charged rule among the 657 rules indexed. Embeddings: nomic-embed-text. Query: the masked summary without its sanction and procedure sentences. Searched: the conduct rules (0100-8000). Hit: a charged rule is among the top 5 rules (each rule by its best chunk).

Charged rules by part of the Rulebook: conduct rules (0100-8000) 1450, Code of Procedure (9000) 0, Uniform Practice Code (11000) 2, arbitration and mediation codes (12000-14000) 2.

| Retriever | Hit rate, top 5 | MRR |
| --- | --- | --- |
| bm25 | 0.713 | 0.512 |
| dense | 0.463 | 0.301 |
| hybrid | 0.672 | 0.483 |

Code: freeze-4 at b459136.

| ID | Test | Expectation | Result | Verdict |
| --- | --- | --- | --- | --- |
| H3 | A charged rule among the top 5 rules, bm25 | >= 70% | 71.3% | Met |

## Hybrid hit rate by summary length

| length   |   cases |   hybrid_hit |
|:---------|--------:|-------------:|
| short    |     317 |        0.618 |
| medium   |     316 |        0.677 |
| long     |     317 |        0.719 |

## Hybrid hit rate by charged series

| series      |   cases |   hybrid_hit |
|:------------|--------:|-------------:|
| FINRA:1000  |      56 |        0.518 |
| FINRA:11000 |       2 |        0     |
| FINRA:13000 |       1 |        0     |
| FINRA:2000  |     176 |        0.773 |
| FINRA:3000  |     509 |        0.798 |
| FINRA:4000  |     182 |        0.544 |
| FINRA:5000  |      48 |        0.896 |
| FINRA:6000  |      42 |        0.905 |
| FINRA:7000  |       2 |        1     |
| FINRA:8000  |     286 |        0.619 |

## What fills the top 5

Share of all top-5 slots by part of the Rulebook:

| Part | bm25 | dense | hybrid |
| --- | --- | --- | --- |
| conduct rules (0100-8000) | 1.000 | 1.000 | 1.000 |
| Code of Procedure (9000) | 0.000 | 0.000 | 0.000 |
| Uniform Practice Code (11000) | 0.000 | 0.000 | 0.000 |
| arbitration and mediation codes (12000-14000) | 0.000 | 0.000 | 0.000 |

Rules retrieved most often (share of scored cases with the rule in their top 5):

- **bm25:** 3110 37%, 2330 27%, 8312 20%, 8210 18%, 2360 14%, 5110 14%, 3120 13%, 1240 13%, 2210 11%, 1013 11%
- **dense:** 8312 35%, 4111 26%, 3110 25%, 6490 18%, 6282 16%, 2150 12%, 1220 12%, 5320 12%, 2210 12%, 6623 12%
- **hybrid:** 8312 32%, 3110 29%, 4111 23%, 8210 19%, 1220 16%, 4530 15%, 2210 14%, 6730 12%, 1010 11%, 6282 10%
