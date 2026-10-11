# Retrieval (H3), validation split

414 cases; 362 charged at least one FINRA Rulebook rule other than 2010 and are scored (52 charged only SEC, MSRB or other rules and are left out). 0 scored cases have no charged rule among the 657 rules indexed. Embeddings: bge-m3. Query: the masked summary without its sanction and procedure sentences. Searched: the conduct rules (0100-8000). Hit: a charged rule is among the top 5 rules (each rule by its best chunk).

Charged rules by part of the Rulebook: conduct rules (0100-8000) 491, Code of Procedure (9000) 0, Uniform Practice Code (11000) 0, arbitration and mediation codes (12000-14000) 0.

| Retriever | Hit rate, top 5 | MRR |
| --- | --- | --- |
| bm25 | 0.644 | 0.471 |
| dense | 0.401 | 0.288 |
| hybrid | 0.564 | 0.422 |

## Hybrid hit rate by summary length

| length   |   cases |   hybrid_hit |
|:---------|--------:|-------------:|
| short    |     121 |        0.471 |
| medium   |     120 |        0.617 |
| long     |     121 |        0.603 |

## Hybrid hit rate by charged series

| series     |   cases |   hybrid_hit |
|:-----------|--------:|-------------:|
| FINRA:1000 |      40 |        0.525 |
| FINRA:2000 |      61 |        0.705 |
| FINRA:3000 |     157 |        0.771 |
| FINRA:4000 |      70 |        0.343 |
| FINRA:5000 |       8 |        0.875 |
| FINRA:6000 |       7 |        0.857 |
| FINRA:7000 |       2 |        0     |
| FINRA:8000 |     109 |        0.422 |

## What fills the top 5

Share of all top-5 slots by part of the Rulebook:

| Part | bm25 | dense | hybrid |
| --- | --- | --- | --- |
| conduct rules (0100-8000) | 1.000 | 1.000 | 1.000 |
| Code of Procedure (9000) | 0.000 | 0.000 | 0.000 |
| Uniform Practice Code (11000) | 0.000 | 0.000 | 0.000 |
| arbitration and mediation codes (12000-14000) | 0.000 | 0.000 | 0.000 |

Rules retrieved most often (share of scored cases with the rule in their top 5):

- **bm25:** 3110 31%, 8312 23%, 2330 22%, 8210 19%, 2360 18%, 1240 15%, 1220 13%, 1013 13%, 4111 12%, 4530 11%
- **dense:** 2360 22%, 8312 22%, 4530 20%, 3110 17%, 2370 14%, 7380 12%, 5320 12%, 5110 12%, 5122 12%, 2231 11%
- **hybrid:** 3110 30%, 8312 27%, 2360 25%, 4530 19%, 5110 17%, 2370 14%, 1013 13%, 8210 13%, 4111 13%, 2330 12%
