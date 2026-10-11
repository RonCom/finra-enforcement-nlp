# Answers, validation split

50 cases charged with a FINRA Rulebook rule; model gemma4:26b, answer prompt v1, judge prompt v1. Rule 2010 is left out of the counts.

| Measure | Value |
| --- | --- |
| Declined | 0.540 |
| A charged rule among the 5 retrieved chunks | 0.620 |
| At least one charged rule cited | 0.400 |
| Cited rules that FINRA charged (precision) | 0.636 |
| Charged rules cited (recall) | 0.304 |
| Citations dropped as not in the excerpts | 1 |
| Judge: correct | 0.340 (50 judged) |
