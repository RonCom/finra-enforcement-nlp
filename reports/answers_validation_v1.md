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

Declined although a charged rule was among the excerpts: 11 of 27 declines.

## Declines with a charged rule in the excerpts

- **2016051927501** (charged FINRA:6730, FINRA:2010, MSRB:G-14, FINRA:3110, MSRB:G-27; excerpts 6730, 6181, 6623, 6622, 7730): I cannot answer this question because the conduct involves reporting to the Municipal Securities Rulemaking Board’s (MSRB’s) Real-time Transaction Reporting System (RTRS), which is governed by MSRB rules rather than the provided FINRA rule excerpts.
- **2017052494701** (charged NASD:3010, FINRA:3110, FINRA:2010, SEC_SECTION:17, SEC_RULE:17a-3, FINRA:4511, SEC_RULE:15l-1, FINRA:2111; excerpts 2270, 2330, 2111, 2264, 2242): The provided excerpts do not contain the rules necessary to determine which rules were violated by the described conduct.
- **2018056485301** (charged NASD:3010, FINRA:3110, FINRA:2010; excerpts 3130, 2330, 3110, 1220, 2370): I cannot answer this question because the provided excerpts do not contain the specific rules violated by the conduct described.
- **2018057298701** (charged FINRA:2111, FINRA:2010, FINRA:3110; excerpts 2010, 2270, 2341, 2111, 6730): The provided excerpts do not contain the rules governing supervisory failures or the specific obligations of a firm to supervise its associated persons.
- **2019061187802** (charged FINRA:3110, FINRA:2010; excerpts 3110, 2330, 2370, 2210, 2342): The provided excerpts do not contain the specific rules governing the suitability of mutual fund transactions or the specific requirements regarding sales charge discounts and breakpoints, therefore I cannot determine which rules were violated.
- **2019063686204** (charged FINRA:3110, FINRA:2010; excerpts 2330, 2111, 2360, 3110, 1220): The provided excerpts do not contain the specific rule numbers required to identify the violations for the conduct described.
- **2020067785301** (charged FINRA:3310, FINRA:2010, FINRA:3110, SA_SECTION:5; excerpts 3310, 4210, 3110, 3120, 2242): The provided excerpts do not contain the specific rules required to address the violations regarding the resale of unregistered securities or the specific rules referenced as [RULE] in the conduct description.
- **2020068469601** (charged FINRA:2210, FINRA:3110, FINRA:4511, FINRA:2010; excerpts 2231, 4210, 2264, 2210, 3230): The provided excerpts do not contain the rules necessary to determine which FINRA rules were violated by the described conduct.

## Answers citing no charged rule (3)

- **2021069375301** (charged FINRA:4511, FINRA:2010; cited 3110, 2210; excerpts 1013, 3110, 2330, 2210, 2231)
- **2022077233201** (charged FINRA:1122, FINRA:2010; cited 1010, 3110; excerpts 8312, 1010, 3110, 2360, 1013)
- **2023078087101** (charged FINRA:1122, FINRA:2010, FINRA:2020; cited 1010, 4530; excerpts 1010, 2210, 2121, 4111, 4530)
