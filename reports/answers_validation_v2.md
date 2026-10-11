# Answers, validation split

50 cases charged with a FINRA Rulebook rule; model gemma4:26b, answer prompt v2, judge prompt v1. Rule 2010 is left out of the counts.

| Measure | Value |
| --- | --- |
| Declined | 0.440 |
| A charged rule among the 5 retrieved chunks | 0.620 |
| At least one charged rule cited | 0.460 |
| Cited rules that FINRA charged (precision) | 0.558 |
| Charged rules cited (recall) | 0.348 |
| Citations dropped as not in the excerpts | 1 |
| Judge: correct | 0.360 (50 judged) |

Declined although a charged rule was among the excerpts: 7 of 22 declines.

## Declines with a charged rule in the excerpts

- **2016051927501** (charged FINRA:6730, FINRA:2010, MSRB:G-14, FINRA:3110, MSRB:G-27; excerpts 6730, 6181, 6623, 6622, 7730): I cannot answer this question because the excerpts provided do not contain the rules governing MSRB reporting or the specific supervisory requirements mentioned in the conduct described.
- **2017052494701** (charged NASD:3010, FINRA:3110, FINRA:2010, SEC_SECTION:17, SEC_RULE:17a-3, FINRA:4511, SEC_RULE:15l-1, FINRA:2111; excerpts 2270, 2330, 2111, 2264, 2242): I cannot answer this question because the provided excerpts do not contain the rules necessary to address the conduct described, such as rules regarding supervision, books and records, or the specific obligations related to Listed BDCs and Reg BI.
- **2019063686204** (charged FINRA:3110, FINRA:2010; excerpts 2330, 2111, 2360, 3110, 1220): I cannot answer this question because the provided excerpts do not contain the specific rule numbers that were removed from the description of the conduct.
- **2019063946701** (charged FINRA:8210, FINRA:2010; excerpts 2111, 8210, 2214, 2114, 2121): The provided excerpts do not contain a rule that specifies the obligations or requirements regarding a person's duty to appear for on-the-record testimony requested by FINRA in connection with an investigation.
- **2020067785301** (charged FINRA:3310, FINRA:2010, FINRA:3110, SA_SECTION:5; excerpts 3310, 4210, 3110, 3120, 2242): I cannot answer this question because the provided excerpts do not contain the specific rules required to address the conduct described, such as rules regarding Customer Identification Programs (CIP) or the resale of unregistered securities.
- **2021071665401** (charged FINRA:8210, FINRA:2010; excerpts 1220, 2210, 8210, 3270, 4530): I cannot answer this question because the provided excerpts do not contain the rules governing the provision of false information to FINRA or the requirements for disclosing outside business activities.
- **2022075867001** (charged FINRA:8210, FINRA:2010; excerpts 8210, 1240, 6432, 2241, 2251): The provided excerpts do not contain a rule that addresses a person's refusal to provide on-the-record testimony or a failure to cooperate with a FINRA investigation.

## Answers citing no charged rule (5)

- **2020068469601** (charged FINRA:2210, FINRA:3110, FINRA:4511, FINRA:2010; cited 2231, 3230; excerpts 2231, 4210, 2264, 2210, 3230)
- **2021069375301** (charged FINRA:4511, FINRA:2010; cited 3110, 2210; excerpts 1013, 3110, 2330, 2210, 2231)
- **2022076673601** (charged FINRA:1122, FINRA:2010; cited 4530, 1013; excerpts 4530, 1013, 4111, 1240, 8312)
- **2022077233201** (charged FINRA:1122, FINRA:2010; cited 1010, 3110; excerpts 8312, 1010, 3110, 2360, 1013)
- **2023078087101** (charged FINRA:1122, FINRA:2010, FINRA:2020; cited 1010, 4111, 4530; excerpts 1010, 2210, 2121, 4111, 4530)
