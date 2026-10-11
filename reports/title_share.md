# Rule titles in the masked summaries, test split

Titles from 739 Rulebook rules, matched as phrases anywhere in the text, ignoring case. 950 of 1143 test cases were charged with a FINRA rule in the Rulebook (NASD rules as their FINRA successors; Rule 2010 and procedural rules left out).

**8.9% of them have the title of a charged rule in their masked summary.**

## By series

| series      |   cases |   with_title |
|:------------|--------:|-------------:|
| FINRA:1000  |      56 |        0.143 |
| FINRA:11000 |       2 |        0     |
| FINRA:13000 |       1 |        0     |
| FINRA:2000  |     176 |        0.312 |
| FINRA:3000  |     509 |        0.149 |
| FINRA:4000  |     182 |        0.126 |
| FINRA:5000  |      48 |        0.188 |
| FINRA:6000  |      39 |        0.256 |
| FINRA:7000  |       2 |        0     |
| FINRA:8000  |     286 |        0.028 |

## By rule (charged in 5+ test cases)

A title found as often in cases not charged with the rule is conduct vocabulary; one found mostly in charged cases points to the charge.

|   rule | title                                                                                                  |   charged cases |   title in summary, charged |   title in summary, not charged |
|-------:|:-------------------------------------------------------------------------------------------------------|----------------:|----------------------------:|--------------------------------:|
|   3110 | supervision                                                                                            |             329 |                       0.088 |                           0.01  |
|   8210 | provision of information and testimony and inspection and copying of books                             |             278 |                       0     |                           0     |
|   4511 | general requirements                                                                                   |             159 |                       0     |                           0     |
|   2111 | suitability                                                                                            |              73 |                       0.274 |                           0.016 |
|   3270 | outside business activities of registered persons                                                      |              61 |                       0     |                           0     |
|   3310 | anti-money laundering compliance program                                                               |              52 |                       0     |                           0     |
|   3260 | discretionary accounts                                                                                 |              38 |                       0.026 |                           0.001 |
|   3280 | private securities transactions of an associated person                                                |              38 |                       0     |                           0     |
|   2210 | communications with the public                                                                         |              33 |                       0.121 |                           0     |
|   1122 | filing of misleading information as to membership or registration                                      |              30 |                       0     |                           0     |
|   6730 | transaction reporting                                                                                  |              29 |                       0.172 |                           0.005 |
|   4110 | capital compliance                                                                                     |              25 |                       0.04  |                           0     |
|   1210 | registration requirements                                                                              |              25 |                       0.12  |                           0     |
|   3240 | prohibition on borrowing from or lending to customers                                                  |              24 |                       0     |                           0     |
|   3210 | accounts at other broker-dealers and financial institutions                                            |              17 |                       0     |                           0     |
|   2232 | customer confirmations                                                                                 |              17 |                       0.706 |                           0.001 |
|   2330 | members' responsibilities regarding deferred variable annuities                                        |              16 |                       0     |                           0     |
|   5123 | private placements of securities                                                                       |              15 |                       0     |                           0     |
|   2360 | options                                                                                                |              13 |                       0.923 |                           0.023 |
|   4530 | reporting requirements                                                                                 |              13 |                       0     |                           0.01  |
|   2150 | improper use of customers' securities or funds; prohibition against guarantees and sharing in accounts |              11 |                       0     |                           0     |
|   2040 | payments to unregistered persons                                                                       |               9 |                       0     |                           0     |
|   2121 | fair prices and commissions                                                                            |               9 |                       0     |                           0     |
|   6622 | transaction reporting                                                                                  |               8 |                       0     |                           0.01  |
|   2020 | use of manipulative, deceptive or other fraudulent devices                                             |               7 |                       0     |                           0     |
|   5310 | best execution and interpositioning                                                                    |               7 |                       0     |                           0     |
|   5110 | corporate financing rule — underwriting terms and arrangements                                         |               6 |                       0     |                           0     |
|   3120 | supervisory control system                                                                             |               6 |                       0.333 |                           0     |
|   2241 | research analysts and research reports                                                                 |               5 |                       0     |                           0     |
|   5190 | notification requirements for offering participants                                                    |               5 |                       0     |                           0     |
|   5210 | publication of transactions and quotations                                                             |               5 |                       0     |                           0     |
|   8311 | effect of a suspension, revocation, cancellation, bar or other disqualification                        |               5 |                       0     |                           0     |
