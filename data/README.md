# Data

## survey_bangladesh.csv

Collected by the authors. 352 responses via an online questionnaire, North South University
ethics approval, written informed consent obtained. No names, contact details,
IP addresses, dates or free text were collected, so the file as distributed is
the file as analysed.

Two context items, then sixteen 5-point Likert items in four blocks:

| Prefix | Construct | Items |
|---|---|---|
| `PB` | Purchase Behavior (outcome) | PB1-PB4 |
| `ATTD` | Attitude | ATTD1-ATTD4 |
| `SN` | Subjective Norm | SN1-SN4 |
| `PBC` | Perceived Behavioral Control | PBC1-PBC4 |

The outcome is the mean of PB1-PB4 binarised at 2.5, the scale midpoint.
`PBC` items are predictors and must not enter the outcome. Because `PBC1`
also starts with `PB`, selecting outcome columns by prefix silently captures
them; `cogdml.data` handles this in one place and everything imports it.

## external_north_india.xlsx

253 responses, green purchase behaviour, North India. Used once, for the
transferability check. Collected by Rakesh Kumar and published as *Dataset of college student's
green purchase intention and behavior*, Mendeley Data V1, 14 November 2022,
doi:10.17632/b9hdy2swyy.1. Not redistributed here; `fetch_external.sh`
downloads it from Mendeley.

Rescaled from 7 points to 5, `SN4` and `PBC4` have no counterpart and are
imputed with training-set means, outcome binarised at GPB >= 4.0.

The two samples differ in geography, product domain, instrument and scale at
the same time, so the check cannot attribute any performance change to a
single cause.
