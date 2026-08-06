# HCDR detailed EDA from the reference notebook

Source: `notebooks/top-voted/home-credit-default-risk/02-complete-eda-feature-importance/home-credit-complete-eda-feature-importance.ipynb`.

This output reproduces the reference notebook's analytical scope with the current
official local CSV files. Static CSV/PNG artifacts replace legacy inline Plotly and
Cufflinks output. `TARGET=1` means payment difficulty in the competition data.

## Dataset overview

- Tables analyzed: 8
- Application-train rows: 307,511
- Application-train bad rate: 8.0729%

## Highest missing percentages by table

- `application_train`: COMMONAREA_AVG (69.87%), COMMONAREA_MEDI (69.87%), COMMONAREA_MODE (69.87%)
- `application_test`: COMMONAREA_AVG (68.72%), COMMONAREA_MEDI (68.72%), COMMONAREA_MODE (68.72%)
- `POS_CASH_balance`: CNT_INSTALMENT_FUTURE (0.26%), CNT_INSTALMENT (0.26%), MONTHS_BALANCE (0.00%)
- `bureau_balance`: MONTHS_BALANCE (0.00%), SK_ID_BUREAU (0.00%), STATUS (0.00%)
- `previous_application`: RATE_INTEREST_PRIMARY (99.64%), RATE_INTEREST_PRIVILEGED (99.64%), AMT_DOWN_PAYMENT (53.64%)
- `installments_payments`: AMT_PAYMENT (0.02%), DAYS_ENTRY_PAYMENT (0.02%), AMT_INSTALMENT (0.00%)
- `credit_card_balance`: AMT_PAYMENT_CURRENT (20.00%), AMT_DRAWINGS_ATM_CURRENT (19.52%), AMT_DRAWINGS_OTHER_CURRENT (19.52%)
- `bureau`: AMT_ANNUITY (71.47%), AMT_CREDIT_MAX_OVERDUE (65.51%), DAYS_ENDDATE_FACT (36.92%)

## Strongest numeric Pearson correlations with TARGET

`EXT_SOURCE_3` (-0.1789), `EXT_SOURCE_2` (-0.1605), `EXT_SOURCE_1` (-0.1553), `DAYS_BIRTH` (+0.0782), `REGION_RATING_CLIENT_W_CITY` (+0.0609), `REGION_RATING_CLIENT` (+0.0589), `DAYS_LAST_PHONE_CHANGE` (+0.0552), `DAYS_ID_PUBLISH` (+0.0515), `REG_CITY_NOT_WORK_CITY` (+0.0510), `FLAG_EMP_PHONE` (+0.0460)

## Random Forest importance

`EXT_SOURCE_2` (0.3389), `EXT_SOURCE_3` (0.3040), `EXT_SOURCE_1` (0.0718), `DAYS_BIRTH` (0.0474), `DAYS_EMPLOYED` (0.0204), `NAME_EDUCATION_TYPE` (0.0163), `DAYS_ID_PUBLISH` (0.0144), `AMT_ANNUITY` (0.0130), `CODE_GENDER` (0.0129), `AMT_GOODS_PRICE` (0.0122)

## Interpretation limits

- This is descriptive benchmark EDA, not evidence of production suitability.
- Correlation and impurity-based importance are associative, not causal.
- Random Forest importance can favor continuous or high-cardinality features.
- The reference notebook's category-by-target charts divide each target count by
  category population; this reproduction reports the equivalent value explicitly as
  `bad_rate` and preserves category support counts.
