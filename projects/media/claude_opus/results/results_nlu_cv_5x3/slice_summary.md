# Robustness slices – media_claude

| slice          |   support |   accuracy |   precision |   recall |   f1 |
|:---------------|----------:|-----------:|------------:|---------:|-----:|
| ALL            |        17 |          1 |           1 |        1 |    1 |
| AMENITY        |         0 |          0 |         nan |      nan |  nan |
| DATEISH        |         9 |          1 |         nan |      nan |  nan |
| NOT_AMENITY    |        17 |          1 |           1 |        1 |    1 |
| NOT_DATEISH    |         8 |          1 |         nan |      nan |  nan |
| NOT_ROUTE_CODE |         9 |          1 |           1 |        1 |    1 |
| NOT_SHORT      |        10 |          1 |           1 |        1 |    1 |
| NOT_TEMPORAL   |        17 |          1 |           1 |        1 |    1 |
| ROUTE_CODE     |         8 |          1 |         nan |      nan |  nan |
| SHORT          |         7 |          1 |         nan |      nan |  nan |
| TEMPORAL       |         0 |          0 |         nan |      nan |  nan |

Notes:
- Accuracy is computed from per-row correctness.
- Macro P/R/F1 are NaN if the slice contains < 2 distinct gold intents.
- Slices with support=0 are included for completeness.
