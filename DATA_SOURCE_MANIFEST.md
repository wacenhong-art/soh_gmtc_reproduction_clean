# Data Source Manifest — Split-independent readiness

**Scope:** official raw-data acquisition and provenance only. This manifest assigns no train, validation, test, first-20%, or GAN-target role.

## Official archives acquired on 2026-09-11

| DATASET | BATTERY_ID | SOURCE_URL | SOURCE_ORGANIZATION | FILE_NAME | FILE_SIZE_BYTES | SHA256 | RAW_OR_PROCESSED | LICENSE_OR_ACCESS_NOTE |
|---|---|---|---|---|---:|---|---|---|
| NASA | B0005, B0006, B0007, B0018 | https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip | NASA Ames Prognostics Center of Excellence | `NASA_5_Battery_Data_Set.zip` | 209708670 | `82302A7DB4FC1B34E0B6676326610438D43B816BDF11A69D1D012A464EF2F92E` | RAW_ARCHIVE | NASA repository says users employ data at their own risk; the portal does not specify a license. |
| CALCE | CS2_35 | https://web.calce.umd.edu/batteries/data/CS2_35.zip | University of Maryland CALCE Battery Group | `CS2_35.zip` | 37056519 | `84D88CB98A43C02049F16C78806A78EC19084776EA5DF8A4B7C2F8F7BA4589E5` | RAW_ARCHIVE | CALCE requests citation of its database and referenced articles for publication use. |
| CALCE | CS2_36 | https://web.calce.umd.edu/batteries/data/CS2_36.zip | University of Maryland CALCE Battery Group | `CS2_36.zip` | 37881685 | `2E674F58BB6DC005C73BEF10E87EDBBD97AFD041DDE77A1D461A3C4ECAC1B027` | RAW_ARCHIVE | Same CALCE access/citation note. |
| CALCE | CS2_37 | https://web.calce.umd.edu/batteries/data/CS2_37.zip | University of Maryland CALCE Battery Group | `CS2_37.zip` | 41513048 | `8ED8AECBCBEA2BCB17F455923A974BE01EA21D7389210E1912D177DE5B6AF4A5` | RAW_ARCHIVE | Same CALCE access/citation note. |
| CALCE | CS2_38 | https://web.calce.umd.edu/batteries/data/CS2_38.zip | University of Maryland CALCE Battery Group | `CS2_38.zip` | 42811100 | `98787427C10478425EFD4F54D823805724DAFF1D2FCBD822B3A2D7006A1051D8` | RAW_ARCHIVE | Same CALCE access/citation note. |

## Roster and boundary

`NASA_ROSTER=B0005,B0006,B0007,B0018`

`CALCE_ROSTER=CS2_35,CS2_36,CS2_37,CS2_38`

`RAW_DATA_LOCATION=raw_data/ (Git ignored)`

`UNTRACKED_数据_DIRECTORY_STATUS=HISTORICAL_UNPROVENANCED_FEATURE_AND_CAPACITY_FILES__IGNORED__NOT_USED_AS_CANONICAL_SOURCE`

`TRAIN_SET_ASSIGNED=False`

`VALIDATION_SET_ASSIGNED=False`

`TEST_SET_ASSIGNED=False`

`TARGET_FIRST20_ASSIGNED=False`

`GAN_TARGET_POPULATION_ASSIGNED=False`

`SCF_GAN_TRAINING_EXECUTED=False`
