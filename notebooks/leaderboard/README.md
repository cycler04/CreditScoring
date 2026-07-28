# Top-leaderboard public code

This folder records the public Kaggle code that can be tied confidently to the
top three teams of the three credit competitions in this project.

Important: a leaderboard position does not imply that the team published its
final solution. Kaggle exports all seven accessible items below as **scripts**
(`.py` or `.R`), not as Jupyter `.ipynb` notebooks. They are preserved exactly
as returned by `kaggle kernels pull`, together with `kernel-metadata.json`.
These files should not be described as exact winning submissions unless the
author explicitly says so.

## Availability

| Competition | Rank | Team | Public code found | Local result |
|---|---:|---|---|---|
| GiveMeSomeCredit | 1 | Perfect Storm | None found | unavailable |
| GiveMeSomeCredit | 2 | Gxav | None found | unavailable |
| GiveMeSomeCredit | 3 | occupy | None found | unavailable |
| Home Credit Default Risk | 1 | Home Aloan | 3 public member kernels | downloaded |
| Home Credit Default Risk | 2 | ikiri_DS | 1 public Giba post-processing kernel | downloaded |
| Home Credit Default Risk | 3 | alijs & Evgeny | None found | unavailable |
| Home Credit Model Stability | 1 | yuuniee | 3 public kernels by `yuuniekiri` | downloaded |
| Home Credit Model Stability | 2 | Amazing Badger | None found | unavailable |
| Home Credit Model Stability | 3 | ZAT | None found | unavailable |

The 2011 GiveMeSomeCredit competition predates normal Kaggle notebook
integration; no public kernels belonging to its top-three teams were returned
by Kaggle search.

## Downloaded sources

### Home Aloan — Home Credit Default Risk rank 1

- `tunguz/xgb-simple-features` — Python script by team member Bojan Tunguz.
- `ogrellier/lighgbm-with-selected-features` — Python script by team member
  Olivier.
- `ogrellier/good-fun-with-ligthgbm` — Python script by Olivier.

These are public competition kernels from team members, not the complete final
Home Aloan ensemble.

### ikiri_DS — Home Credit Default Risk rank 2

- `titericz/giba-post-processing-user-id-boost` — R post-processing script by
  team member Giba.

This captures the published user-ID post-processing idea, not the team's full
second-place training pipeline.

### yuuniee — Home Credit Model Stability rank 1

- `yuuniekiri/fork-of-home-credit-risk-lightgbm`
- `yuuniekiri/fork-of-home-credit-catboost-inference`
- `yuuniekiri/fork-of-home-credit-lightautoml-inference`

These public Python scripts are authored by the winning account. They are
useful components, but they should not automatically be equated with the exact
private-leaderboard ensemble.

## Refresh

```bash
./scripts/download_leaderboard_notebooks.sh
```

The machine-readable search/download record is in
[`manifest.json`](manifest.json).
