#!/usr/bin/env python3
"""Record HCMS Kaggle submission attempts without inventing scores or ranks."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

OUTPUT = Path("outputs/hcms")
NOTEBOOK_SUBMISSION = OUTPUT / "submissions/kaggle_notebook_submission.json"


def main() -> None:
    metrics = pd.read_csv(OUTPUT / "models/metrics.csv")
    local_auc = (
        metrics.loc[
            metrics["split"].eq("test") & metrics["protocol"].eq("out_of_time"),
            ["model", "auc"],
        ]
        .set_index("model")["auc"]
        .to_dict()
    )
    stability = (
        pd.read_json(OUTPUT / "stability/stability_metric.json")
        .set_index("model")["stability"]
        .to_dict()
    )
    models = sorted(local_auc, key=local_auc.get, reverse=True)
    attempts_path = OUTPUT / "submissions/submission_attempts.json"
    previous_attempts = (
        json.loads(attempts_path.read_text(encoding="utf-8"))
        if attempts_path.exists()
        else {}
    )
    attempted_at = previous_attempts.get(
        "attempted_at_utc", datetime.now(UTC).isoformat()
    )
    rows = []
    for rank, model in enumerate(models, start=1):
        rows.append(
            {
                "model_rank": rank,
                "model": model,
                "file_name": f"{model}.csv",
                "submission_id": "",
                "status": "REJECTED_HTTP_400",
                "public_auc": "",
                "private_auc": "",
                "official_leaderboard_rank": "",
                "hypothetical_public_rank": "",
                "hypothetical_top_bracket": "",
                "public_leaderboard_teams": 3856,
                "hypothetical_top_percent": "",
                "local_oot_auc": local_auc[model],
                "local_stability": stability[model],
                "submission_attempted_at_utc": attempted_at,
                "submission_error": (
                    "Local file upload returned HTTP 400. The competition "
                    "deadline was 2024-05-27 and code submissions require a "
                    "Kaggle kernel/version; a code-submission probe returned "
                    "HTTP 403."
                ),
            }
        )
    notebook_submissions = []
    if NOTEBOOK_SUBMISSION.exists():
        notebook_record = json.loads(
            NOTEBOOK_SUBMISSION.read_text(encoding="utf-8")
        )
        notebook_submissions = notebook_record.get(
            "submissions", [notebook_record]
        )
        for notebook_submission in notebook_submissions:
            rows.append(
                {
                    "model_rank": len(rows) + 1,
                    "model": notebook_submission["model"],
                    "file_name": notebook_submission["file_name"],
                    "submission_id": notebook_submission["submission_id"],
                    "status": notebook_submission["status"],
                    "public_auc": notebook_submission.get("public_auc", ""),
                    "private_auc": notebook_submission.get("private_auc", ""),
                    "official_leaderboard_rank": notebook_submission.get(
                        "official_leaderboard_rank", ""
                    ),
                    "hypothetical_public_rank": "",
                    "hypothetical_top_bracket": "",
                    "public_leaderboard_teams": 3856,
                    "hypothetical_top_percent": "",
                    "local_oot_auc": notebook_submission.get("local_oot_auc", ""),
                    "local_stability": "",
                    "submission_attempted_at_utc": notebook_submission[
                        "submitted_at_utc"
                    ],
                    "submission_error": notebook_submission.get("note", ""),
                }
            )
    destination = OUTPUT / "submissions/submission_scores.csv"
    pd.DataFrame(rows).to_csv(destination, index=False)
    attempts_path.write_text(
        json.dumps(
            {
                "competition": "home-credit-credit-risk-model-stability",
                "competition_deadline": "2024-05-27T23:59:00",
                "attempted_at_utc": attempted_at,
                "file_upload_attempts": {
                    model: {
                        "file": f"{model}.csv",
                        "http_status": 400,
                        "submission_id": None,
                    }
                    for model in models
                },
                "code_submission_probe_http_status": 403,
                "kaggle_submissions_after_attempt": notebook_submissions,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} explicit submission-status rows to {destination}")


if __name__ == "__main__":
    main()
