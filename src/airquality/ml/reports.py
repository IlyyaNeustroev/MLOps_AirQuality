"""Отчёты об обучении: человекочитаемый .txt и машинно-читаемый .json."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import classification_report


def write_training_report(
    reports_dir: Path,
    best_params: dict[str, Any],
    best_cv_score: float,
    y_true: pd.Series,
    y_pred: Any,
) -> tuple[Path, Path]:
    """Пишет ``train_report_<ts>.txt`` и ``train_report_<ts>.json``."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = f"{datetime.now():%Y%m%d_%H%M%S}"
    txt_path = reports_dir / f"train_report_{ts}.txt"
    json_path = reports_dir / f"train_report_{ts}.json"

    report_text = classification_report(y_true, y_pred, zero_division=0)
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    txt_path.write_text(
        "BEST_PARAMS\n"
        f"{json.dumps(best_params, ensure_ascii=False, indent=2)}\n\n"
        f"BEST_CV_SCORE\n{best_cv_score}\n\n"
        f"CLASSIFICATION_REPORT\n{report_text}",
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(
            {
                "best_params": best_params,
                "best_cv_score": best_cv_score,
                "classification_report": report_dict,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return txt_path, json_path
