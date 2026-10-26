"""Мониторинг дрейфа данных: PSI (Population Stability Index) по каждому признаку."""

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-6
MIN_ROWS = 10


def _clean(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).dropna()


def population_stability_index(baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI между двумя распределениями. < 0.1 — стабильно, > 0.25 — сильный дрейф."""
    b_hist, edges = np.histogram(baseline, bins=bins)
    c_hist, _ = np.histogram(current, bins=edges)
    b_pct = np.clip(b_hist / max(b_hist.sum(), 1), EPS, None)
    c_pct = np.clip(c_hist / max(c_hist.sum(), 1), EPS, None)
    return float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))


def drift_report(
    current_df: pd.DataFrame, baseline_df: pd.DataFrame, features: Sequence[str]
) -> list[dict[str, Any]]:
    """Сравнивает текущие данные с базовыми по каждому признаку."""
    rows: list[dict[str, Any]] = []
    for col in features:
        if col not in current_df.columns or col not in baseline_df.columns:
            continue
        cur, base = _clean(current_df[col]), _clean(baseline_df[col])
        if len(cur) < MIN_ROWS or len(base) < MIN_ROWS:
            continue
        rows.append(
            {
                "feature": col,
                "baseline_mean": float(base.mean()),
                "current_mean": float(cur.mean()),
                "mean_shift": float(cur.mean() - base.mean()),
                "psi": population_stability_index(base, cur),
            }
        )
    return rows


def save_drift_report(rows: list[dict[str, Any]], monitoring_dir: Path) -> Path:
    """Сохраняет отчёт в ``monitoring/drift_YYYYMMDD_HHMMSS.csv``."""
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    path = monitoring_dir / f"drift_{datetime.now():%Y%m%d_%H%M%S}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
