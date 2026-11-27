from __future__ import annotations

import json

import pandas as pd

from dashboard.config import OUTPUTS_DIR


_RESULT_FILES = (
    ("results_naive_persistence.json", "Persistence"),
    ("results_naive_historical_average.json", "Historical Average"),
    ("results_stgcn_phys_seed42.json", "STGCN"),
    ("results_hgmwn_phys_seed42.json", "HGMWN"),
)


def _read_result(filename: str) -> dict[str, object] | None:
    path = OUTPUTS_DIR / filename
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return result if isinstance(result, dict) else None


def load_evaluation_results() -> pd.DataFrame:
    """Load real model metrics produced by the training scripts."""
    rows: list[dict[str, object]] = []
    for filename, display_name in _RESULT_FILES:
        result = _read_result(filename)
        if result is None:
            continue
        overall = result.get("overall")
        if not isinstance(overall, dict):
            continue
        rows.append(
            {
                "Mô hình": display_name,
                "MAE": overall.get("MAE"),
                "RMSE": overall.get("RMSE"),
                "MAPE (%)": overall.get("MAPE"),
                "Seed": result.get("seed"),
                "File": filename,
            }
        )
    return pd.DataFrame(rows)


def load_horizon_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for filename, display_name in _RESULT_FILES:
        result = _read_result(filename)
        if result is None:
            continue
        horizon = result.get("horizon")
        if not isinstance(horizon, dict):
            continue
        for horizon_name, metrics in horizon.items():
            if not isinstance(metrics, dict):
                continue
            rows.append(
                {
                    "Mô hình": display_name,
                    "Cửa sổ": str(horizon_name),
                    "MAE": metrics.get("MAE"),
                    "RMSE": metrics.get("RMSE"),
                    "MAPE (%)": metrics.get("MAPE"),
                }
            )
    return pd.DataFrame(rows)
