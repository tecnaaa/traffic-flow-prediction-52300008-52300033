from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.config import (
    CORRELATION_FILE,
    NODES_FILE,
    NUMERIC_TRAFFIC_COLUMNS,
    TRAFFIC_COLUMNS,
    TRAFFIC_FILE,
)


@dataclass
class DataBundle:
    nodes: pd.DataFrame
    traffic: pd.DataFrame
    correlation: pd.DataFrame | None
    errors: list[str]
    warnings: list[str]


def _file_stamp(path: Path) -> tuple[str, int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return str(path), stat.st_mtime_ns, stat.st_size


def _read_nodes(path: Path, errors: list[str]) -> pd.DataFrame:
    if not path.exists():
        errors.append(f"Không tìm thấy file tọa độ nút: {path.name}.")
        return pd.DataFrame(columns=["node_id", "road_name", "lat", "lon"])

    try:
        nodes = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        errors.append(f"Không đọc được {path.name}: {exc}")
        return pd.DataFrame(columns=["node_id", "road_name", "lat", "lon"])

    nodes.columns = [str(column).strip().lower() for column in nodes.columns]
    required = {"node_id", "road_name", "lat", "lon"}
    missing = sorted(required.difference(nodes.columns))
    if missing:
        errors.append(f"{path.name} thiếu cột bắt buộc: {', '.join(missing)}.")
        return pd.DataFrame(columns=["node_id", "road_name", "lat", "lon"])

    nodes = nodes[["node_id", "road_name", "lat", "lon"]].copy()
    nodes["node_id"] = nodes["node_id"].astype("string").str.strip()
    nodes["road_name"] = nodes["road_name"].astype("string").fillna("Không rõ")
    for column in ("lat", "lon"):
        nodes[column] = pd.to_numeric(nodes[column], errors="coerce")

    valid_coordinates = (
        nodes["lat"].between(-90, 90)
        & nodes["lon"].between(-180, 180)
    )
    invalid_count = int((~valid_coordinates).sum())
    if invalid_count:
        nodes = nodes.loc[valid_coordinates].copy()
    if nodes.empty:
        errors.append(f"{path.name} không có tọa độ hợp lệ để hiển thị.")
    elif invalid_count:
        errors.append(f"{invalid_count} dòng tọa độ không hợp lệ trong {path.name} đã bị bỏ qua.")
    return nodes.dropna(subset=["node_id", "lat", "lon"])


def _read_traffic(path: Path, errors: list[str]) -> pd.DataFrame:
    if not path.exists():
        errors.append(f"Không tìm thấy file lưu lượng: {path.name}.")
        return pd.DataFrame(columns=TRAFFIC_COLUMNS)

    try:
        traffic = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        errors.append(f"Không đọc được {path.name}: {exc}")
        return pd.DataFrame(columns=TRAFFIC_COLUMNS)

    traffic.columns = [str(column).strip().lower() for column in traffic.columns]
    if "timestamp" not in traffic.columns:
        if len(traffic.columns) != len(TRAFFIC_COLUMNS):
            errors.append(
                f"{path.name} không có header nhận diện được và có "
                f"{len(traffic.columns)} cột; cần {len(TRAFFIC_COLUMNS)} cột."
            )
            return pd.DataFrame(columns=TRAFFIC_COLUMNS)
        traffic = pd.read_csv(path, header=None, names=TRAFFIC_COLUMNS, encoding="utf-8-sig")

    missing = sorted(set(TRAFFIC_COLUMNS).difference(traffic.columns))
    if missing:
        errors.append(f"{path.name} thiếu cột bắt buộc: {', '.join(missing)}.")
        return pd.DataFrame(columns=TRAFFIC_COLUMNS)

    traffic = traffic[TRAFFIC_COLUMNS].copy()
    traffic["timestamp"] = pd.to_datetime(traffic["timestamp"], errors="coerce")
    traffic["node_id"] = traffic["node_id"].astype("string").str.strip()
    traffic["road_name"] = traffic["road_name"].astype("string")
    for column in NUMERIC_TRAFFIC_COLUMNS:
        traffic[column] = pd.to_numeric(traffic[column], errors="coerce")
    traffic = traffic.dropna(subset=["timestamp", "node_id"])
    if traffic.empty:
        errors.append(f"{path.name} không có bản ghi hợp lệ sau khi kiểm tra schema.")
    return traffic


def _read_correlation(path: Path, warnings: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        warnings.append("Chưa có outputs/correlation_matrix.csv; trang tương quan sẽ dùng dữ liệu thô nếu đủ.")
        return None
    try:
        correlation = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        warnings.append(f"Không đọc được correlation_matrix.csv: {exc}")
        return None
    if correlation.empty:
        warnings.append("Ma trận tương quan hiện có đang rỗng.")
        return None
    correlation.index = correlation.index.astype(str)
    correlation.columns = correlation.columns.astype(str)
    return correlation.apply(pd.to_numeric, errors="coerce")


@st.cache_data(show_spinner=False)
def _load_data(
    nodes_stamp: tuple[str, int, int] | None,
    traffic_stamp: tuple[str, int, int] | None,
    correlation_stamp: tuple[str, int, int] | None,
) -> DataBundle:
    del nodes_stamp, traffic_stamp, correlation_stamp
    errors: list[str] = []
    warnings: list[str] = []
    nodes = _read_nodes(NODES_FILE, errors)
    traffic = _read_traffic(TRAFFIC_FILE, errors)
    correlation = _read_correlation(CORRELATION_FILE, warnings)
    return DataBundle(nodes, traffic, correlation, errors, warnings)


def load_data() -> DataBundle:
    return _load_data(
        _file_stamp(NODES_FILE),
        _file_stamp(TRAFFIC_FILE),
        _file_stamp(CORRELATION_FILE),
    )


def latest_readings(traffic: pd.DataFrame) -> pd.DataFrame:
    if traffic.empty:
        return traffic.copy()
    ordered = traffic.dropna(subset=["timestamp"]).sort_values("timestamp")
    return ordered.drop_duplicates("node_id", keep="last").copy()


def calculate_correlation(traffic: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "road_name", "congestion_ratio_0_to_1"}
    if traffic.empty or not required.issubset(traffic.columns):
        return pd.DataFrame()
    observed = traffic.dropna(subset=["timestamp", "road_name", "congestion_ratio_0_to_1"]).copy()
    if observed.empty:
        return pd.DataFrame()
    observed["window"] = observed["timestamp"].dt.floor("15min")
    pivot = observed.pivot_table(
        index="window",
        columns="road_name",
        values="congestion_ratio_0_to_1",
        aggfunc="mean",
    )
    if pivot.shape[1] < 2:
        return pd.DataFrame()
    return pivot.corr(min_periods=2)
