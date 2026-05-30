import json
import math
import statistics
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Shanghai")

DATE_PATTERNS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y年%m月%d日",
    "%m/%d/%Y",
    "%d/%m/%Y",
]


def _try_parse_date(val: str) -> Optional[datetime]:
    if not val or not isinstance(val, str):
        return None
    val = val.strip()
    for fmt in DATE_PATTERNS:
        try:
            dt = datetime.strptime(val, fmt)
            if dt.year >= 1900 and dt.year <= 2100:
                return dt
        except ValueError:
            continue
    return None


def _is_numeric(val: Any) -> bool:
    if val is None or val == "":
        return False
    try:
        float(str(val))
        return True
    except (ValueError, TypeError):
        return False


def _guess_column_type(values: List[Any]) -> str:
    total = len(values)
    if total == 0:
        return "empty"
    non_empty = [v for v in values if v is not None and v != "" and v != "None"]
    if len(non_empty) < total * 0.3:
        return "sparse"

    date_count = 0
    num_count = 0
    for v in non_empty:
        s = str(v)
        if _try_parse_date(s):
            date_count += 1
        elif _is_numeric(s):
            num_count += 1
    n = len(non_empty)
    if date_count / n > 0.6:
        return "date"
    if num_count / n > 0.7:
        return "numeric"
    unique_ratio = len(set(str(v) for v in non_empty)) / n
    if unique_ratio > 0.9:
        return "categorical"
    return "text"


def _compute_numeric_stats(values: List[float]) -> Dict[str, Any]:
    n = len(values)
    if n == 0:
        return {"count": 0}
    sorted_vals = sorted(values)
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if n > 1 else 0

    def percentile(data: List[float], p: float) -> float:
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        if f == c:
            return data[f]
        return data[f] + (data[c] - data[f]) * (k - f)

    return {
        "count": n,
        "min": sorted_vals[0],
        "q1": percentile(sorted_vals, 25),
        "median": statistics.median(sorted_vals),
        "q3": percentile(sorted_vals, 75),
        "max": sorted_vals[-1],
        "mean": round(mean, 2),
        "std": round(stdev, 2),
        "sum": round(sum(values), 2),
        "cv": round(stdev / mean * 100, 1) if mean != 0 else None,
    }


def _detect_anomalies(rows: List[Dict], numeric_cols: List[str], stats_map: Dict[str, Dict]) -> List[Dict]:
    anomalies = []
    for col in numeric_cols:
        stats = stats_map.get(col, {})
        mean = stats.get("mean")
        std = stats.get("std")
        if mean is None or std is None or std == 0:
            continue
        threshold = 3.0
        for i, row in enumerate(rows):
            val = row.get(col)
            if val is None or val == "":
                continue
            try:
                v = float(val)
            except (ValueError, TypeError):
                continue
            z = abs(v - mean) / std
            if z > threshold:
                anomalies.append({
                    "row_index": i,
                    "column": col,
                    "value": v,
                    "z_score": round(z, 2),
                    "direction": "high" if v > mean else "low",
                    "row_preview": {k: row[k] for k in list(row.keys())[:6]},
                })
    anomalies.sort(key=lambda a: a["z_score"], reverse=True)
    return anomalies[:15]


def _detect_dimension_metric_pairs(
    rows: List[Dict], columns: List[str], col_types: Dict[str, str]
) -> List[Dict]:
    dim_cols = [c for c in columns if col_types.get(c) in ("categorical", "text")]
    num_cols = [c for c in columns if col_types.get(c) == "numeric"]

    pairs = []
    for dim in dim_cols[:5]:
        for metric in num_cols[:5]:
            counter = Counter()
            for row in rows:
                key = row.get(dim)
                val = row.get(metric)
                if key is None or val is None or val == "" or key == "":
                    continue
                try:
                    counter[str(key)] += float(val)
                except (ValueError, TypeError):
                    pass
            if len(counter) >= 2:
                top_items = counter.most_common(10)
                pairs.append({
                    "dimension": dim,
                    "metric": metric,
                    "top_n": [{"key": k, "value": round(v, 2)} for k, v in top_items],
                    "unique_count": len(counter),
                })
    return pairs[:8]


def _analyze_date_column(values: List[str]) -> Optional[Dict]:
    dates = []
    for v in values:
        dt = _try_parse_date(v)
        if dt:
            dates.append(dt)
    if len(dates) < 3:
        return None

    dates.sort()
    day_count = Counter(d.date() for d in dates)
    month_count: Counter = Counter()
    weekday_count: Counter = Counter()
    for d in dates:
        month_count[d.strftime("%Y-%m")] += 1
        weekday_count[d.strftime("%A")] += 1

    daily_avg = len(dates) / max(len(day_count), 1)
    peak_day = day_count.most_common(1)[0] if day_count else (None, 0)
    peak_month = month_count.most_common(3)
    busy_weekday = weekday_count.most_common(1)[0] if weekday_count else (None, 0)

    date_range = f"{dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}"

    trend_hint = ""
    if len(month_count) >= 3:
        months_sorted = sorted(month_count.keys())
        first_avg = sum(month_count[m] for m in months_sorted[:1]) / 1
        last_avg = sum(month_count[m] for m in months_sorted[-1:]) / 1
        if last_avg > first_avg * 1.15:
            trend_hint = "up"
        elif last_avg < first_avg * 0.85:
            trend_hint = "down"
        else:
            trend_hint = "stable"

    return {
        "total_dates": len(dates),
        "unique_days": len(day_count),
        "date_range": date_range,
        "daily_avg": round(daily_avg, 1),
        "peak_day": str(peak_day[0]),
        "peak_day_count": peak_day[1],
        "peak_months": [{"month": m, "count": c} for m, c in peak_month],
        "busiest_weekday": busy_weekday[0],
        "trend": trend_hint,
        "weekday_distribution": dict(weekday_count),
    }


def profile_table(table_store, table_name: str) -> Dict[str, Any]:
    rows, columns = table_store.execute_sql(f'SELECT * FROM "{table_name}"')
    total_rows = len(rows)

    if total_rows == 0:
        return {"table_name": table_name, "total_rows": 0, "error": "表中无数据"}

    col_types: Dict[str, str] = {}
    col_stats: Dict[str, Dict] = {}
    col_unique: Dict[str, int] = {}
    col_missing: Dict[str, int] = {}
    date_analysis: List[Dict] = []
    numeric_cols: List[str] = []

    for col in columns:
        values = [row.get(col) for row in rows]
        col_type = _guess_column_type(values)
        col_types[col] = col_type

        non_empty = [v for v in values if v is not None and v != "" and v != "None"]
        col_unique[col] = len(set(str(v) for v in non_empty))
        col_missing[col] = total_rows - len(non_empty)

        if col_type == "numeric":
            numeric_cols.append(col)
            nums = [float(v) for v in non_empty if _is_numeric(v)]
            col_stats[col] = _compute_numeric_stats(nums)
        elif col_type == "date":
            date_result = _analyze_date_column(non_empty)
            if date_result:
                date_analysis.append({"column": col, **date_result})

    stats_map = {col: col_stats.get(col, {}) for col in numeric_cols}
    anomalies = _detect_anomalies(rows, numeric_cols, stats_map)
    dim_metric_pairs = _detect_dimension_metric_pairs(rows, columns, col_types)

    return {
        "table_name": table_name,
        "total_rows": total_rows,
        "total_columns": len(columns),
        "columns": columns,
        "column_types": [
            {
                "name": col,
                "type": col_types[col],
                "unique_values": col_unique[col],
                "missing": col_missing[col],
                "missing_pct": round(col_missing[col] / total_rows * 100, 1),
            }
            for col in columns
        ],
        "numeric_stats": {col: col_stats.get(col, {}) for col in numeric_cols},
        "anomalies": anomalies,
        "top_n_pairs": dim_metric_pairs,
        "date_analysis": date_analysis,
    }