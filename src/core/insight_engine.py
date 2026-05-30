import math
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.core.data_profiler import _is_numeric, _try_parse_date


def _is_noise_column(col: str) -> bool:
    skip_prefixes = ["sheet_", "求和项", "计数项", "平均值项", "最大值项", "最小值项"]
    skip_exact = ["sheet"]
    cl = col.strip()
    if cl.lower() in skip_exact:
        return True
    for p in skip_prefixes:
        if cl.lower().startswith(p.lower()):
            return True
    return False


def compute_correlation_matrix(
    rows: List[Dict],
    numeric_cols: List[str],
    col_stats: Dict[str, Dict],
) -> List[Dict]:
    numeric_cols = [c for c in numeric_cols if not _is_noise_column(c)]
    correlations = []
    paired = set()

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            ca, cb = numeric_cols[i], numeric_cols[j]
            key = (ca, cb) if ca < cb else (cb, ca)
            if key in paired:
                continue
            paired.add(key)

            paired_vals = []
            for row in rows:
                a = row.get(ca)
                b = row.get(cb)
                if a is None or b is None or a == "" or b == "":
                    continue
                try:
                    paired_vals.append((float(a), float(b)))
                except (ValueError, TypeError):
                    continue

            n = len(paired_vals)
            if n < 3:
                continue

            mean_a = sum(v[0] for v in paired_vals) / n
            mean_b = sum(v[1] for v in paired_vals) / n
            cov = sum((v[0] - mean_a) * (v[1] - mean_b) for v in paired_vals) / n
            std_a = math.sqrt(sum((v[0] - mean_a) ** 2 for v in paired_vals) / n)
            std_b = math.sqrt(sum((v[1] - mean_b) ** 2 for v in paired_vals) / n)

            if std_a == 0 or std_b == 0:
                continue

            r = cov / (std_a * std_b)
            r = round(r, 3)

            strength = "weak"
            if abs(r) >= 0.7:
                strength = "strong"
            elif abs(r) >= 0.4:
                strength = "moderate"

            correlations.append({
                "column_a": ca,
                "column_b": cb,
                "coefficient": r,
                "strength": strength,
                "direction": "positive" if r > 0 else "negative",
                "sample_size": n,
                "mean_a": col_stats.get(ca, {}).get("mean"),
                "mean_b": col_stats.get(cb, {}).get("mean"),
            })

    correlations.sort(key=lambda c: abs(c["coefficient"]), reverse=True)
    return correlations[:12]


def compute_pareto_analysis(
    rows: List[Dict],
    dimension_cols: List[str],
    numeric_cols: List[str],
) -> List[Dict]:
    dimension_cols = [c for c in dimension_cols if not _is_noise_column(c)]
    numeric_cols = [c for c in numeric_cols if not _is_noise_column(c)]
    pareto_results = []

    for dim in dimension_cols[:6]:
        for metric in numeric_cols[:6]:
            counter = Counter()
            null_count = 0
            for row in rows:
                key = row.get(dim)
                val = row.get(metric)
                if key is None or val is None or val == "" or key == "":
                    null_count += 1
                    continue
                try:
                    counter[str(key)] += float(val)
                except (ValueError, TypeError):
                    null_count += 1

            if len(counter) < 2:
                continue

            total = sum(counter.values())
            if total <= 0:
                continue

            sorted_items = counter.most_common()
            cum_sum = 0.0
            vital_few = []
            trivial_many_count = 0
            trivial_many_pct = 0.0
            crossed = False

            for key, val in sorted_items:
                pct = round(val / total * 100, 1)
                cum_sum += pct
                if not crossed:
                    vital_few.append({"key": key, "value": round(val, 2), "pct": pct, "cum_pct": round(cum_sum, 1)})
                    if cum_sum >= 80:
                        crossed = True
                else:
                    trivial_many_count += 1
                    trivial_many_pct += pct

            if not vital_few:
                continue

            concentration = round(vital_few[0]["pct"], 1)
            vital_count = len(vital_few)
            total_items = len(counter)

            pareto_results.append({
                "dimension": dim,
                "metric": metric,
                "total": round(total, 2),
                "vital_count": vital_count,
                "trivial_count": trivial_many_count,
                "total_categories": total_items,
                "concentration_pct": concentration,
                "vital_few": vital_few[:8],
                "trivial_pct": round(trivial_many_pct, 1),
            })

    pareto_results.sort(key=lambda p: p["concentration_pct"], reverse=True)
    return pareto_results[:8]


def detect_trends(
    rows: List[Dict],
    date_cols: List[str],
    numeric_cols: List[str],
) -> List[Dict]:
    date_cols = [c for c in date_cols if not _is_noise_column(c)]
    numeric_cols = [c for c in numeric_cols if not _is_noise_column(c)]
    trends = []

    for date_col in date_cols[:4]:
        dates_with_indices = []
        for i, row in enumerate(rows):
            val = row.get(date_col)
            if val is None or val == "":
                continue
            dt = _try_parse_date(str(val))
            if dt:
                dates_with_indices.append((i, dt))

        if len(dates_with_indices) < 5:
            continue

        dates_with_indices.sort(key=lambda x: x[1])

        for num_col in numeric_cols[:6]:
            time_series = []
            for idx, dt in dates_with_indices:
                val = rows[idx].get(num_col)
                if val is None or val == "":
                    continue
                try:
                    time_series.append((dt, float(val)))
                except (ValueError, TypeError):
                    continue

            if len(time_series) < 6:
                continue

            time_series.sort(key=lambda x: x[0])

            values = [v for _, v in time_series]
            mean_val = sum(values) / len(values)
            if mean_val == 0:
                continue

            n = len(values)
            indices = list(range(n))
            mean_x = (n - 1) / 2
            mean_y = mean_val

            num_cov = sum((indices[k] - mean_x) * (values[k] - mean_y) for k in range(n))
            den_x = sum((indices[k] - mean_x) ** 2 for k in range(n))
            den_y = sum((values[k] - mean_y) ** 2 for k in range(n))

            if den_x == 0 or den_y == 0:
                continue

            slope = num_cov / den_x
            trend_strength = abs(num_cov) / math.sqrt(den_x * den_y) if den_y > 0 else 0

            first_vals = [v for _, v in time_series[:max(2, n // 3)]]
            last_vals = [v for _, v in time_series[-max(2, n // 3):]]
            first_avg = sum(first_vals) / len(first_vals) if first_vals else 0
            last_avg = sum(last_vals) / len(last_vals) if last_vals else 0

            if first_avg != 0:
                change_pct = round((last_avg - first_avg) / abs(first_avg) * 100, 1)
            else:
                change_pct = 0

            direction = "stable"
            if abs(change_pct) < 5 or trend_strength < 0.2:
                direction = "stable"
            elif change_pct > 0:
                direction = "up"
            else:
                direction = "down"

            if direction != "stable" or abs(change_pct) > 10:
                trends.append({
                    "date_column": date_col,
                    "numeric_column": num_col,
                    "data_points": n,
                    "first_avg": round(first_avg, 2),
                    "last_avg": round(last_avg, 2),
                    "change_pct": change_pct,
                    "direction": direction,
                    "trend_strength": round(trend_strength, 3),
                    "time_range": f"{time_series[0][0].strftime('%Y-%m-%d')} ~ {time_series[-1][0].strftime('%Y-%m-%d')}",
                })

    trends.sort(key=lambda t: abs(t["change_pct"]), reverse=True)
    return trends[:8]


def discover_insights(
    table_store,
    table_name: str,
    column_types: List[Dict],
    numeric_stats: Dict[str, Dict],
    top_n_pairs: List[Dict],
    date_analysis: List[Dict],
) -> Dict[str, Any]:
    rows, columns = table_store.execute_sql(f'SELECT * FROM "{table_name}"')

    col_type_map = {ct["name"]: ct["type"] for ct in column_types}
    numeric_cols = [ct["name"] for ct in column_types if ct["type"] == "numeric"]
    date_cols = [ct["name"] for ct in column_types if ct["type"] == "date"]
    dimension_cols = [ct["name"] for ct in column_types if ct["type"] in ("categorical", "text")]

    correlations = compute_correlation_matrix(rows, numeric_cols, numeric_stats)
    pareto = compute_pareto_analysis(rows, dimension_cols, numeric_cols)
    trends = detect_trends(rows, date_cols, numeric_cols)

    meaningful_correlations = [c for c in correlations if c["strength"] in ("strong", "moderate")]
    meaningful_pareto = [p for p in pareto if p["concentration_pct"] >= 30 and p["vital_count"] <= 5]
    meaningful_trends = [t for t in trends if t["direction"] != "stable"]

    summary_insights = []

    if meaningful_correlations:
        top_corr = meaningful_correlations[0]
        summary_insights.append(
            f"「{top_corr['column_a']}」与「{top_corr['column_b']}」呈"
            f"{'正' if top_corr['direction'] == 'positive' else '负'}相关"
            f"（r={top_corr['coefficient']}），属{top_corr['strength']}相关"
        )

    if meaningful_pareto:
        top_pareto = meaningful_pareto[0]
        summary_insights.append(
            f"「{top_pareto['dimension']}」维度中，前 {top_pareto['vital_count']} 个类别"
            f"贡献了 {top_pareto['metric']} 的 >=80%（头部集中度 {top_pareto['concentration_pct']}%）"
        )

    if meaningful_trends:
        top_trend = meaningful_trends[0]
        direction_text = {"up": "上升", "down": "下降", "stable": "稳定"}
        summary_insights.append(
            f"「{top_trend['numeric_column']}」随时间呈{direction_text[top_trend['direction']]}趋势"
            f"（变化 {top_trend['change_pct']:+.1f}%）"
        )

    return {
        "table_name": table_name,
        "correlations": correlations,
        "pareto_analysis": pareto,
        "trends": trends,
        "summary_insights": summary_insights,
    }