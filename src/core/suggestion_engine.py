from typing import Any, Dict, List, Optional

def generate_suggestions(
    mode: str,
    question: str,
    sql_data: Optional[Dict] = None,
    profile: Optional[Dict] = None,
    insights: Optional[Dict] = None,
    table_names: Optional[List[str]] = None,
    context: Any = None,
) -> List[Dict[str, str]]:
    suggestions = []

    if mode == "sql" and sql_data:
        suggestions = _suggest_from_sql(question, sql_data, table_names or [])
    elif mode == "profile" and profile:
        suggestions = _suggest_from_profile(insights or {}, table_names or [])

    if context:
        suggestions = _augment_with_context(suggestions, context, question, table_names or [])

    suggestions = _deduplicate(suggestions, question)
    return suggestions[:3]


def _suggest_from_sql(question: str, sql_data: Dict, table_names: List[str]) -> List[Dict]:
    suggestions = []
    columns = sql_data.get("columns", [])
    rows = sql_data.get("rows", [])
    row_count = sql_data.get("row_count", 0)
    sql = sql_data.get("sql", "")
    columns_lower = [c.lower() for c in columns]

    has_group_by = "group by" in sql.lower()
    has_date = any(
        "日期" in c or "时间" in c or "月份" in c or "date" in c.lower() or "time" in c.lower()
        for c in columns
    )
    has_geo = any(
        kw in cl for kw in ["区域", "省份", "城市", "地区", "门店", "仓库"]
        for cl in columns_lower
    )
    has_product = any(
        kw in cl for kw in ["产品", "商品", "货品", "品类", "分类"]
        for cl in columns_lower
    )
    has_customer = any(
        kw in cl for kw in ["客户", "购货", "会员", "用户", "姓名"]
        for cl in columns_lower
    )

    numeric_cols = []
    label_cols = []
    for c in columns:
        is_num = False
        for r in rows[:3]:
            v = r.get(c)
            if v is not None and v != "" and not isinstance(v, bool):
                try:
                    float(v)
                    is_num = True
                    break
                except (ValueError, TypeError):
                    pass
        if is_num:
            numeric_cols.append(c)
        else:
            label_cols.append(c)

    is_aggregate = has_group_by and row_count >= 2
    is_top_n = row_count <= 10 and has_group_by
    has_multiple_num = len(numeric_cols) >= 2

    if is_aggregate and row_count >= 10:
        if label_cols and numeric_cols:
            lc = label_cols[0]
            nc = numeric_cols[0]
            suggestions.append({
                "text": f"哪些{lc}的{nc}最高？Top 5",
                "icon": "🔝",
            })
            suggestions.append({
                "text": f"哪些{lc}的{nc}最低？",
                "icon": "🔻",
            })

    if has_date and numeric_cols:
        nc = numeric_cols[0]
        suggestions.append({
            "text": f"{nc}的日/月趋势是怎样的？",
            "icon": "📈",
        })

    if has_geo and numeric_cols and not has_geo in ("区域", "省份", "城市") or has_geo:
        pass

    if has_multiple_num and len(numeric_cols) >= 2:
        n1, n2 = numeric_cols[0], numeric_cols[1]
        if has_group_by:
            suggestions.append({
                "text": f"对比{n1}和{n2}的关系",
                "icon": "🔗",
            })

    if has_product and has_geo and len(table_names) >= 2:
        suggestions.append({
            "text": f"按区域分析产品的销售分布",
            "icon": "🗺",
        })

    if row_count >= 30 and has_group_by:
        suggestions.append({
            "text": "数据画像：分析数据的整体画像和亮点",
            "icon": "📊",
        })

    if has_group_by and has_date:
        suggestions.append({
            "text": "换个时间维度对比看看？",
            "icon": "🔄",
        })

    if row_count >= 5 and has_group_by and numeric_cols:
        nc = numeric_cols[0]
        suggestions.append({
            "text": f"{nc}最大的异常值有哪些？",
            "icon": "⚠️",
        })

    if len(table_names) >= 2:
        suggestions.append({
            "text": "对比不同表的数据差异",
            "icon": "⚖️",
        })

    if row_count == 1:
        suggestions.append({
            "text": "查看明细数据",
            "icon": "🔍",
        })

    if has_customer and numeric_cols:
        nc = numeric_cols[0]
        suggestions.append({
            "text": f"哪些客户的{nc}增长最快？",
            "icon": "👤",
        })

    return suggestions


def _suggest_from_profile(insights: Dict, table_names: List[str]) -> List[Dict]:
    suggestions = []

    if not insights:
        return suggestions

    for tn, ins in insights.items():
        correlations = ins.get("correlations", [])
        pareto = ins.get("pareto_analysis", [])
        trends = ins.get("trends", [])

        meaningful_corr = [c for c in correlations if c.get("strength") in ("strong", "moderate")]
        meaningful_pareto = [p for p in pareto if p.get("concentration_pct", 0) >= 30]
        meaningful_trends = [t for t in trends if t.get("direction") != "stable"]

        if meaningful_corr:
            top = meaningful_corr[0]
            suggestions.append({
                "text": f"「{top['column_a']}」和「{top['column_b']}」为什么{'正' if top['direction'] == 'positive' else '负'}相关？看看具体明细",
                "icon": "🔗",
            })

        if meaningful_pareto:
            top = meaningful_pareto[0]
            suggestions.append({
                "text": f"「{top['dimension']}」中头部产品贡献了多少利润？",
                "icon": "📐",
            })

        if meaningful_trends:
            top = meaningful_trends[0]
            suggestions.append({
                "text": f"预测「{top['numeric_column']}」下个周期会继续{'上升' if top['direction'] == 'up' else '下降'}吗？",
                "icon": "🔮",
            })

        for corr in meaningful_corr[1:2]:
            suggestions.append({
                "text": f"对比「{corr['column_a']}」和「{corr['column_b']}」的异常数据点",
                "icon": "🔍",
            })

    if len(table_names) >= 2:
        suggestions.append({
            "text": "跨表关联分析：找出表之间的隐藏关系",
            "icon": "🕸",
        })

    suggestions.append({
        "text": "数据亮点：发现数据中的规律和模式",
        "icon": "✨",
    })

    return suggestions


def _augment_with_context(suggestions: List[Dict], context: Any, question: str, table_names: List[str]) -> List[Dict]:
    prev = context.last_turn()
    if not prev or not prev.columns:
        return suggestions

    if not context.is_likely_followup(question):
        return suggestions

    context_suggestions = []

    if not question.endswith("？") and not question.endswith("?"):
        scope_words = [
            f"换个维度看{prev.columns[0]}" if prev.columns else "",
        ]
        if len(prev.columns) >= 2:
            context_suggestions.append({
                "text": f"按「{prev.columns[0]}」再看一次？",
                "icon": "🔄",
            })

    if len(table_names) >= 2 and prev.mode == "sql":
        context_suggestions.append({
            "text": "和其他月的数据对比看看",
            "icon": "⚖️",
        })

    if prev.mode == "profile":
        context_suggestions.append({
            "text": "回到数据画像总览",
            "icon": "🔙",
        })

    return suggestions + context_suggestions[:2]


def _deduplicate(suggestions: List[Dict], question: str) -> List[Dict]:
    seen = set()
    result = []
    q_clean = question.replace(" ", "").replace("，", "").replace("？", "").replace("?", "")

    for s in suggestions:
        key = s["text"].replace(" ", "")
        if key in seen:
            continue
        if key in q_clean or q_clean in key:
            continue
        seen.add(key)
        result.append(s)

    return result