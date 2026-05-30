"""
MVP-8 测试套件：多表关联 + 多文件知识融合
覆盖场景：关系检测、UNION ALL 跨表聚合、INNER JOIN 跨表关联、单表对照
"""

import json
import sys
import time
from pathlib import Path

import requests

BASE = "http://localhost:1234"
PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  --  {detail}")


def stream_chat(question: str, timeout: int = 120) -> dict:
    """发送流式聊天请求，返回结构化结果"""
    result = {
        "mode": "",
        "sql": "",
        "columns": [],
        "row_count": 0,
        "aggregates": {},
        "answer": "",
        "events": set(),
    }
    try:
        resp = requests.post(
            f"{BASE}/api/chat/stream",
            json={"question": question},
            stream=True,
            timeout=timeout,
        )
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
                t = ev.get("type", "")
                result["events"].add(t)
                if t == "sql_data":
                    sd = ev.get("sql_data", {})
                    result["sql"] = sd.get("sql", "")
                    result["columns"] = sd.get("columns", [])
                    result["row_count"] = sd.get("row_count", 0)
                    result["mode"] = ev.get("mode", "")
                    result["aggregates"] = ev.get("aggregates", {})
                elif t == "token":
                    result["answer"] += ev.get("text", "")
            except (json.JSONDecodeError, KeyError):
                pass
    except requests.exceptions.Timeout:
        result["error"] = "TIMEOUT"
    except requests.exceptions.ConnectionError:
        result["error"] = "CONNECTION_REFUSED"
    return result


# ============================================================
# 第 1 组：关系检测 API 测试
# ============================================================
print("\n" + "=" * 60)
print("第 1 组：关系检测 API")
print("=" * 60)

# 1.1 手动触发检测
print("\n[1.1] 手动触发关系检测 POST /api/table/relationships/detect")
try:
    r = requests.post(f"{BASE}/api/table/relationships/detect", timeout=30)
    data = r.json()
    rels = data.get("relationships", [])
    check("返回 status 200", r.status_code == 200, f"got {r.status_code}")
    check("返回 count 字段", "count" in data)
    check(
        "检测到关联关系",
        len(rels) > 0,
        f"count={data.get('count', 0)}",
    )
    if rels:
        first = rels[0]
        check(
            "关联包含 table_a",
            "table_a" in first,
        )
        check(
            "关联包含 column_a",
            "column_a" in first,
        )
        check(
            "关联包含 table_b",
            "table_b" in first,
        )
        check(
            "关联包含 column_b",
            "column_b" in first,
        )
        check(
            "关联包含 confidence",
            "confidence" in first,
        )
        check(
            "关联包含 relation_type",
            "relation_type" in first,
        )
        check(
            "confidence 在 0~1 之间",
            0 <= first["confidence"] <= 1,
            f"confidence={first['confidence']}",
        )
        # 统计类型分布
        types = {}
        for rel in rels:
            types[rel["relation_type"]] = types.get(rel["relation_type"], 0) + 1
        print(f"    类型分布: {types}")
        print(f"    Top-3 关联:")
        for rel in rels[:3]:
            print(
                f"      {rel['table_a']}.{rel['column_a']} "
                f"<-> {rel['table_b']}.{rel['column_b']} "
                f"({rel['relation_type']}, conf={rel['confidence']})"
            )
    else:
        print("    (当前仅 1 张表，检测不到关联)")
except requests.exceptions.ConnectionError:
    check("服务可连接", False, "服务器未启动")


# 1.2 获取缓存的关联
print("\n[1.2] 获取缓存关联 GET /api/table/relationships")
try:
    r = requests.get(f"{BASE}/api/table/relationships", timeout=30)
    data = r.json()
    check("返回 status 200", r.status_code == 200)
    check("返回 count 字段", "count" in data)
    print(f"    缓存关联数: {data.get('count', 0)}")
except requests.exceptions.ConnectionError:
    check("服务可连接", False, "服务器未启动")


# ============================================================
# 第 2 组：单表查询（对照测试）
# ============================================================
print("\n" + "=" * 60)
print("第 2 组：单表查询（对照）")
print("=" * 60)


def run_chat_test(name: str, question: str, expectations: dict):
    """执行一次聊天查询测试并验证"""
    print(f"\n[{name}] 问题: {question}")
    result = stream_chat(question)
    print(f"    模式: {result['mode']}")
    print(f"    SQL: {result['sql'][:120]}...")
    print(f"    行数: {result['row_count']}, 列: {result['columns'][:5]}")

    if "sql_data" not in result["events"]:
        check(f"{name} - 触发 SQL 模式", False, "未收到 sql_data 事件")
        return

    check(
        f"{name} - 生成 SQL",
        bool(result["sql"]),
        result.get("error", ""),
    )
    check(
        f"{name} - 有返回数据",
        result["row_count"] > 0,
        f"rows={result['row_count']}",
    )
    for key, expected in expectations.items():
        if key == "min_rows":
            check(
                f"{name} - 行数 >= {expected}",
                result["row_count"] >= expected,
                f"rows={result['row_count']}",
            )
        elif key == "max_rows":
            check(
                f"{name} - 行数 <= {expected}",
                result["row_count"] <= expected,
                f"rows={result['row_count']}",
            )
        elif key == "aggregates_nonempty":
            check(
                f"{name} - 含聚合数据",
                bool(result["aggregates"]),
            )
        elif key == "answer_contains":
            if isinstance(expected, list):
                for word in expected:
                    check(
                        f"{name} - 回答含 '{word}'",
                        word in result["answer"],
                    )
            else:
                check(
                    f"{name} - 回答含 '{expected}'",
                    expected in result["answer"],
                )

    if result["aggregates"]:
        sums = result["aggregates"].get("sums", {})
        if sums:
            print(f"    聚合值: {sums}")


# 2.1 按区域统计销售数量
run_chat_test(
    "2.1 单表按区域统计",
    "2月销售出库表中按区域统计销售数量",
    {"min_rows": 10, "aggregates_nonempty": True, "answer_contains": "区域"},
)

# 2.2 查询 Top-N 产品
run_chat_test(
    "2.2 单表产品 Top-N",
    "2月销售出库表中销售数量最多的5个产品是哪些",
    {"min_rows": 3, "aggregates_nonempty": True, "answer_contains": ["产品", "数量"]},
)


# ============================================================
# 第 3 组：跨表查询（UNION ALL 聚合模式）
# ============================================================
print("\n" + "=" * 60)
print("第 3 组：跨表查询（UNION ALL 聚合）")
print("=" * 60)

# 3.1 按购货单位对比两个月的销售数量
run_chat_test(
    "3.1 跨表-购货单位对比销量",
    "对比2月和4月各购货单位的销售数量",
    {
        "min_rows": 20,
        "aggregates_nonempty": True,
        "answer_contains": ["2月", "4月", "购货单位"],
    },
)

# 3.2 按产品对比两个月的销售额
run_chat_test(
    "3.2 跨表-产品对比销售额",
    "对比2月和4月各产品的销售金额",
    {
        "min_rows": 10,
        "aggregates_nonempty": True,
        "answer_contains": ["产品", "金额"],
    },
)

# 3.3 按区域对比两个月的销售数量
run_chat_test(
    "3.3 跨表-区域对比销量",
    "对比2月和4月各区域的销售数量",
    {
        "min_rows": 10,
        "aggregates_nonempty": True,
        "answer_contains": "区域",
    },
)

# 3.4 汇总对比（全表总量对比）
run_chat_test(
    "3.4 跨表-月度总量对比",
    "对比2月和4月的总销售数量和总销售金额",
    {
        "min_rows": 1,
        "aggregates_nonempty": True,
        "answer_contains": ["2月", "4月"],
    },
)


# ============================================================
# 第 4 组：数据画像 + 跨表场景
# ============================================================
print("\n" + "=" * 60)
print("第 4 组：数据画像")
print("=" * 60)

result = stream_chat("给2月的数据做个数据画像")
check("4.1 数据画像 - 收到回答", len(result["answer"]) > 100, f"len={len(result['answer'])}")
has_row_like = any(w in result["answer"] for w in ["行", "记录", "数据", "条", "总"])
has_col_like = any(w in result["answer"] for w in ["列", "字段", "维度", "列数", "指标"])
check("4.2 数据画像 - 回答含关键信息", has_row_like and has_col_like, f"answer[:200]={result['answer'][:200]}")


# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 60)
print(f"测试完成: 通过 {PASS}/{PASS + FAIL}, 失败 {FAIL}/{PASS + FAIL}")
if FAIL == 0:
    print("全部通过!")
else:
    print(f"有 {FAIL} 个测试失败")
    sys.exit(1)