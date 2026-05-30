import json
import re
from typing import Any, Dict, List, Tuple, Optional

from langchain_openai import ChatOpenAI

from src.config.settings import get_llm_config
from src.core.table_store import TableStore

SQL_SYSTEM_PROMPT = """你是一个 SQL 查询生成器。根据用户问题和数据库表结构，生成一条 SQLite 查询语句。

规则：
1. 只生成 SELECT 语句，禁止 INSERT/UPDATE/DELETE/DROP/ALTER/CREATE
2. 明确列出需要的列名，不要用 SELECT *
3. 表名和列名必须用双引号包裹，因为它们是中文命名的
4. WHERE 条件中字符串用单引号
5. 对于模糊匹配字符串，使用 LIKE '%关键词%'
6. 如果用户问\"明细\"，返回详细行数据；如果问\"汇总/统计/合计\"，使用 GROUP BY 和聚合函数（SUM、COUNT、AVG 等）
7. 日期列的值格式类似 '2026-04-09'，查询月份用 LIKE '2026-04-%'
8. 不要加 LIMIT 限制，返回所有匹配行
9. 只输出 SQL 语句，不要有任何解释或 markdown 标记
10. 如果无法生成有效 SQL，输出: IMPOSSIBLE
11. 如果 schema 中提示了表间关联关系，你应当使用 INNER JOIN 或 LEFT JOIN 进行跨表查询
12. 跨表 JOIN 时，SELECT 和 WHERE 中的列名前要加表别名以消除歧义，如 t1.\"列名\", t2.\"列名\"

可用的表结构：
{schema}

用户问题：{question}

SQL:"""


class SqlQueryEngine:
    def __init__(self, table_store: TableStore | None = None):
        llm_cfg = get_llm_config()
        self.llm = ChatOpenAI(
            model=llm_cfg["model_name"],
            base_url=llm_cfg["base_url"],
            api_key=llm_cfg["api_key"],
            temperature=0,
        )
        self.table_store = table_store or TableStore()
        self.max_retries = 2
        self.max_result_rows = 200

    def can_handle(self, question: str) -> bool:
        if not self.table_store.get_table_names():
            return False

        data_keywords = [
            "统计", "汇总", "合计", "平均", "最多", "最少", "排名",
            "销量", "销售额", "金额", "数量", "库存", "明细",
            "哪个", "哪家", "多少", "第几", "占比", "对比",
        ]
        question_lower = question.lower()
        if any(kw in question_lower for kw in data_keywords):
            return True
        if any(kw in question for kw in ["表", "Sheet", "sheet", "数据"]):
            return True
        return False

    def query(self, question: str) -> Dict[str, Any]:
        schema = self.table_store.get_schema_for_llm()

        sql = None
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                sql = self._generate_sql(schema, question, last_error)
                if not sql or sql.strip() == "IMPOSSIBLE":
                    return {
                        "success": False,
                        "error": "无法将问题转换为 SQL 查询，请换一种问法",
                        "mode": "sql",
                    }
                sql = self._clean_sql(sql)
                rows, columns = self.table_store.execute_sql(sql)
                aggregates = self._compute_aggregates(rows, columns)

                inflation_error = self._detect_join_inflation(sql, aggregates)
                if inflation_error and attempt < self.max_retries:
                    last_error = inflation_error
                    continue

                return {
                    "success": True,
                    "mode": "sql",
                    "sql": sql,
                    "columns": columns,
                    "rows": rows,
                    "row_count": len(rows),
                    "aggregates": aggregates,
                }
            except Exception as e:
                last_error = str(e)
                if attempt >= self.max_retries:
                    return {
                        "success": False,
                        "error": f"SQL 查询失败: {last_error}",
                        "sql": sql,
                        "mode": "sql",
                    }

    def _compute_aggregates(self, rows: List[Dict], columns: List[str]) -> Dict[str, Any]:
        if not rows:
            return {}

        result = {"row_count": len(rows)}
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        numeric_columns: List[str] = []

        for col in columns:
            values = []
            for row in rows:
                val = row.get(col)
                if val is not None and val != "":
                    try:
                        values.append(float(val))
                    except (ValueError, TypeError):
                        pass
            if values:
                numeric_columns.append(col)
                sums[col] = sum(values)
                counts[col] = len(values)

        result["numeric_columns"] = numeric_columns
        result["sums"] = sums
        result["counts"] = counts
        result["avg"] = {col: sums[col] / counts[col] for col in numeric_columns}

        return result

    def _generate_sql(self, schema: str, question: str, last_error: Optional[str] = None) -> str:
        prompt = SQL_SYSTEM_PROMPT.format(schema=schema, question=question)
        if last_error:
            prompt += f"\n\n上一次生成的 SQL 执行失败，错误信息: {last_error}\n请修正后重新生成 SQL:"
        resp = self.llm.invoke(prompt)
        return resp.content.strip() if hasattr(resp, "content") else str(resp).strip()

    def _clean_sql(self, sql: str) -> str:
        sql = sql.strip()
        sql = re.sub(r"^```(?:sql)?\s*\n?", "", sql)
        sql = re.sub(r"\n?```\s*$", "", sql)
        sql = sql.rstrip(";")
        return sql

    def generate_answer(self, question: str, sql_result: Dict[str, Any]) -> str:
        if not sql_result.get("success"):
            return f"抱歉，数据查询遇到问题: {sql_result.get('error', '未知错误')}"

        rows = sql_result["rows"]
        if not rows:
            return "根据数据库中现有的数据，没有找到匹配的记录。"

        summary_prompt = self._build_summary_prompt(question, sql_result)
        resp = self.llm.invoke(summary_prompt)
        return resp.content.strip() if hasattr(resp, "content") else str(resp).strip()

    def _build_summary_prompt(self, question: str, sql_result: Dict[str, Any]) -> str:
        rows = sql_result["rows"]
        columns = sql_result["columns"]
        aggregates = sql_result.get("aggregates", {})

        if not rows:
            return ""

        agg_text = ""
        if aggregates and aggregates.get("sums"):
            agg_text = "\n系统预计算的精确汇总（请直接使用以下数值，禁止自行计算）：\n"
            agg_text += f"- 总行数: {aggregates['row_count']}\n"
            for col in aggregates.get("numeric_columns", []):
                agg_text += f"- {col} 合计: {aggregates['sums'][col]}, 平均值: {aggregates['avg'][col]:.2f}\n"

        return f"""根据以下 SQL 查询结果和系统预计算的精确汇总数据，用中文给用户一个清晰的回答。

用户问题：{question}
{agg_text}
查询明细数据（共 {len(rows)} 行，列: {', '.join(columns)}）：
{json.dumps(rows[:self.max_result_rows], ensure_ascii=False, indent=2)}

规则：
1. 所有汇总数字（合计、总计、总量等）**必须**使用上面"系统预计算的精确汇总"中的数值，绝对不要自己重新计算
2. 如果结果很少（≤20行），列出完整的明细
3. 如果结果较多，给出汇总分析和关键发现
4. 数值较大的金额用千分位或"万元"格式化
5. 突出关键数字和结论
6. 如果结果被截断，说明"共 {len(rows)} 条，仅展示前 {self.max_result_rows} 条" """

    def stream_generate_answer(self, question: str, sql_result: Dict[str, Any]):
        if not sql_result.get("success"):
            yield f"抱歉，数据查询遇到问题: {sql_result.get('error', '未知错误')}"
            return

        rows = sql_result["rows"]
        if not rows:
            yield "根据数据库中现有的数据，没有找到匹配的记录。"
            return

        summary_prompt = self._build_summary_prompt(question, sql_result)
        for chunk in self.llm.stream(summary_prompt):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                yield token

    def _detect_join_inflation(self, sql: str, aggregates: Dict[str, Any]) -> Optional[str]:
        sql_upper = sql.upper()
        if "JOIN" not in sql_upper:
            return None

        if "FROM (SELECT" in sql_upper:
            return None

        table_pattern = r'"(_[^"]+)"'
        tables = list(dict.fromkeys(re.findall(table_pattern, sql)))

        if len(tables) < 2:
            return None

        sums = aggregates.get("sums", {})
        if not sums:
            return None

        for col_name, inflated_total in sums.items():
            col_clean = col_name.strip().strip('"')
            for table in tables[:2]:
                try:
                    check_sql = f'SELECT SUM(CAST("{col_clean}" AS REAL)) FROM "{table}"'
                    rows, _ = self.table_store.execute_sql(check_sql)
                    if rows:
                        single_val = list(rows[0].values())[0]
                        if single_val is not None:
                            single_sum = float(single_val)
                            if single_sum > 0 and inflated_total > single_sum * 3:
                                return (
                                    f"JOIN 导致数据膨胀 {inflated_total / single_sum:.0f} 倍。"
                                    f"禁止对原始明细表直接 JOIN，必须先用子查询 GROUP BY 预聚合每个表，再对聚合后的结果做 JOIN 或 UNION ALL"
                                )
                except Exception:
                    continue

        return None