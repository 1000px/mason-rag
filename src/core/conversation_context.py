from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TurnContext:
    question: str
    mode: str
    tables: List[str] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    row_count: int = 0
    sql: Optional[str] = None
    summary: str = ""

    def brief(self) -> str:
        lines = [
            f"问题：{self.question}",
            f"涉及表：{', '.join(self.tables[:3])}" if self.tables else "",
            f"结果列：{', '.join(self.columns[:8])}" if self.columns else "",
            f"结果行数：{self.row_count}",
        ]
        return "\n".join(l for l in lines if l)


class ConversationContext:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.turns: List[TurnContext] = []

    def add_turn(self, turn: TurnContext):
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def last_turn(self) -> Optional[TurnContext]:
        return self.turns[-1] if self.turns else None

    def context_for_sql(self) -> str:
        if not self.turns:
            return ""
        last = self.turns[-1]
        if not last.columns or not last.tables:
            return ""
        parts = [
            "【上一轮查询上下文 — 如果用户的问题需要参考上一轮的查询结果，请使用以下信息】",
            last.brief(),
        ]
        if last.sql:
            parts.append(f"上一轮 SQL：{last.sql}")
        return "\n".join(parts)

    def is_likely_followup(self, question: str) -> bool:
        if not self.turns:
            return False
        if len(question) <= 10:
            return True
        followup_words = [
            "哪个", "哪些", "它", "这", "那", "其", "上面", "刚刚", "刚才",
            "最高", "最低", "最多", "最少", "最大", "最小", "第几", "占多少",
            "继续", "接着", "再", "还有", "其他", "别的", "另外",
            "具体", "明细", "展开", "详细", "查看", "看看",
        ]
        if any(w in question for w in followup_words):
            return True
        return False

    def extract_table_hint(self) -> List[str]:
        tables = []
        for turn in reversed(self.turns):
            for t in turn.tables:
                if t not in tables:
                    tables.append(t)
        return tables

    def all_column_names(self) -> List[str]:
        cols = []
        seen = set()
        for turn in reversed(self.turns):
            for c in turn.columns:
                if c not in seen:
                    cols.append(c)
                    seen.add(c)
        return cols


_conversations: Dict[str, ConversationContext] = {}


def get_context(session_id: str) -> ConversationContext:
    if session_id not in _conversations:
        _conversations[session_id] = ConversationContext()
    return _conversations[session_id]