"""cache_strategy.py — 问答缓存策略分类器 (纯规则, 零 LLM / 零外部 I/O)

背景:
  原流式缓存 Key = md5(query + ctx:history_digest + user:user_id), 其中 history_digest
  来自最近 5 条消息的哈希. 由于对话历史持续变化, 用户重复询问同一个独立医学问题时,
  也会因 history_digest 变化而无法命中之前的缓存.

  本模块提供:
    - CacheStrategy 枚举 (INDEPENDENT / CONTEXTUAL / SENSITIVE_OR_UNCERTAIN)
    - normalize_query(query)                 安全文本规范化 (仅去空白, 不归并语义)
    - classify_cache_strategy(query, history) 问题类型判定
    - decide_cache_query(query, history)      返回 (strategy, key_text)

  缓存策略:
    - INDEPENDENT            → 用规范化 query 生成稳定 Key (不含历史/用户), 可跨无关历史复用
    - CONTEXTUAL             → 依赖历史指代/省略信息, 跳过普通问答缓存读写
    - SENSITIVE_OR_UNCERTAIN → 个体化医疗决策 / 无法可靠分类, 跳过普通问答缓存读写

  分类原则 (保守优先): 规则无法可靠判定为"独立通用知识问题"时, 一律归入
  CONTEXTUAL 或 SENSITIVE_OR_UNCERTAIN, 绝不默认归入 INDEPENDENT, 避免错误复用
  不同患者 / 不同医疗场景下的答案.
"""
import re
from enum import Enum
from typing import Tuple


class CacheStrategy(str, Enum):
    """缓存策略枚举 (继承 str, 便于直接拼进日志)"""
    INDEPENDENT = "INDEPENDENT"
    CONTEXTUAL = "CONTEXTUAL"
    SENSITIVE_OR_UNCERTAIN = "SENSITIVE_OR_UNCERTAIN"


# 独立通用知识问题使用的缓存 Key 前缀 (带 v2 版本号, 与旧格式 Key 隔离, 避免旧数据误命中)
INDEPENDENT_STREAM_PREFIX = "llm:stream:v2:"
INDEPENDENT_CACHE_PREFIX = "llm:cache:v2:"


# ---- 空白字符 (含全角空格 U+3000), 仅用于规范化 ----
_WHITESPACE_RE = re.compile(r"[\s\u3000]+")


def normalize_query(query: str) -> str:
    """安全文本规范化: 仅去除首尾空白 + 合并多余空白.

    刻意不做更激进的处理: 不归并不同药名 / 剂型 / 剂量 / 否定词 / 医学术语,
    保证"高血压怎么治疗"和"高血压 怎么  治疗"能命中同一缓存, 但不会把
    "阿莫西林"与"阿奇霉素"归并成同一问题.
    """
    if not query:
        return ""
    return _WHITESPACE_RE.sub(" ", query).strip()


# ---- 上下文指代 (依赖历史) 判定 ----
# 匹配: 第三人称指代 / 指示代词 / 明确回指短语 / 序数回指 (第X个/种)
# 说明:
#   - 单字 "他/它" 排除 "其他/其它" (其中 "其他/其它" 不构成回指)
#   - "这" 及 "这~" 复合词 (这个/这些/这样/这种/这里...) 均视为指示代词
#   - "那" 及 "那~" 复合词视为回指, 但排除连词 "那么"
_CONTEXTUAL_RE = re.compile(
    r"(?<!其)它|(?<!其)他|她|"
    r"这|"
    r"那(?!么)|"
    r"此药|此病|此症状|此方法|此情况|"
    r"该药|该药物|该病|该症状|该疾病|"
    r"刚才说的|前面说的|之前说的|上面说的|以上说的|"
    r"前面提到的|之前提到的|上面提到的|"
    r"第[一二三四五六七八九十百0-9]+[个种条项]|"
    r"上一个|前一个|"
    r"和上一个|和上面|和之前|跟这个|跟那个"
)

# ---- 个体化 (第一人称) 判定 ----
# 医学问答中, "我/我的/本人" 几乎总是用户针对自身情况的个体化询问
# (如 "我胃疼吃什么药" / "阿莫西林适合我吗" / "根据我的肾功能..."),
# 这类答案与个体强相关, 不可作为通用知识共享.
_PERSONAL_WORDS = ("我", "本人")

# ---- 问题句判定 (用于识别 INDEPENDENT 通用知识问题) ----
_QUESTION_MARKERS = (
    "什么", "怎么", "如何", "为什么", "为何",
    "哪", "多少", "多久", "区别", "是否", "有无",
)
_QUESTION_PARTICLES = ("吗", "呢", "么")


def _is_contextual(q: str) -> bool:
    return bool(_CONTEXTUAL_RE.search(q))


def _is_personal(q: str) -> bool:
    return any(w in q for w in _PERSONAL_WORDS)


def _is_question_like(q: str) -> bool:
    if "？" in q or "?" in q:
        return True
    if any(m in q for m in _QUESTION_MARKERS):
        return True
    return q.endswith(_QUESTION_PARTICLES)


def classify_cache_strategy(query: str, history: str = "") -> str:
    """将用户问题分类为 INDEPENDENT / CONTEXTUAL / SENSITIVE_OR_UNCERTAIN.

    判定顺序 (后三项均为"跳过缓存", 只有 INDEPENDENT 会进入缓存读写):
      1. 空问题                     → SENSITIVE_OR_UNCERTAIN
      2. 含上下文指代/回指          → CONTEXTUAL
      3. 含第一人称 (个体化)         → SENSITIVE_OR_UNCERTAIN
      4. 问题句式 (通用知识提问)     → INDEPENDENT
      5. 其余 (无法可靠判定)         → SENSITIVE_OR_UNCERTAIN (保守)

    :param query:   用户问题原文
    :param history: 格式化后的历史上下文字符串 (当前规则以 query 为主, history 预留用于后续细化)
    :return: CacheStrategy 值 (字符串)
    """
    q = (query or "").strip()
    if not q:
        return CacheStrategy.SENSITIVE_OR_UNCERTAIN
    if _is_contextual(q):
        return CacheStrategy.CONTEXTUAL
    if _is_personal(q):
        return CacheStrategy.SENSITIVE_OR_UNCERTAIN
    if _is_question_like(q):
        return CacheStrategy.INDEPENDENT
    return CacheStrategy.SENSITIVE_OR_UNCERTAIN


def decide_cache_query(query: str, history: str = "") -> Tuple[str, str]:
    """计算缓存策略与用于生成 Redis Key 的查询文本.

    :return: (strategy, key_text)
      - INDEPENDENT: key_text 为规范化后的 query (不含历史摘要 / 用户标识)
      - 其他:        key_text 为空字符串, 表示跳过缓存
    """
    strategy = classify_cache_strategy(query, history)
    if strategy == CacheStrategy.INDEPENDENT:
        return strategy, normalize_query(query)
    return strategy, ""
