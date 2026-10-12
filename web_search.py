"""web_search.py — 意图识别 / 医疗领域判断 / 联网搜索 (规则引擎版)

本模块是 RouterAgent 的规则底座, 全部为纯规则 (关键词 + 正则), 不调用 LLM,
因此路由耗时 < 1ms。

对外提供:
  - classify_query_intent(query) -> "garbage" | "identity" | "greeting" | "medical" | "general"
  - is_medical_query(query)      -> bool
  - needs_web_search(query)      -> bool
  - web_search_async(query)      -> {"success": bool, "search_context": str, "results": list}
  - 静态话术常量: GARBAGE_RESPONSE / GREETING_RESPONSE / IDENTITY_RESPONSE / NON_MEDICAL_DISCLAIMER

联网搜索说明:
  web_search_async 走 DeepSeek Responses API 的原生 web_search 工具 (/responses 接口),
  由 DeepSeek 服务端托管执行联网, 复用 DEEPSEEK_API_KEY, 无需第三方搜索 key。
  注意: DeepSeek 的 /chat/completions 不支持服务端联网; 且实测仅 deepseek-v4-pro
  能真正触发 web_search (flash 系列不会联网, 只会凭自身知识作答)。
  未配置 key 或调用失败时返回空结果, 调用方退化为纯 LLM 作答, 不影响主链路可用性。
"""
import logging
import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv

logger = logging.getLogger("WebSearch")

load_dotenv()

# ==================== DeepSeek 原生联网搜索配置 ====================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
# 联网搜索专用模型: 实测 deepseek-v4-pro 可触发, deepseek-v4-flash 不可
WEBSEARCH_MODEL = os.getenv("DEEPSEEK_WEBSEARCH_MODEL", "deepseek-v4-pro")

# 惰性单例客户端, 避免模块导入阶段就建立连接
_client = None

# ==================== 静态话术 ====================
GARBAGE_RESPONSE = "抱歉, 我没有理解您的问题。请描述得更具体一些, 例如您的症状或想了解的健康问题。"
GREETING_RESPONSE = "您好! 我是您的智能健康助手, 可以为您解答医疗健康相关问题。请问有什么可以帮您?"
IDENTITY_RESPONSE = (
    "我是您的智能健康助手, 专注于医疗健康领域的问答服务。"
    "如果您有健康方面的疑问, 欢迎随时向我提问。"
)
NON_MEDICAL_DISCLAIMER = "【提示】本系统主要面向医疗健康领域, 以下回答仅供参考。\n\n"

# ==================== 关键词表 ====================
# 问候语 (短句精确匹配, 避免"你好, 我头疼"被误判为问候)
GREETING_KEYWORDS = [
    "你好", "您好", "哈喽", "嗨", "hi", "hello", "hey",
    "早上好", "中午好", "下午好", "晚上好", "早安", "晚安",
    "在吗", "在不在", "谢谢", "多谢", "感谢", "再见", "拜拜", "bye",
]

# 身份/模型询问 (走快车道, 返回固定话术, 避免暴露底层大模型信息)
IDENTITY_KEYWORDS = [
    "你是谁", "你叫什么", "你是什么", "你的名字", "你是什么模型",
    "你用的什么模型", "用的什么模型", "什么模型", "什么大模型", "哪个大模型",
    "你是gpt", "你是ai", "你是人工智能", "你是机器人", "你是人吗",
    "谁开发的", "开发者是谁", "你是豆包", "你是claude", "你是deepseek",
]

# 医疗领域关键词 (症状 / 疾病 / 诊疗 / 科室 / 指标 / 中医 / 健康)
MEDICAL_KEYWORDS = [
    # 症状
    "发烧", "发热", "咳嗽", "头痛", "头晕", "恶心", "呕吐", "腹泻", "腹痛", "胃痛",
    "胸痛", "心慌", "心悸", "失眠", "乏力", "过敏", "皮疹", "水肿", "便秘", "出血",
    "疼痛", "疼", "痛", "炎症", "感染", "抽搐", "麻木", "流鼻涕", "咽喉",
    # 疾病
    "感冒", "流感", "肺炎", "高血压", "低血压", "糖尿病", "冠心病", "心梗", "中风",
    "脑梗", "贫血", "肝炎", "肾炎", "胃溃疡", "胃炎", "关节炎", "骨质疏松", "痛风",
    "甲状腺", "肿瘤", "癌症", "癌", "哮喘", "结核", "鼻炎", "咽炎", "湿疹", "抑郁",
    "焦虑", "癫痫", "结石", "痔疮", "前列腺", "阴道炎", "痛经", "月经", "肌瘤",
    # 诊疗
    "治疗", "诊断", "用药", "药物", "副作用", "剂量", "服药", "打针", "输液", "手术",
    "检查", "化验", "体检", "血常规", "疫苗", "接种", "处方", "禁忌", "复发",
    "预防", "康复", "护理", "医嘱", "住院", "急诊",
    # 通用医学问法 (覆盖"有哪些症状/吃什么药/是什么病/怎么治"等自然表述)
    "症状", "体征", "病因", "病情", "发病", "并发症", "中毒",
    "不良反应", "治愈", "疗效", "药",
    # 科室 / 角色
    "医生", "医院", "门诊", "科室", "内科", "外科", "儿科", "妇科", "口腔",
    "皮肤科", "眼科", "骨科", "中医", "西医", "药师", "护士",
    # 指标
    "血糖", "血压", "血脂", "心率", "体温", "白细胞", "血小板", "肝功能", "肾功能",
    # 中医
    "针灸", "艾灸", "中药", "草药", "方剂", "经络", "气血", "阴虚", "阳虚", "上火",
    "舌苔", "脉象", "汤药", "拔罐", "推拿", "穴位", "养生",
    # 健康
    "饮食", "营养", "维生素", "蛋白质", "减肥", "运动", "睡眠", "保健",
]

# 时效性关键词 (需要联网搜索才能回答的问题)
WEB_SEARCH_KEYWORDS = [
    "天气", "气温", "空气质量", "pm2.5", "新闻", "实时", "股价", "股票",
    "汇率", "油价", "金价", "航班", "高铁", "火车票", "比分", "赛程",
    "票房", "排行榜", "热搜", "热榜", "导航", "路线规划",
    "现在几点", "今天几号", "最新", "疫情",
]


# ==================== 规则函数 ====================
def _normalize(query: str) -> str:
    """统一小写, 便于匹配含英文的关键词"""
    return (query or "").strip().lower()


def _is_garbage(query: str) -> bool:
    """判断是否为无意义输入 (过短 / 无有效字符 / 纯重复字符)"""
    text = (query or "").strip()
    if len(text) < 2:
        return True
    # 至少包含一个中文、字母或数字, 否则视为纯符号/表情
    if not re.search(r"[\u4e00-\u9fff0-9a-zA-Z]", text):
        return True
    # 同一字符重复 (如 "aaaaa" / "。。。。。")
    if len(set(text)) == 1:
        return True
    return False


def _is_identity(query: str) -> bool:
    """判断是否为身份/模型询问"""
    text = (query or "").strip()
    lower = _normalize(text)
    for keyword in IDENTITY_KEYWORDS:
        if keyword in text or keyword in lower:
            return True
    return False


def _is_greeting(query: str) -> bool:
    """判断是否为打招呼 (仅对短句生效, 避免长句中含"你好"被误判)"""
    text = (query or "").strip()
    lower = _normalize(text)
    # 长句 (含标点或超过 6 字) 不走问候快车道
    if len(text) > 6:
        return False
    for keyword in GREETING_KEYWORDS:
        if text == keyword or lower == keyword:
            return True
    return False


def is_medical_query(query: str) -> bool:
    """医疗领域深度匹配: 命中任一医疗关键词即认为属于医疗领域"""
    text = (query or "").strip()
    if not text:
        return False
    lower = _normalize(text)
    for keyword in MEDICAL_KEYWORDS:
        if keyword in text or keyword in lower:
            return True
    return False


def needs_web_search(query: str) -> bool:
    """判断是否需要联网搜索 (时效性 / 实时信息需求)"""
    text = (query or "").strip()
    if not text:
        return False
    lower = _normalize(text)
    for keyword in WEB_SEARCH_KEYWORDS:
        if keyword in text or keyword in lower:
            return True
    return False


def classify_query_intent(query: str) -> str:
    """粗粒度意图分类

    :return: "garbage" | "identity" | "greeting" | "medical" | "general"
    """
    if _is_garbage(query):
        return "garbage"
    if _is_identity(query):
        return "identity"
    if _is_greeting(query):
        return "greeting"
    if is_medical_query(query):
        return "medical"
    return "general"


# ==================== 联网搜索 ====================
def _get_client():
    """惰性创建 DeepSeek 异步客户端 (OpenAI 兼容)"""
    global _client
    if _client is None:
        from openai import AsyncOpenAI

        _client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return _client


async def web_search_async(query: str) -> Dict[str, Any]:
    """联网搜索 — 调用 DeepSeek Responses API 的原生 web_search 工具

    由 DeepSeek 服务端托管执行联网 (tools=[{"type":"web_search"}]), 模型自行发起
    多次 search / open_page 并把结果综合成带来源的回答, 作为 search_context 返回。

    失败或未触发搜索时返回空结果 (success=False), 调用方退化为纯 LLM 作答。

    :return: {"success": bool, "search_context": str, "results": List[{"url": str}]}
    """
    query = (query or "").strip()
    if not query:
        return {"success": False, "search_context": "", "results": []}
    if not DEEPSEEK_API_KEY:
        logger.warning("[WebSearch] 未配置 DEEPSEEK_API_KEY, 跳过联网搜索")
        return {"success": False, "search_context": "", "results": []}

    try:
        client = _get_client()
        resp = await client.responses.create(
            model=WEBSEARCH_MODEL,
            tools=[{"type": "web_search"}],
            tool_choice={"type": "web_search"},
            input=query,
        )
    except Exception as e:
        logger.error(f"[WebSearch] DeepSeek 联网搜索调用失败: {e}")
        return {"success": False, "search_context": "", "results": []}

    # output_text: 模型综合搜索结果后的回答 (已含引用来源)
    context = (getattr(resp, "output_text", "") or "").strip()

    # 从 web_search_call 的 open_page 动作里收集来源 URL, 去掉 DeepSeek 追加的 ws_call_id 片段
    urls: List[str] = []
    triggered = False
    for item in getattr(resp, "output", None) or []:
        if getattr(item, "type", "") != "web_search_call":
            continue
        triggered = True
        action = getattr(item, "action", None)
        url = getattr(action, "url", None) if action is not None else None
        if url:
            clean = url.split("#ws_call_id=")[0]
            if clean and clean not in urls:
                urls.append(clean)

    success = bool(context) and triggered
    logger.info(
        f"[WebSearch] DeepSeek 联网 | 触发={triggered} | 来源数={len(urls)} | "
        f"context长度={len(context)} | query: {query[:30]}"
    )
    return {
        "success": success,
        "search_context": context,
        "results": [{"url": u} for u in urls],
    }