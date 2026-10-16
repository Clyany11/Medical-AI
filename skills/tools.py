"""
skills/tools.py — Agent 工具函数集
设计原则:
  - 每个工具函数都是 async (不阻塞事件循环)
  - 接口统一: 输入为 dict, 输出为 dict
  - 工具定义 (get_tool_definitions) 遵循 OpenAI function calling 格式
  - 工具实现复用现有模块, 不重复造轮子
"""
import json
import logging
import re
import time
from typing import Dict, Any, List, Optional, Tuple
logger = logging.getLogger("AgentApp")
async def knowledge_search(
    query: str,
    collection_type: str = "modern_medicine",
    milvus_manager=None,
    ancient_books_manager=None,
) -> Dict[str, Any]:
    """
    知识库检索工具 — 从 Milvus 检索医学知识
    :param query: 搜索查询
    :param collection_type: "modern_medicine" 或 "ancient_books"
    :param milvus_manager: Milvus 管理器实例
    :param ancient_books_manager: 古籍 Milvus 管理器实例
    :return: {"context": str, "docs": list, "doc_count": int}
    """
    t0 = time.time()
    # 管理器选择逻辑:
    # 优先判断是否为古籍查询 — 当 collection_type 为 "ancient_books" 且古籍管理器存在时,
    # 使用 ancient_books_manager (对应古籍专用的 Milvus collection)
    # 否则回退到通用的 milvus_manager (现代医学 collection)
    # 两者都不可用时返回空结果, 避免后续空指针异常
    if collection_type == "ancient_books" and ancient_books_manager:
        manager = ancient_books_manager
    elif milvus_manager:
        manager = milvus_manager
    else:
        return {"context": "", "docs": [], "doc_count": 0, "error": "no manager"}
    try:
        # manager.search() 是异步调用 Milvus 混合检索 (IVF_FLAT 稠密向量 + BM25 稀疏检索 + RRF 融合排序,
        # 命中子块后按 parent_id 回表取父块正文),
        # 返回按相关性排序的文档列表
        results = await manager.search(query)
        # format_docs() 将检索结果文档列表转换为纯文本字符串,
        # 拼接后作为 LLM 生成回答时的上下文 (context) 注入 Prompt
        context = manager.format_docs(results) if results else ""
        duration = round((time.time() - t0) * 1000, 2)
        logger.info(
            f"[Tool:knowledge_search] collection={collection_type} | "
            f"结果数={len(results)} | 耗时={duration}ms"
        )
        return {
            "context": context,
            "docs": results,
            "doc_count": len(results),
            "duration_ms": duration,
        }
    except Exception as e:
        logger.error(f"[Tool:knowledge_search] 检索失败: {e}")
        return {"context": "", "docs": [], "doc_count": 0, "error": str(e)}
async def web_search_tool(query: str) -> Dict[str, Any]:
    """
    联网搜索工具 — 调用外部搜索 API
    :param query: 搜索查询
    :return: {"context": str, "urls": list, "success": bool}
    """
    t0 = time.time()
    try:
        # 延迟导入 (lazy import): web_search 模块依赖较重 (网络请求库等),
        # 仅在实际需要联网搜索时才加载, 避免影响启动速度和不必要的依赖
        from web_search import web_search_async
        search_result = await web_search_async(query)
        duration = round((time.time() - t0) * 1000, 2)
        context = search_result.get("search_context", "")
        # URL 提取: 遍历搜索结果列表, 收集所有非空的 URL;
        # 使用列表推导 + if 过滤掉空字符串和 None, 保证返回的 URL 列表都是有效链接
        urls = [
            r.get("url", "")
            for r in search_result.get("results", [])
            if r.get("url")
        ]
        logger.info(
            f"[Tool:web_search] 成功={search_result.get('success')} | "
            f"结果数={len(urls)} | 耗时={duration}ms"
        )
        return {
            "context": context,
            "urls": urls,
            "success": search_result.get("success", False),
            "duration_ms": duration,
        }
    except Exception as e:
        logger.error(f"[Tool:web_search] 搜索失败: {e}")
        return {"context": "", "urls": [], "success": False, "error": str(e)}
def extract_doc_urls(
    retrieval_docs: List, is_medical: bool = True, query: str = ""
) -> Tuple[str, str, List]:
    """
    从检索文档中提取图片和 PDF URL (带相关性过滤)
    设计要点:
      - 不额外调用 embedding API, 不增加 TTFT
      - 使用 query 与 doc.page_content 的字符级关键词重叠度作为轻量相关性代理
      - 重叠度不足时不返回 URL, 避免不相关配图展示在前端
      - 最多返回 1 张图片 + 1 个文档 URL (可同时存在)
    :param retrieval_docs: Milvus 检索结果文档列表
    :param is_medical: 是否医疗查询 (非医疗不返回 URL)
    :param query: 用户原始查询 (用于相关性过滤, 为空时跳过过滤直接返回 top-1)
    :return: (top_image_url, top_pdf_url, image_urls_list)
    """
    if not is_medical or not retrieval_docs:
        return "", "", []
    top_image_url = ""
    top_pdf_url = ""
    for doc in retrieval_docs:
        if not hasattr(doc, 'metadata'):
            continue
        # ---- 相关性过滤 ----
        # 当 query 非空时, 检查 doc.page_content 是否与 query 有足够的关键词重叠.
        # 原理: 如果 Milvus 召回的文档内容里连用户问题的核心词汇都不包含,
        # 那它关联的图片/PDF 也大概率与用户问题无关, 不应展示.
        # 阈值选择: 要求 query 中至少有 2 个双字符片段 (bigram) 在 doc 内容中出现,
        # 这个条件非常宽松 (一个 10 字的 query 有 9 个 bigram, 只需命中 2 个),
        # 所以相关文档几乎不会被误杀; 但完全无关的文档 (如 query 问"头疼",
        # 但 top-1 文档全文讲的是"糖尿病饮食") 的 bigram 重叠接近 0, 会被正确过滤.
        if query and hasattr(doc, 'page_content') and doc.page_content:
            if not _is_doc_relevant(query, doc.page_content):
                continue
        url = doc.metadata.get("image_url", "")
        if url and not top_image_url:
            # 验证是有效的 HTTP(S) URL (过滤本地路径, 空白, 损坏的链接格式)
            if _is_valid_url(url):
                top_image_url = url
        pdf_url = doc.metadata.get("pdf_url", "")
        if pdf_url and not top_pdf_url:
            # 同样验证 pdf_url, 并排除 .txt 文件链接 (txt 文件在浏览器中无法正常预览,
            # 用户体验差 — 前端截图显示点击 .txt 链接报错"无法打开")
            if _is_valid_url(pdf_url) and not pdf_url.lower().endswith(".txt"):
                top_pdf_url = pdf_url
        # 两者都找到后无需继续遍历
        if top_image_url and top_pdf_url:
            break
    # 使用现有的提取函数生成完整列表 (供 /chatbot 非流式接口的 image_urls 字段)
    image_urls_list = []
    try:
        from milvus_schema import extract_image_urls_from_search_results
        image_urls_list = extract_image_urls_from_search_results(retrieval_docs)
    except ImportError:
        pass
    return top_image_url, top_pdf_url, image_urls_list
def _is_valid_url(url: str) -> bool:
    """
    验证 URL 是否为有效的 HTTP/HTTPS 链接
    过滤掉:
      - 本地文件路径 (如 /data/xxx.pdf, C:\\xxx.pdf)
      - 空白或格式损坏的字符串
      - 非 http/https 协议的链接
    :return: True 表示是合法的可访问 URL
    """
    if not url or not url.strip():
        return False
    stripped = url.strip()
    return stripped.startswith("http://") or stripped.startswith("https://")
def _is_doc_relevant(query: str, doc_content: str, min_overlap: int = 2) -> bool:
    """
    轻量相关性判断 — 基于 bigram (双字符片段) 重叠
    设计考量:
      - 零 API 调用, 微秒级, 不影响 TTFT
      - bigram 比 unigram 更有区分力 (单字如"的""是"到处都有),
        又比分词 (jieba) 轻量 (不需要额外依赖和字典加载)
      - min_overlap=2: 只要有 2 个 bigram 命中即认为相关, 这是非常宽松的门槛,
        主要目的是过滤"完全无关"的文档, 而非精确判断相关性
      - 短查询 (≤6字) 自动降为 min_overlap=1, 避免过滤过严
    :param query: 用户问题 (通常 5~30 字)
    :param doc_content: 文档内容 (通常几百字)
    :param min_overlap: 最少需要命中的 bigram 数量 (长查询默认)
    :return: True 表示文档与 query 有基本相关性
    """
    if not query or not doc_content:
        return False
    # 提取 query 的所有 bigram (双字符滑动窗口)
    q = query.strip()
    if len(q) < 2:
        return True  # query 太短无法判断, 放行
    query_bigrams = {q[i: i + 2] for i in range(len(q) - 1)}
    # 去掉高频无意义 bigram (标点/停用词片段)
    query_bigrams -= {"的是", "是什", "什么", "怎么", "吗？", "吗?", "了吗", "么办", "吃什么"}
    if not query_bigrams:
        return True  # 过滤后无有效 bigram, 放行
    # 短查询 (有效 bigram ≤ 4 个) 降低阈值, 避免过滤过严
    # 例如 "头疼吃什么药" 过滤后仅剩 3 个有效 bigram, 要求 2 个命中太严格
    effective_threshold = 1 if len(query_bigrams) <= 4 else min_overlap
    # 在 doc_content 前 500 字中检查重叠 (文档很长时只看前部, 控制计算量)
    content_slice = doc_content[:500]
    overlap = sum(1 for bg in query_bigrams if bg in content_slice)
    return overlap >= effective_threshold
def get_tool_definitions() -> List[Dict[str, Any]]:
    """
    获取所有工具的定义 (用于 Agent 工具注入)
    格式兼容 OpenAI function calling schema
    """
    # 返回 OpenAI function-calling 兼容的工具定义列表,
    # 每个工具包含 name, description, parameters (JSON Schema 格式)
    # Agent 可以根据这些定义决定调用哪个工具
    #
    # 注意: "handler" 字段是自定义扩展 (非 OpenAI 标准格式),
    # 它存储了工具对应的实际 async 函数引用, 供 Orchestrator 直接调用,
    # 避免通过字符串名称再做一次函数查找映射
    return [
        {
            "name": "knowledge_search",
            "description": "从医学知识库中检索相关文档, 适用于需要医学知识回答的问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询文本",
                    },
                    "collection_type": {
                        "type": "string",
                        "enum": ["modern_medicine", "ancient_books"],
                        "description": "知识库类型: modern_medicine (现代医学) 或 ancient_books (古籍)",
                    },
                },
                "required": ["query"],
            },
            "handler": knowledge_search,
        },
        {
            "name": "web_search",
            "description": "联网搜索获取最新信息, 适用于需要实时数据, 非医学领域问题, 或知识库中没有的内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询文本",
                    },
                },
                "required": ["query"],
            },
            "handler": web_search_tool,
        },
    ]


# ============================================================================
# Neo4j 图检索工具 — 把自然语言问题转成 Cypher, 查图后作为上下文交给大模型
# 设计要点:
#   - schema 运行时自动内省 (db.labels / db.relationshipTypes / db.propertyKeys), 进程内 TTL 缓存
#   - Text2Cypher: 用 LLM 把问题转成只读 Cypher
#   - 只读双重保证: Neo4j 会话 READ_ACCESS + 关键字黑名单校验
#   - 结果格式化为纯文本, 与 Milvus context 拼接后一起注入 Prompt
#   - 未配置 / 未连接 Neo4j 时静默返回空 context, 不影响主链路
# ============================================================================

# 图结构 schema 缓存: 启动时预热一次, 之后进程生命周期内永不再查 (图结构固定不变)
_SCHEMA_CACHE: Dict[str, str] = {"text": ""}

# Cypher 写操作 / 高风险过程调用黑名单 (词边界匹配, 避免误伤含 set 的单词)
_CYPHER_FORBIDDEN_RE = re.compile(
    r"\b(create|merge|delete|detach|set|remove|drop|foreach|call|apoc|load|using)\b",
    re.IGNORECASE,
)


async def _query_graph_schema(neo4j_manager) -> str:
    """实际执行 3 条内省语句, 把图结构拼成一段文字描述 (不读缓存)"""
    labels: List[str] = []
    rel_types: List[str] = []
    prop_keys: List[str] = []
    try:
        r = await neo4j_manager.run_read(
            "CALL db.labels() YIELD label RETURN label"
        )
        labels = [row.get("label") for row in r.get("rows", []) if row.get("label")]
        r = await neo4j_manager.run_read(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        )
        rel_types = [
            row.get("relationshipType")
            for row in r.get("rows", [])
            if row.get("relationshipType")
        ]
        r = await neo4j_manager.run_read(
            "CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey"
        )
        prop_keys = [
            row.get("propertyKey") for row in r.get("rows", []) if row.get("propertyKey")
        ]
    except Exception as e:
        logger.warning(f"[graph_search] schema 内省失败: {e}")
        return ""

    if not labels and not rel_types:
        return ""

    lines = []
    if labels:
        lines.append("- 节点标签 (Label): " + ", ".join(labels))
    if rel_types:
        lines.append("- 关系类型 (Relationship): " + ", ".join(rel_types))
    if prop_keys:
        lines.append("- 属性字段 (Property): " + ", ".join(prop_keys))
    return "\n".join(lines)


async def warmup_graph_schema(neo4j_manager) -> str:
    """
    启动时预热图结构: 查一次并写入进程内缓存, 之后永不再查.
    图结构固定不变, 由 main lifespan 在 Neo4j 初始化后调用; 未连接时静默跳过.
    """
    if neo4j_manager is None or not neo4j_manager.is_healthy():
        return ""
    schema_text = await _query_graph_schema(neo4j_manager)
    if schema_text:
        _SCHEMA_CACHE["text"] = schema_text
        logger.info(f"[graph_search] 图结构已预热 (仅查一次):\n{schema_text}")
    else:
        logger.warning("[graph_search] 图结构预热为空 (图为空或内省失败)")
    return schema_text


async def _introspect_graph_schema(neo4j_manager) -> str:
    """读取图结构: 优先用启动时预热好的缓存; 缓存为空时惰性查一次兜底"""
    if _SCHEMA_CACHE["text"]:
        return _SCHEMA_CACHE["text"]
    schema_text = await _query_graph_schema(neo4j_manager)
    if schema_text:
        _SCHEMA_CACHE["text"] = schema_text
    return schema_text


def _extract_cypher(text: str) -> str:
    """从 LLM 输出中提取 Cypher: 去掉 markdown 代码块围栏与多余说明"""
    if not text:
        return ""
    s = text.strip()
    # 去掉 ```cypher ... ``` / ``` ... ``` 围栏, 取第一个代码块的内容
    if "```" in s:
        parts = s.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 1:  # 奇数段 = 围栏内部内容
                s = part
                break
    s = s.strip()
    # 去掉可能残留的语言标识行 (如 "cypher")
    if s.lower().startswith("cypher"):
        s = s[len("cypher"):].lstrip()
    return s.strip()


def _is_safe_read_cypher(cypher: str) -> bool:
    """校验 Cypher 是否为单条只读语句 (关键字黑名单 + 语句数 + 必须含 RETURN)"""
    if not cypher:
        return False
    low = cypher.lower()
    if _CYPHER_FORBIDDEN_RE.search(cypher):
        return False
    # 去掉结尾分号后不允许再出现分号 (防止多语句注入)
    if ";" in cypher.rstrip().rstrip(";"):
        return False
    # 纯读查询必须带 RETURN
    if "return" not in low:
        return False
    return True


async def graph_search(
    query: str,
    llm_client=None,
    model: str = "deepseek-chat",
    neo4j_manager=None,
) -> Dict[str, Any]:
    """
    图检索工具 — 自然语言问题 → Cypher → 查 Neo4j → 上下文文本
    :param query: 用户问题
    :param llm_client: LLM 客户端 (用于 Text2Cypher)
    :param model: 生成 Cypher 使用的模型
    :param neo4j_manager: Neo4j 管理器实例
    :return: {"context": str, "rows": list, "cypher": str, "count": int}
    """
    t0 = time.time()
    if neo4j_manager is None or not neo4j_manager.is_healthy():
        return {"context": "", "rows": [], "cypher": "", "count": 0, "error": "neo4j not available"}
    if llm_client is None:
        return {"context": "", "rows": [], "cypher": "", "count": 0, "error": "no llm_client"}

    schema_text = await _introspect_graph_schema(neo4j_manager)
    if not schema_text:
        return {"context": "", "rows": [], "cypher": "", "count": 0, "error": "empty graph or schema"}

    prompt = (
        "你是 Neo4j Cypher 专家。请根据下面的图数据库结构, 把用户问题转换成一条【只读】Cypher 查询。\n\n"
        f"图结构:\n{schema_text}\n\n"
        "要求:\n"
        "1. 只输出一条 Cypher 语句, 不要任何解释文字, 不要 markdown 代码块。\n"
        "2. 只允许 MATCH / OPTIONAL MATCH / WHERE / WITH / RETURN / ORDER BY / LIMIT。\n"
        "3. 严禁 CREATE / MERGE / DELETE / SET / REMOVE / CALL。\n"
        "4. 用中文别名返回有意义的字段, 结果加 LIMIT 20。\n"
        "5. 如果问题与图数据无关, 只输出: RETURN null\n\n"
        f"用户问题: {query}\n"
        "Cypher:"
    )
    try:
        resp = await llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"[graph_search] Text2Cypher 调用失败: {e}")
        return {"context": "", "rows": [], "cypher": "", "count": 0, "error": str(e)}

    cypher = _extract_cypher(raw)
    if not _is_safe_read_cypher(cypher):
        logger.warning(f"[graph_search] 生成的 Cypher 被拒绝: {cypher[:120]}")
        return {"context": "", "rows": [], "cypher": cypher, "count": 0, "error": "unsafe or invalid cypher"}

    result = await neo4j_manager.run_read(cypher)
    rows = result.get("rows", [])
    if not rows:
        return {"context": "", "rows": [], "cypher": cypher, "count": 0, "error": result.get("error", "")}

    # 格式化为上下文文本, 与 Milvus context 拼接后一起注入 Prompt
    lines = ["【知识图谱查询结果】"]
    for i, row in enumerate(rows, 1):
        lines.append(f"{i}. {json.dumps(row, ensure_ascii=False)}")
    context = "\n".join(lines)

    duration = round((time.time() - t0) * 1000, 2)
    logger.info(
        f"[Tool:graph_search] cypher={cypher[:80]} | "
        f"结果数={len(rows)} | 耗时={duration}ms"
    )
    return {
        "context": context,
        "rows": rows,
        "cypher": cypher,
        "count": len(rows),
        "duration_ms": duration,
    }