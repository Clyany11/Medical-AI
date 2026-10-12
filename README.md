# Agent_online — 医疗智能问答 Agent

基于 **RAG(检索增强生成) + Multi-Agent(多智能体) + 数据飞轮** 的医疗领域智能问答系统。

系统以 FastAPI 为服务框架,整合了意图路由、混合向量检索、大模型生成、质量反思、三层记忆、Redis 多层缓存、语音全链路(ASR/TTS)、知识图谱查询、用户认证与数据飞轮反馈等能力,面向医疗场景提供低延迟、可解释、可持续进化的问答服务。

---

## 目录

- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [环境变量配置](#环境变量配置)
- [API 接口](#api-接口)
- [核心流程说明](#核心流程说明)
- [数据初始化](#数据初始化)
- [测试](#测试)
- [日志与监控](#日志与监控)

---

## 核心特性

| 能力 | 说明 |
| --- | --- |
| **Multi-Agent 编排** | RouterAgent(规则路由,零 LLM 调用) → Skills(工具) → MedicalAgent(RAG 生成) → ReflectionAgent(四维度质量反思与自我修正) |
| **RAG 混合检索** | Milvus 稠密向量(HNSW/IVF_FLAT)+ BM25 稀疏检索 + RRF 融合排序 + Rerank 精排 |
| **双知识库** | 现代医学文献库 + 中医古籍库,按意图自动切换检索集合 |
| **Redis 三层缓存防护** | 防击穿(分布式锁 + 双重检查)、防穿透(`<EMPTY>` 占位)、防雪崩(±10% TTL 抖动) |
| **三层记忆机制** | 短期记忆(Redis List / LRU 降级)、长期记忆(Redis Hash)、语义记忆(Milvus) |
| **ReAct 循环** | Reasoning + Acting,支持工具调用(知识库检索 / 联网搜索 / 图谱查询) |
| **联网搜索** | 规则引擎判断是否需要联网 + 医疗相关性过滤 + 文档 URL 提取 |
| **知识图谱** | Neo4j 图数据库,支持自然语言转 Cypher(Text2Cypher)只读查询 |
| **语音全链路** | ASR(豆包)→ RAG + LLM → TTS(豆包),支持流式合成 |
| **数据飞轮** | 交互记录、👍/👎 反馈、专家审核队列,支撑模型持续优化 |
| **用户认证** | JWT 令牌 + bcrypt 密码哈希(注册 / 登录 / 会话鉴权) |
| **会话管理** | 基于 MySQL 的多轮会话与历史消息持久化 |
| **Token 计费** | Token 用量统计与计费定时任务 |
| **可观测性** | 审计中间件、`/health` 健康检查、`/metrics` 运行时指标、降级管理 |

---

## 系统架构

```
                          ┌────────────────────────────┐
   用户请求  ───────────▶ │      FastAPI (端口 8103)     │
                          └──────────────┬─────────────┘
                                         │
                          ┌──────────────▼─────────────┐
                          │   RouterAgent  (规则路由)    │   < 1ms, 零 LLM 调用
                          │  web_search / medical_rag   │
                          │  / chitchat / greeting ...  │
                          └──────────────┬─────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              ▼                          ▼                          ▼
     ┌─────────────────┐      ┌────────────────────┐      ┌──────────────────┐
     │  联网搜索 Skill  │      │  RAG 混合检索 Skill │      │  图谱查询 Skill   │
     │  web_search     │      │  Milvus + BM25      │      │  Neo4j Text2Cypher│
     │                 │      │  + RRF + Rerank     │      │                   │
     └────────┬────────┘      └──────────┬─────────┘      └─────────┬─────────┘
              │                          │                          │
              └──────────────────────────┼──────────────────────────┘
                                         ▼
                          ┌─────────────────────────────┐
                          │   MedicalAgent (RAG 生成)     │
                          │   DeepSeek LLM 流式/非流式     │
                          └──────────────┬──────────────┘
                                         │  异步
                          ┌──────────────▼──────────────┐
                          │  ReflectionAgent (质量反思)   │
                          │  准确性/完整性/安全性/相关性   │
                          └──────────────┬──────────────┘
                                         ▼
                          ┌─────────────────────────────┐
                          │  记忆持久化 + 数据飞轮记录     │
                          │  短期/长期/语义记忆 + 反馈队列 │
                          └─────────────────────────────┘
```

请求链路:`用户请求 → 意图路由 → 检索/搜索 → 生成 → 反思 → 记忆与反馈记录`。

---

## 技术栈

- **服务框架**:FastAPI + Uvicorn
- **大模型**:DeepSeek(OpenAI 兼容接口)
  - `deepseek-flash`:快路径(禁用 thinking),用于低延迟问答
  - `deepseek-v4-pro`:深度思考 / 联网场景
- **向量检索**:Milvus(Milvus Lite 本地文件模式)
- **Embedding**:智谱 `embedding-3`(2048 维,**必须与旧向量库 `pdf_agent.db` 向量空间一致**)
- **缓存 / 记忆**:Redis
- **关系数据库**:MySQL(库名 `aidoctor`)
- **图数据库**:Neo4j
- **对象存储**:阿里云 OSS(语音 / 图片)
- **语音**:豆包 ASR / TTS(WebSocket 协议,见 [protocols/protocols.py](file:///e:/AI/Agent_online/protocols/protocols.py))
- **认证**:JWT + bcrypt
- **测试/压测**:pytest 风格的 smoke / regression 套件 + Locust 压测

---

## 目录结构

```
Agent_online/
├── app.py                     # 主入口:FastAPI 应用、路由、缓存、RAG 主流程
├── .env                       # 环境变量(服务凭证与连接配置,勿提交)
├── aidoctor.sql               # MySQL 数据库初始化脚本
│
├── agents/                    # Multi-Agent 架构
│   ├── base.py                #   BaseAgent:ReAct + Reflection + Self-Correction
│   ├── router_agent.py        #   规则路由(意图识别,零 LLM 调用)
│   ├── medical_agent.py       #   医学问答生成 Agent(RAG)
│   ├── reflection_agent.py    #   质量反思与修正 Agent
│   └── orchestrator.py        #   编排器:execute / execute_stream
│
├── skills/
│   └── tools.py               # Agent 工具:知识库检索/联网搜索/图谱查询
│
├── memory/
│   └── manager.py             # 三层记忆:短期/长期/语义
│
├── protocols/                 # 豆包语音 WebSocket 二进制协议
│   └── protocols.py
├── asr_websocket.py           # ASR(语音识别)WebSocket 客户端
├── auc_http_demo.py           # ASR HTTP 调用示例
├── tts_bidirection.py         # TTS 双向流式
├── tts_http_demo.py           # TTS HTTP 调用示例
├── voice_pipeline.py          # 语音全链路路由(ASR → RAG+LLM → TTS)
│
├── milvus_db.py               # Milvus 向量库管理器(现代医学 + 古籍)
├── milvus_schema.py           # 集合 Schema 与图片 URL 提取
├── vectors.py / vectors_redis.py  # 向量化工具
├── build_medical_milvus.py    # 构建现代医学向量库
├── build_ancient_books.py     # 构建中医古籍向量库
├── neo4j_db.py                # Neo4j 图数据库管理器
├── redis_db.py                # Redis 缓存/记忆管理器(单例)
├── model.py                   # DeepSeek LLM 客户端工厂
├── models.py                  # NL2Cypher 相关 Pydantic 模型
│
├── chat_service.py            # 会话管理(MySQL:会话/消息)
├── auth.py                    # JWT 认证(注册/登录/鉴权)
├── web_search.py              # 联网搜索与意图判断规则引擎
├── feedback_logger.py         # 数据飞轮:交互记录
├── feedback_routes.py         # 数据飞轮路由:反馈/专家审核
├── token_billing.py           # Token 计费
├── billing_scheduler.py       # 计费定时任务
├── middleware.py              # 审计中间件 / 降级管理 / 扩展注册
├── testAliyunfiles.py         # 阿里云 OSS 上传
│
├── index.html / chat.html / login.html   # 前端页面
│
├── smoke_test_suite/          # 冒烟测试
├── regression_test/           # 回归测试
└── stress_test/               # Locust 压测
```

---

## 快速开始

### 1. 环境要求

- Python 3.10+
- 可访问的 Redis / MySQL / Neo4j / Milvus(或使用 Milvus Lite 本地文件)
- 阿里云 OSS、豆包语音、DeepSeek、智谱 等服务凭证

### 2. 安装依赖

项目未内置 `requirements.txt`,请按需安装(核心依赖):

```bash
pip install fastapi uvicorn python-dotenv pydantic \
            openai redis pymysql neo4j pymilvus \
            langchain-core requests aiohttp websockets \
            bcrypt pyjwt oss2 rank_bm25
```

### 3. 配置环境变量

复制并按实际情况填写 `.env`(见下方 [环境变量配置](#环境变量配置))。

### 4. 初始化数据

```bash
# 初始化 MySQL(会话、用户、反馈等表)
mysql -u root -p < aidoctor.sql

# 构建向量库(首次或数据更新时执行)
python build_medical_milvus.py     # 现代医学向量库
python build_ancient_books.py      # 中医古籍向量库
```

### 5. 启动服务

```bash
python app.py
```

服务默认监听 `0.0.0.0:8103`。

- 接口文档:http://127.0.0.1:8103/docs
- 健康检查:http://127.0.0.1:8103/health
- 前端页面:http://127.0.0.1:8103/login.html

---

## 环境变量配置

`.env` 中包含所有服务的连接信息与密钥。以下为所需配置项(**请勿将真实密钥提交到版本库**):

```ini
# ---- DeepSeek LLM ----
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# ---- 智谱 Embedding(2048 维,需与旧向量库一致) ----
ZHIPU_API_KEY=your_zhipu_api_key

# ---- Milvus(本地文件模式) ----
MILVUS_ENDPOINT=E:/AI/35agent/pdf_agent.db

# ---- Redis ----
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# ---- MySQL(库名 aidoctor) ----
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=aidoctor

# ---- JWT ----
JWT_SECRET_KEY=your_jwt_secret
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440

# ---- Neo4j ----
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# ---- 阿里云 OSS ----
OSS_ACCESS_KEY_ID=your_oss_key_id
OSS_ACCESS_KEY_SECRET=your_oss_key_secret
OSS_BUCKET_NAME=your_bucket
OSS_ENDPOINT=your_oss_endpoint

# ---- 豆包语音(ASR/TTS) ----
DOUBAO_APP_ID=your_app_id
DOUBAO_ACCESS_TOKEN=your_access_token
```

> 注意:Embedding 模型与维度必须与已有向量库(`pdf_agent.db`)保持一致,否则检索结果会失真。

---

## API 接口

### 问答

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/` | 非流式问答(主接口,压测调用) |
| `POST` | `/chat_stream` | 流式问答(SSE) |

`POST /` 请求示例:

```json
{ "question": "高血压怎么治疗?", "username": "user_001" }
```

响应示例:

```json
{
  "response": "LLM 生成的答案",
  "status": 200,
  "time": "2026-06-26 10:00:00",
  "session_id": "uuid",
  "image_urls": [
    { "rank": 0, "image_url": "https://...", "pdf_url": "https://...", "text_preview": "..." }
  ]
}
```

### 运维 / 监控

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 服务健康检查 |
| `GET` | `/metrics` | 运行时监控指标 |
| `POST` | `/graph_query` | 通用 Cypher 查询(只读) |
| `GET` | `/get_oss_sign` | 获取 OSS 上传签名 |
| `POST` | `/tts` | 文本转语音 |

### 语音(前缀 `/voice`)

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/voice/asr` | 音频文件 → 流式 ASR 识别文本 |
| `POST` | `/voice/chat_tts` | 文本 → RAG + LLM + 流式 TTS |
| `POST` | `/voice/full_pipeline` | 音频 → ASR → RAG+LLM → TTS 全链路 |
| `POST` | `/voice/tts_only` | 纯文本 → 流式 TTS(不经 LLM) |

### 认证(前缀 `/auth`)

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/auth/register` | 用户注册 |
| `POST` | `/auth/login` | 用户登录(返回 JWT) |
| `GET` | `/auth/me` | 获取当前登录用户 |

### 会话

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/get_session_list` | 获取会话列表 |
| `POST` | `/get_session_messages` | 获取会话历史消息 |

### 数据飞轮

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/feedback` | 提交用户反馈(👍 / 👎) |
| `GET` | `/feedback/stats` | 查看反馈统计数据 |
| `POST` | `/expert/submit` | 专家提交标准答案 |
| `GET` | `/expert/queue` | 查看专家审核队列 |

---

## 核心流程说明

### 1. 意图路由(RouterAgent)

纯规则引擎,零 LLM 调用,耗时 < 1ms。路由优先级:

```
web_search > medical_rag > chitchat
```

垃圾输入 / 问候语 / 身份询问走快车道,直接返回静态回复。

### 2. 混合检索(RAG)

- 稠密向量:Milvus(HNSW / IVF_FLAT 索引)
- 稀疏检索:BM25
- 融合排序:RRF(Reciprocal Rank Fusion)
- 精排:Rerank
- 双库切换:根据意图选择「现代医学库」或「中医古籍库」

### 3. 缓存防护(Redis)

- **防击穿**:分布式锁 + 双重检查(Double Check)
- **防穿透**:空结果写入 `<EMPTY>` 占位符,短 TTL
- **防雪崩**:TTL 增加 ±10% 随机抖动

### 4. ReAct 与反思

- `BaseAgent` 实现 ReAct 循环(Reasoning + Acting),通过 prompt 注入 JSON 描述工具定义(非原生 function calling)。
- `ReflectionAgent` 对生成结果做四维度加权评估:
  `准确性 × 0.35 + 完整性 × 0.25 + 安全性 × 0.25 + 相关性 × 0.15`,低于阈值 `0.65` 触发修正。
- 为保障首 Token 时延(TTFT),快路径默认关闭反思(`reflect_enabled=False`)。

### 5. 三层记忆

| 层级 | 存储 | 说明 |
| --- | --- | --- |
| 短期记忆 | Redis List `chat:history:{session_id}` | Redis 不可用时降级为本地 LRU |
| 长期记忆 | Redis Hash `memory:user:{user_id}` | 用户偏好等长期信息 |
| 语义记忆 | Milvus | 跨会话语义检索 |

TTL 默认 7 天。

### 6. 数据飞轮

每次请求生成唯一 `session_id`,前端据此提交 👍/👎 反馈。反馈与交互记录写入 MySQL 并做 JSONL 备份(如 [feedback_logs/thumbs_up.jsonl](file:///e:/AI/Agent_online/feedback_logs/thumbs_up.jsonl)),经专家审核后回流用于优化。

---

## 数据初始化

| 脚本 | 作用 |
| --- | --- |
| [aidoctor.sql](file:///e:/AI/Agent_online/aidoctor.sql) | 初始化 MySQL 库表(用户、会话、消息、反馈等) |
| [build_medical_milvus.py](file:///e:/AI/Agent_online/build_medical_milvus.py) | 构建现代医学向量库 |
| [build_ancient_books.py](file:///e:/AI/Agent_online/build_ancient_books.py) | 构建中医古籍向量库 |

---

## 测试

```bash
# 冒烟测试
python smoke_test_suite/smoke_test.py

# 回归测试
python regression_test/run_regression.py

# 压测(Locust)
python stress_test/run_stress_test.py
# 或直接使用 Locust:
locust -f stress_test/locustfile_basic.py        # 基础场景
locust -f stress_test/locustfile_peak.py         # 峰值场景
```

---

## 日志与监控

- 业务日志:写入根目录 [app.log](file:///e:/AI/Agent_online/app.log),同时输出到控制台
- 审计日志:由 `AuditMiddleware` 记录所有请求(跳过 `/health`、`/metrics`、`/docs`、`/openapi.json`)
- 降级事件:由 `FallbackManager` 记录,可通过 `/metrics` 查看运行时指标
- 反馈数据:MySQL + [feedback_logs/](file:///e:/AI/Agent_online/feedback_logs) JSONL 备份

---

## 注意事项

1. **向量空间一致性**:更换 Embedding 模型或维度前,必须重建向量库,否则检索结果不可用。
2. **密钥安全**:`.env` 含真实凭证,请加入 `.gitignore`,严禁提交或外泄。
3. **CORS**:默认 `allow_origins=["*"]`,生产环境请收敛为白名单。
4. **生产部署**:`uvicorn.run(...)` 默认 `workers=1`,高并发场景建议按需调整并置于反向代理之后。
5. **服务依赖**:启动时会初始化 Milvus(双集合)、检查 Redis、初始化 Neo4j 并预热图结构,任一依赖异常会影响对应能力。
