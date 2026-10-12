"""
Agent_online 冒烟测试 - 配置文件

======================================
冒烟测试(Smoke Test)目标: 用最短时间验证系统核心功能是否"能用",
不追求覆盖率和性能,   只确认关键路径通畅   (服务启动后的第一道关卡)
"""

# ==================== 目标服务配置 ====================
TARGET_HOST = "http://127.0.0.1:8103"

# API Key (如果 AuthMiddleware 开启了认证, 当前并未开启)
API_KEY = "XXXXXX"


# ==================== 超时配置 ====================
TIMEOUTS = {
    "health": 5,   # 健康检查   (应该极快)
    "non_stream": 60, # 非流式问答 (含LLM推理, 线上实测需要较长时间)
    "stream": 90,  # 流式问答   (含RAG + LLM完整链路)
    "voice": 30,   # 语音接口
    "feedback": 10, # 反馈接口
    "session": 10, # 会话管理
}


# ==================== 冒烟测试用例数据 ====================
# 每个接口只用  1~2 个代表性输入, 快速验证
SMOKE_QUESTION = "高血压怎么治疗?"
SMOKE_TCM_QUESTION = "黄帝内经讲了什么养生之道?"
SMOKE_GREETING = "你好"
SMOKE_TTS_TEXT = "这是一条冒烟测试的语音合成文本。"

# 冒烟测试用户

SMOKE_USER = "smoke_test_user"
SMOKE_EXPERT = "smoke_doctor_001"

# ==================== 核心接口清单 ====================
# 冒烟测试必须验证的接口 (P0级别, 任一失败即阻断上线)
P0_ENDPOINTS = [
    ("GET", "/health", "健康检查"),
    ("POST", "/", "非流式问答"),
    ("POST", "/chat_stream", "流式问答"), ]


# 重要接口 (P1级别, 失败需评估)
P1_ENDPOINTS = [
    ("GET", "/metrics", "系统指标"),
    ("POST", "/feedback", "用户反馈"),
    ("GET", "/feedback/stats", "反馈统计"),
    ("GET", "/expert/queue", "专家队列"), ]


# 可选接口 (P2级别, 语音等增强功能)
P2_ENDPOINTS = [
    ("POST", "/voice/tts_only", "语音合成"),
    ("POST", "/voice/asr", "语音识别"), ]

# ==================== 判定阈值 ====================
# 冒烟测试关注"能不能用",  只做基础可用性判定
SMOKE_CRITERIA = {

    "health_max_ms": 2000, # 健康检查响应上限
    "non_stream_max_ms": 15000, # 非流式响应上限 (宽松, 含冷启动)
    "min_answer_length": 5, # 回答最小长度 (避免空响应)
}

# ==================== 报告配置 ====================
REPORT_DIR = "smoke_test_suite/reports"
