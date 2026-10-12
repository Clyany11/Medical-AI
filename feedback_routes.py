"""
feedback_routes.py — 数据飞轮反馈 API 路由
提供以下接口:
  POST /feedback              — 用户提交点赞/点踩反馈
  GET  /feedback/stats        — 查看反馈统计数据 (运营复盘)
  POST /expert/submit         — 专家提交标准答案 (数据飞轮闭环)
  GET  /expert/queue          — 查看专家审核队列
与 app.py 的接口约定:
  from feedback_routes import router as feedback_router
  app.include_router(feedback_router)
"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from feedback_logger import (
    log_thumbs_up,
    log_thumbs_down,
    submit_expert_answer,
    get_feedback_stats,
    get_expert_queue,
)
logger = logging.getLogger("FeedbackRoutes")
# app.py 通过 `from feedback_routes import router as feedback_router` 导入
router = APIRouter(tags=["数据飞轮"])
# ===================== Pydantic 请求/响应模型 =====================
class FeedbackRequest(BaseModel):
    """用户提交点赞/点踩反馈的请求体"""
    session_id: str = Field(..., description="由 POST / 接口返回的会话 ID")
    feedback: str   = Field(..., description="反馈类型: thumbs_up 或 thumbs_down")
    user_id: Optional[str] = Field("anonymous", description="用户标识")
    # 点踩扩展字段 (点赞时不传)
    feedbackType: Optional[str] = Field(None, description="点踩原因枚举值: harmful / false_info / not_helpful / other")
    content: Optional[str] = Field(None, description="用户输入的反馈意见文本")
    answer: Optional[str] = Field(None, description="被点踩的 AI 回复内容")
class ExpertSubmitRequest(BaseModel):
    """专家提交标准答案的请求体"""
    task_id: str       = Field(..., description="专家审核任务 ID (点踩时自动生成)")
    expert_id: str     = Field(..., description="专家标识, 如 doctor_001")
    expert_answer: str = Field(..., description="专家提供的标准医学答案")
    notes: Optional[str] = Field(None, description="备注, 如参考的临床指南, 文献来源等")
# ===================== 路由定义 =====================
@router.post("/feedback", summary="提交用户反馈(点赞/点踩)")
async def submit_feedback(req: FeedbackRequest):
    """
    接收前端传回的点赞或点踩信号, 驱动数据飞轮
    - 点赞: 将本次交互写入正样本库 (thumbs_up.jsonl)
    - 点踩: 写入高价值负样本库 (thumbs_down.jsonl) + 创建专家审核任务
    前端调用示例:
        POST /feedback
        {"session_id": "abc-123", "feedback": "thumbs_down", "user_id": "user_001"}
    点踩返回的 expert_task_id 可用于后续查询专家审核进度
    """
    if req.feedback == "thumbs_up":
        logger.info(f"session_id: {req.session_id} | user_id: {req.user_id}")
        success = await log_thumbs_up(req.session_id, req.user_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"未找到 session_id={req.session_id}，请确认会话 ID 正确"
            )
        return {
            "success": True,
            "feedback": "thumbs_up",
            "message": "感谢您的正面反馈! 您的评价帮助我们持续改进。",
        }
    elif req.feedback == "thumbs_down":
        # 校验点踩原因枚举值
        valid_reasons = ["harmful", "false_info", "not_helpful", "other"]
        if req.feedbackType and req.feedbackType not in valid_reasons:
            raise HTTPException(
                status_code=400,
                detail=f"feedbackType 字段必须为以下之一：{valid_reasons}"
            )
        result = await log_thumbs_down(req.session_id, req.user_id, req.feedbackType, req.content, req.answer)
        if not result["success"]:
            raise HTTPException(
                status_code=404,
                detail=f"未找到 session_id={req.session_id}, 请确认会话 ID 正确"
            )
        return {
            "success": True,
            "feedback": "thumbs_down",
            "message": "感谢您的反馈! 我们已将此问题提交给医疗专家审核, 将尽快优化答案质量。",
            "expert_task_id": result["expert_task_id"],
        }
    else:
        raise HTTPException(
            status_code=400,
            detail="feedback 字段必须为 'thumbs_up' 或 'thumbs_down'"
        )
@router.get("/feedback/stats", summary="查看反馈统计数据")
async def feedback_stats():
    """
    返回数据飞轮的运营统计数据, 用于复盘和监控
    返回字段:
      - total_interactions:    总交互次数
      - thumbs_up:             点赞次数 (正样本数量)
      - thumbs_down:           点踩次数 (负样本数量)
      - expert_queue_pending:  待专家审核的任务数
    """
    return await get_feedback_stats()
@router.post("/expert/submit", summary="专家提交标准答案")
async def expert_submit(req: ExpertSubmitRequest):
    """
    专家审核点踩数据并提交标准答案, 完成数据飞轮闭环
    专家工作流:
      1. GET /expert/queue?status=pending  — 查看待审核任务
      2. 审核 bad_generation (Agent 的错误回复) 和 query (用户问题)
      3. POST /expert/submit               — 提交标准医学答案
      4. 标注数据用于模型微调和知识库优化
    提交后任务状态变更为 completed，expert_answer 回写到审核记录
    """
    success = await submit_expert_answer(
        task_id=req.task_id,
        expert_id=req.expert_id,
        expert_answer=req.expert_answer,
        notes=req.notes,
    )
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"未找到 task_id={req.task_id}, 请确认任务 ID 正确"
        )
    return {
        "success": True,
        "message": "专家答案已提交, 感谢您的专业贡献!",
        "task_id": req.task_id,
        "expert_id": req.expert_id,
    }
@router.get("/expert/queue", summary="查看专家审核队列")
async def expert_queue(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    查看专家审核队列, 支持按状态过滤和分页
    :param status: "pending"(待审核) | "completed"(已完成) | 不传(全部)
    :param limit:  每页数量 (默认50)
    :param offset: 分页偏移 (默认0)
    返回字段:
      - tasks:  任务列表 (含 query, bad_generation, milvus_trace 等)
      - total:  符合条件的总任务数
    """
    return await get_expert_queue(status=status, limit=limit, offset=offset)