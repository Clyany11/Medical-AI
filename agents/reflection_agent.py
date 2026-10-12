"""
agents/reflection_agent.py — 反思 Agent
职责: 对生成的回答进行质量评估和修正
  - 评估维度: 准确性, 完整性, 安全性, 相关性
  - 低分时触发修正, 重新生成更高质量的回答
  - 异步执行, 不阻塞主流程 (流式模式下后台运行)
设计要点:
  - 轻量级: 使用 mini 模型做评估, 延迟可控
  - 可选: 通过配置决定是否启用
  - 结果写入 memory, 供后续请求参考
"""
import logging
import time
import json
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger("AgentApp")


# 反思评估结果数据类
# score 是加权平均分 (非简单算术平均),
# 权重: 准确性 * 0.35 + 完整性 * 0.25 + 安全性 * 0.25 + 相关性 * 0.15
# needs_correction 为 True 时, 会触发下游 correct() 方法对回答进行修正重写
@dataclass
class ReflectionResult:
    """反思结果"""
    score: float                    # 总分 (0.0 ~ 1.0), 加权平均, 非简单平均
    accuracy: float = 0.0           # 准确性
    completeness: float = 0.0       # 完整性
    safety: float = 0.0             # 安全性
    relevance: float = 0.0          # 相关性
    feedback: str = ""              # 反思反馈
    needs_correction: bool = False  # 当 score < threshold 时为 True, 触发下游修正流程
    duration_ms: float = 0.0


class ReflectionAgent:
    """
    反思 Agent — 答案质量评估与修正

    两种模式:
      1. 快速反思: 仅打分, 不修正 (用于流式后台评估)
      2. 深度反思: 打分 + 修正 (用于非流式高质量场景)
    """

    def __init__(
        self,
        llm_client: Any,
        model: str = "deepseek-chat",
        # threshold=0.65 表示加权总分低于 65% 的回答将被标记为需要修正
        # 65% 是经过调优的阈值: 太高会导致大量不必要的修正 (浪费延迟),
        # 太低则会放过低质量回答 (影响用户体验)
        threshold: float = 0.65,
    ):
        self.llm_client = llm_client
        # 使用 mini 模型做评估, 平衡质量与延迟 (评估不需要最强模型)
        self.model = model
        # 质量阈值: 加权总分低于此值的回答会触发 needs_correction=True
        self.threshold = threshold

    # ==================== 核心方法: 质量评估 ====================
    # evaluate() 是 ReflectionAgent 的主入口, 由 AgentOrchestrator 调用
    # 流程: 构造评估 Prompt → 调用 LLM → 解析 JSON 评分 → 计算加权总分 → 判定是否需修正
    async def evaluate(
        self, query: str, answer: str, context: str = ""
    ) -> ReflectionResult:
        """
        评估回答质量
        :param query: 原始问题
        :param answer: 生成的回答
        :param context: 检索上下文 (可选)
        :return: ReflectionResult
        """
        t0 = time.time()
        # 构造评估 Prompt
        # answer 截取前 800 字符: 节省 token 开销, 同时保留回答的核心内容供评估
        # (医学回答通常在前 800 字已覆盖关键诊断信息和核心结论)
        eval_prompt = (
            "你是一个医学回答质量评估专家, 请评估以下回答的质量。\n\n"
            f"问题: {query}\n\n"
            f"回答: {answer[:800]}\n\n"
        )
        if context:
            # context 截取前 500 字符: 参考资料只需提供足够的对照信息即可,
            # 无需完整内容, 大幅减少 token 消耗
            eval_prompt += f"参考资料: {context[:500]}\n\n"
        eval_prompt += (
            "请按以下 JSON 格式输出评估结果 (只输出 JSON, 不要其他内容):\n"
            '{"accuracy": 0.8, "completeness": 0.7, "safety": 0.9, "relevance": 0.8, "feedback": "简短反馈"}\n'
            "每个维度分数范围 0.0-1.0。"
        )
        try:
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": eval_prompt}],
                stream=False,
            )
            # 提取 LLM 返回的文本内容 (Chat Completions: choices[0].message.content)
            result_text = response.choices[0].message.content or ""
            # 解析 JSON
            parsed = self._parse_evaluation(result_text)
            duration = round((time.time() - t0) * 1000, 2)
            # 计算加权总分 (非简单平均)
            # 权重设计理由:
            #   accuracy (准确性)*0.35:     医学场景中准确性最为关键, 错误信息可能危及生命
            #   safety (安全性)*0.25:       安全性次之, 回答不能包含有害建议或误导性用药指导
            #   completeness (完整性)*0.25: 确保回答全面, 不遗漏重要注意事项或禁忌
            #   relevance (相关性)*0.15:    权重最低, 因为即使稍有偏题, 只要准确安全仍可接受
            score = (
                parsed.get("accuracy", 0.8) * 0.35 +
                parsed.get("completeness", 0.7) * 0.25 +
                parsed.get("safety", 0.9) * 0.25 +
                parsed.get("relevance", 0.8) * 0.15
            )
            # 构建评估结果, needs_correction 由加权总分与阈值比较决定
            result = ReflectionResult(
                score=round(score, 3),
                accuracy=parsed.get("accuracy", 0.8),
                completeness=parsed.get("completeness", 0.7),
                safety=parsed.get("safety", 0.9),
                relevance=parsed.get("relevance", 0.8),
                feedback=parsed.get("feedback", ""),
                needs_correction=score < self.threshold,
                duration_ms=duration,
            )
            logger.info(
                f"[ReflectionAgent] 评估完成 | score={result.score:.3f} | "
                f"needs_correction={result.needs_correction} | 耗时={duration}ms"
            )
            return result
        except Exception as e:
            # 评估失败时的容错策略: 给予 "疑罪从无" 的默认高分 (0.8)
            # needs_correction=False 确保评估失败不会误触发修正流程
            # 理由: 评估本身是辅助增强, 失败时不应影响主流程正常出答
            logger.warning(f"[ReflectionAgent] 评估失败: {e}")
            duration = round((time.time() - t0) * 1000, 2)
            return ReflectionResult(
                score=0.8, accuracy=0.8, completeness=0.8,
                safety=0.9, relevance=0.8,
                feedback="评估失败, 使用默认分数",
                needs_correction=False,
                duration_ms=duration,
            )

    # 
    async def correct(
        self,
        query: str,
        original_answer: str,
        reflection: ReflectionResult,
        context: str = "",
    ) -> str:
        """
        基于反思结果修正回答
        :param query: 原始问题
        :param original_answer: 原始回答
        :param reflection: 反思结果
        :param context: 检索上下文
        :return: 修正后的回答
        """
        t0 = time.time()
        # 构造修正 Prompt
        # original_answer 截取前 600 字符: 比 evaluate 中的 800 更短,
        # 因为修正时只需要让 LLM 了解原始回答的大致方向和问题所在,
        # 无需完整内容, 省 token 的同时保留足够上下文供修正参考
        correct_prompt = (
            "你是一个医学专家, 之前的回答质量不够好, 请改进。\n\n"
            f"问题: {query}\n\n"
            f"原始回答: {original_answer[:600]}\n\n"
            f"质量评估反馈: {reflection.feedback}\n"
            f"评分: 准确性={reflection.accuracy:.1f}, "
            f"完整性={reflection.completeness:.1f}, "
            f"安全性={reflection.safety:.1f}\n\n"
        )
        if context:
            correct_prompt += f"参考资料: {context[:800]}\n\n"
        correct_prompt += "请生成一个更准确, 更完整, 更安全的回答:"
        try:
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": correct_prompt}],
                stream=False,
            )
            corrected = response.choices[0].message.content or ""
            duration = round((time.time() - t0) * 1000, 2)
            logger.info(
                f"[ReflectionAgent] 修正完成 | 耗时={duration}ms | "
                f"原长度={len(original_answer)} → 新长度={len(corrected)}"
            )
            # 返回修正后的回答, 如果 LLM 返回空内容则回退到原始回答
            # 核心原则: 永远不返回空字符串, 修正失败不应该让用户看不到任何回答
            return corrected if corrected else original_answer
        except Exception as e:
            # 优雅降级: 修正失败时返回原始回答, 而非抛出异常
            # 修正是锦上添花的增强环节, 失败不应丢失已经生成的原始回答
            logger.warning(f"[ReflectionAgent] 修正失败: {e}")
            return original_answer

    def _parse_evaluation(self, text: str) -> Dict[str, Any]:
        """解析 LLM 返回的评估 JSON"""
        try:
            # JSON 提取策略: 从第一个 '{' 到最后一个 '}' 截取
            # LLM 可能在 JSON 前后添加解释性文字 (如 "以下是评估结果:"),
            # 通过 find/rfind 定位 JSON 边界, 跳过这些多余文本
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
        # 解析失败时返回乐观的默认值 (偏高分)
        # 刻意设置偏高的默认值 (0.8/0.7/0.9/0.8) 以避免误触发修正:
        # 如果无法解析评估结果, 宁可放过一个低质量回答, 也不要因解析错误
        # 而无谓地触发修正流程 (修正本身也有延迟和出错风险)
        return {
            "accuracy": 0.8,
            "completeness": 0.7,
            "safety": 0.9,
            "relevance": 0.8,
            "feedback": "JSON解析失败",
        }