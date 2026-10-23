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
    evaluation_ok: bool = True      # 评估是否成功 (解析失败/异常时为 False, 此时 score=-1.0 无效)


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
            parsed = self._parse_evaluation(result_text)
            duration = round((time.time() - t0) * 1000, 2)
            if parsed is None:
                # 解析失败: 不伪造高分, 标记评估无效, 由编排器按保守策略处理
                logger.warning("[ReflectionAgent] 评估 JSON 解析失败")
                return self._invalid_result("评估结果 JSON 解析失败", duration)

            # 校验四个维度分数是否存在且为 0~1 之间的合法数值
            for field_name in ("accuracy", "completeness", "safety", "relevance"):
                if not self._is_valid_score(parsed.get(field_name)):
                    logger.warning(
                        f"[ReflectionAgent] 评估字段 {field_name} 缺失或非法: "
                        f"{parsed.get(field_name)!r}"
                    )
                    return self._invalid_result(
                        f"评估字段 {field_name} 缺失或非法", duration
                    )

            # 计算加权总分 (非简单平均)
            # 权重设计理由:
            #   accuracy (准确性)*0.35:     医学场景中准确性最为关键, 错误信息可能危及生命
            #   safety (安全性)*0.25:       安全性次之, 回答不能包含有害建议或误导性用药指导
            #   completeness (完整性)*0.25: 确保回答全面, 不遗漏重要注意事项或禁忌
            #   relevance (相关性)*0.15:    权重最低, 因为即使稍有偏题, 只要准确安全仍可接受
            accuracy = float(parsed["accuracy"])
            completeness = float(parsed["completeness"])
            safety = float(parsed["safety"])
            relevance = float(parsed["relevance"])
            score = (
                accuracy * 0.35 +
                completeness * 0.25 +
                safety * 0.25 +
                relevance * 0.15
            )
            # 构建评估结果, needs_correction 由加权总分与阈值比较决定
            result = ReflectionResult(
                score=round(score, 3),
                accuracy=accuracy,
                completeness=completeness,
                safety=safety,
                relevance=relevance,
                feedback=str(parsed.get("feedback", "") or ""),
                needs_correction=score < self.threshold,
                duration_ms=duration,
                evaluation_ok=True,
            )
            logger.info(
                f"[ReflectionAgent] 评估完成 | score={result.score:.3f} | "
                f"needs_correction={result.needs_correction} | 耗时={duration}ms"
            )
            return result
        except Exception as e:
            # 评估失败: 不伪造高分, 标记评估无效, 由编排器按保守策略处理
            logger.warning(f"[ReflectionAgent] 评估失败: {e}")
            duration = round((time.time() - t0) * 1000, 2)
            return self._invalid_result(f"评估异常: {e}", duration)

    # 
    async def correct(
        self,
        query: str,
        original_answer: str,
        reflection: ReflectionResult,
        context: str = "",
    ) -> Tuple[str, bool]:
        """
        基于反思结果修正回答.
        :return: (修正后的回答, 是否成功)
          - 成功: 返回非空的修正文本, 第二项为 True
          - 失败 (LLM 异常或返回空): 回退原始回答, 第二项为 False (不伪装成功)
        """
        t0 = time.time()
        # 构造修正 Prompt — 结合原始问题/回答/反馈/证据, 并施加医学安全约束
        correct_prompt = (
            "你是一个医学专家。之前的回答经质量评估后不达标，请基于以下材料改进回答。\n\n"
            f"问题: {query}\n\n"
            f"原始回答: {original_answer[:600]}\n\n"
            f"质量评估反馈: {reflection.feedback}\n"
            f"各维度评分: 准确性={reflection.accuracy:.2f}, "
            f"完整性={reflection.completeness:.2f}, "
            f"安全性={reflection.safety:.2f}, "
            f"相关性={reflection.relevance:.2f}\n\n"
            "修正要求:\n"
            "1. 结合原始问题、原始回答、评估反馈和参考资料修正回答;\n"
            "2. 不得编造参考资料中没有支持的医学事实;\n"
            "3. 不得为了让回答更完整而擅自增加药物剂量、疗程或个体化治疗建议;\n"
            "4. 当参考资料不足时, 明确说明无法从现有证据可靠确认;\n"
            "5. 保留必要的医学安全提示与来源引用, 不得无依据地删除原回答中的有效引用;\n"
            "6. 只输出修正后的回答正文, 不要输出任何解释或思考过程。\n\n"
        )
        if context:
            correct_prompt += f"参考资料: {context[:800]}\n\n"
        correct_prompt += "请输出修正后的回答:"
        try:
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": correct_prompt}],
                stream=False,
            )
            corrected = (response.choices[0].message.content or "").strip()
            duration = round((time.time() - t0) * 1000, 2)
            if not corrected:
                # 修正失败: 返回空则回退原始回答, 不返回空答案, 不伪装成功
                logger.warning(
                    f"[ReflectionAgent] 修正返回空答案 | 耗时={duration}ms | 回退原始回答"
                )
                return original_answer, False
            logger.info(
                f"[ReflectionAgent] 修正完成 | 耗时={duration}ms | "
                f"原长度={len(original_answer)} → 新长度={len(corrected)}"
            )
            return corrected, True
        except Exception as e:
            # 修正失败: 回退原始回答, 不抛出异常, 不伪装成功
            logger.warning(f"[ReflectionAgent] 修正失败: {e}")
            return original_answer, False

    def _parse_evaluation(self, text: str) -> Optional[Dict[str, Any]]:
        """解析 LLM 返回的评估 JSON; 失败时返回 None (不伪造默认高分)."""
        try:
            # JSON 提取策略: 从第一个 '{' 到最后一个 '}' 截取
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    @staticmethod
    def _is_valid_score(value: Any) -> bool:
        """校验评分是否为 0~1 之间的合法数值 (排除 bool/None/字符串)."""
        if isinstance(value, bool) or value is None:
            return False
        if isinstance(value, (int, float)):
            return 0.0 <= float(value) <= 1.0
        return False

    @staticmethod
    def _invalid_result(reason: str, duration_ms: float) -> ReflectionResult:
        """构造评估无效结果: score=-1.0 表示无效, 不触发修正, 不伪装高分."""
        return ReflectionResult(
            score=-1.0,
            accuracy=0.0,
            completeness=0.0,
            safety=0.0,
            relevance=0.0,
            feedback=reason,
            needs_correction=False,
            duration_ms=duration_ms,
            evaluation_ok=False,
        )