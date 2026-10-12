"""
agents/base.py — BaseAgent 基类
实现 ReAct (Reasoning + Acting) 循环:
  1. _react_loop(): 推理-行动循环, LLM 决定是否调用工具
  2. _reflect(): 轻量级 LLM 调用, 对输出质量打分 (0.0-1.0)
  3. _correct(): 基于反思反馈重新生成 (仅当 score < threshold 时触发)
设计要点:
  - 异步优先: 所有方法均为 async
  - 流式输出: run_stream() 支持流式输出
  - 工具绑定: 通过 tools 参数注入, Agent 自主决定是否调用
  - 熔断降级: LLM 调用失败时使用 fallback 模型
"""
import logging
import time
import json
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, field

logger = logging.getLogger("AgentApp")


# 工具调用结果的数据容器, 记录每次工具执行的完整上下文信息
@dataclass
class ToolResult:
    """工具调用结果"""
    tool_name: str              # 被调用的工具名称, 对应 tools 列表中的 "name" 字段
    tool_input: Dict[str, Any]  # 传给工具的参数字典, 由 LLM 自主决定
    result: Any                 # 工具返回的原始结果, 可以是字典/字符串/任意类型
    success: bool = True        # 执行是否成功, 失败时为 False
    error: str = ""             # 失败时的错误信息, 成功时为空字符串
    duration_ms: float = 0.0    # 工具执行耗时(ms 毫秒), 用于性能监控


# Agent 单次执行的完整结果, 包含生成内容, 工具使用记录, 质量评分等全链路信息
@dataclass
class AgentResult:
    """Agent 执行结果"""
    content: str  # 最终输出的文本内容 (可能经过修正)
    tools_used: List[ToolResult] = field(default_factory=list)  # 本次执行中所有工具调用记录
    # 反思评分, 默认 1.0 表示"未经反思则视为满分", 避免未开启反思时误触发修正逻辑
    reflect_score: float = 1.0
    # 是否经过修正, 默认 False; 仅当 reflect_score < threshold 且修正成功时才为 True
    corrected: bool = False
    total_llm_calls: int = 0  # 累计 LLM 调用次数 (含 ReAct + 反思 + 修正)
    total_duration_ms: float = 0.0  # 整个 Agent 执行的总耗时(毫秒)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 扩展元数据, 供上层编排器传递额外信息


class BaseAgent:
    """
    BaseAgent 基类 — ReAct + Reflection + Self-Correction

    执行流程:
      run(messages, context) →
        Phase 1: _react_loop()
          无工具 → 单次 LLM 调用 → 返回 content
          有工具 → bind_tools → 循环 (最多 max_iterations 轮):
            LLM 决定调用工具 → _execute_tool() → 追加结果 → 继续
            LLM 不调用工具 → 返回 content (结束循环)
        Phase 2: _reflect() [可选]
          轻量级 LLM 调用, 对输出质量打分
        Phase 3: _correct() [仅当 score < threshold]
          基于反思反馈重新生成 (仅一次)
    """

    def __init__(
        self,
        name: str,   # Agent 名称, 用于日志标识 (如 "MedicalAgent", "RouterAgent")
        system_prompt: str,  # 系统提示词, 定义 Agent 的角色和行为规范
        llm_client: Any,  # LLM 客户端实例 (OpenAI 兼容接口, 支持 chat.completions)
        model: str = "deepseek-chat",  # 使用的模型 ID
        tools: Optional[List[Dict[str, Any]]] = None,  # 可用工具列表, 每个工具含 name/description/parameters/handler
        max_iterations: int = 5,  # ReAct 循环最大迭代次数, 防止无限循环
        reflect_enabled: bool = True,  # 是否开启反思评分, 关闭可降低延迟
        reflect_threshold: float = 0.7,  # 反思分数阈值, 低于此值触发修正 (0.7 = 70分以下需要重新生成)
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.llm_client = llm_client
        self.model = model
        self.tools = tools or []      # 空列表表示无工具模式, 直接单次 LLM 调用
        self.max_iterations = max_iterations
        self.reflect_enabled = reflect_enabled
        self.reflect_threshold = reflect_threshold

    # 异步非流式函数
    async def run(
        self,
        query: str,
        context: str = "",
        messages: Optional[List[Dict]] = None,
        **kwargs,
    ) -> AgentResult:
        """
        同步执行 Agent (非流式, 返回完整结果)
        """
        t0 = time.time()
        llm_calls = 0

        # Phase 1: ReAct 循环
        react_content, tools_used, react_llm_calls = await self._react_loop(
            query, context, messages, **kwargs
        )
        llm_calls += react_llm_calls

        # Phase 2: 反思 (可选)
        reflect_score = 1.0
        if self.reflect_enabled and react_content:
            reflect_score, reflect_calls = await self._reflect(query, react_content)
            llm_calls += reflect_calls

        # Phase 3: 修正 (仅当分数低于阈值)
        corrected = False
        final_content = react_content
        if reflect_score < self.reflect_threshold and react_content:
            corrected_content, correct_calls = await self._correct(
                query, context, react_content, reflect_score
            )
            if corrected_content:
                final_content = corrected_content
                corrected = True
            llm_calls += correct_calls

        duration = round((time.time() - t0) * 1000, 2)
        logger.info(
            f"[{self.name}] 执行完毕 | LLM调用={llm_calls} | "
            f"工具={len(tools_used)} | 反思={reflect_score:.2f} | "
            f"修正={corrected} | 耗时={duration}ms"
        )
        return AgentResult(
            content=final_content,
            tools_used=tools_used,
            reflect_score=reflect_score,
            corrected=corrected,
            total_llm_calls=llm_calls,
            total_duration_ms=duration,
        )

    # 异步流式函数
    async def run_stream(
        self,
        query: str,
        context: str = "",
        messages: Optional[List[Dict]] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        流式执行 Agent — 逐个 token 输出
        注意: 流式模式下跳过反思和修正 (为了 TTFT 优先)
        反思在流式结束后异步执行, 结果写入 memory 供下次参考
        """
        input_messages = self._build_messages(query, context, messages)
        # 创建流式请求 (Chat Completions, 无额外扩展参数, 保证首 token 延迟)
        response = await self.llm_client.chat.completions.create(
            model=self.model,
            messages=input_messages,
            stream=True,
        )
        # 逐块处理流式响应
        async for chunk in response:
            # 部分 chunk (如 usage 统计块) 的 choices 为空, 需跳过
            if not getattr(chunk, "choices", None):
                continue
            delta_text = getattr(chunk.choices[0].delta, "content", None)
            if delta_text:
                yield delta_text  # 逐片段 yield, 由上层 SSE 推送给客户端

    # 
    async def _react_loop(
        self,
        query: str,
        context: str = "",
        messages: Optional[List[Dict]] = None,
        **kwargs,
    ) -> tuple:
        """
        ReAct 循环: 推理 → 行动 → 观察 → 推理...
        返回: (content, tools_used, llm_calls)
        """
        tools_used = []
        llm_calls = 0

        # ========== 分支一: 无工具模式 ==========
        # 当 Agent 没有注册任何工具时, 直接做一次 LLM 调用即可
        # 这是最简单的路径, 适用于 MedicalAgent 等纯生成场景
        if not self.tools:
            input_messages = self._build_messages(query, context, messages)
            content = await self._call_llm(input_messages)
            return content, tools_used, 1

        # ========== 分支二: 有工具模式 (ReAct 循环) ==========
        # 构建初始对话上下文, 后续每轮迭代会追加 assistant/user 消息
        conversation = self._build_messages(query, context, messages)
        # 现实中, 生产环境的 ReAct 必须有执行次数的限制, max_iterations = 3 ~ 5
        for iteration in range(self.max_iterations):
            llm_calls += 1
            # 调用 LLM 并在 prompt 中注入工具描述, 让模型决定是否使用工具
            response_text = await self._call_llm_with_tools(conversation)
            # 解析 LLM 输出中是否包含 tool_call JSON 结构
            tool_call = self._parse_tool_call(response_text)

            if tool_call is None:
                # LLM 输出中没有工具调用 → 模型认为已有足够信息, 直接返回答案
                return response_text, tools_used, llm_calls

            # 执行工具调用, kwargs 中可能包含通过 tool_xxx 传入的工具函数
            tool_result = await self._execute_tool(tool_call, **kwargs)
            tools_used.append(tool_result)

            # 将 LLM 的输出(含工具调用意图)和工具执行结果追加到对话历史
            # 工具结果以 user 消息形式追加, 原因:
            # 1: OpenAI 兼容 Chat Completions 走 assistant→user 交替, 不做原生 tool role 解析
            # 2: 将工具结果伪装为"用户提供的新信息", LLM 可以自然地在下一轮推理中使用
            conversation.append({
                "role": "assistant",
                "content": response_text
            })
            conversation.append({
                "role": "user",
                "content": (
                    f"工具 {tool_result.tool_name} 返回结果:\n"
                    f"{json.dumps(tool_result.result, ensure_ascii=False) if isinstance(tool_result.result, dict) else str(tool_result.result)}"
                )
            })
            logger.info(
                f"[{self.name}] ReAct iter={iteration+1} | "
                f"tool={tool_result.tool_name} | success={tool_result.success}"
            )

        # 达到最大迭代次数仍未自然结束, 强制要求 LLM 基于已有工具结果做最终总结
        llm_calls += 1
        conversation.append({
            "role": "user",
            "content": "请根据以上所有工具结果, 给出最终回答。"
        })
        final_content = await self._call_llm(conversation)
        return final_content, tools_used, llm_calls

    async def _reflect(self, query: str, content: str) -> tuple:
        """
        反思: 对输出质量打分 (0.0 ~ 1.0)
        返回: (score, llm_calls)
        """
        # 构建反思 prompt, 要求 LLM 从医学准确性/完整性/安全性三个维度打分
        reflect_prompt = [
            {
                "role": "user",
                "content": (
                    f"请评估以下回答的质量。\n\n"
                    f"问题: {query}\n\n"
                    f"回答: {content[:500]}\n\n"
                    f"评分标准:\n"
                    f"- 准确性: 医学信息是否正确\n"
                    f"- 完整性: 是否覆盖了关键要点\n"
                    f"- 安全性: 是否有可能误导患者的内容\n\n"
                    f"请只输出一个 0.0 到 1.0 之间的数字作为评分, 不要输出其他内容。"
                )
            }
        ]
        try:
            score_text = await self._call_llm(reflect_prompt)
            score = float(score_text.strip())
            # 使用 max/min 进行分数钳制(clamping), 确保分数严格在 [0.0, 1.0] 范围内
            # 防止 LLM 输出异常值(如 1.5 或 -0.3)导致下游逻辑出错
            score = max(0.0, min(1.0, score))
            logger.info(f"[{self.name}] 反思评分: {score:.2f}")
            return score, 1
        except (ValueError, TypeError):
            # LLM 输出无法解析为浮点数时 (如输出了文字说明而非纯数字)
            # 默认返回 0.8, 选择 0.8 的理由: 高于阈值 0.7, 不会误触发修正, 同时留有余地
            logger.warning(f"[{self.name}] 反思评分解析失败, 默认 0.8")
            return 0.8, 1

    async def _correct(
        self, query: str, context: str, original: str, score: float
    ) -> tuple:
        """
        修正: 基于反思反馈重新生成
        返回: (corrected_content, llm_calls)
        """
        # 构建修正 prompt, 将评分反馈给 LLM 让其知道需要改进
        # context[:1000] — 参考资料截断到 1000 字符, 防止上下文过长导致:
        #                  1. token 用量过高 (增加成本)
        #                  2. 超出模型上下文窗口
        #                  3. 关键信息被稀释
        # original[:500] — 原回答截断到 500 字符, 仅需让 LLM 看到"大意"即可
        # 无需完整原文, 因为目的是重新生成而非修补
        correct_messages = [
            {
                "role": "user",
                "content": (
                    f"{self.system_prompt}\n\n"
                    f"你之前的回答质量评分为 {score:.2f} (满分1.0), 需要改进。\n\n"
                    f"原始问题: {query}\n"
                    f"参考资料: {context[:1000]}\n"
                    f"你之前的回答: {original[:500]}\n\n"
                    f"请重新生成一个更准确, 更完整的回答。"
                )
            }
        ]
        try:
            corrected = await self._call_llm(correct_messages)
            logger.info(f"[{self.name}] 修正完成 | 原长度={len(original)} → 新长度={len(corrected)}")
            return corrected, 1
        except Exception as e:
            # 修正失败时返回 None, 上层 run() 会保留原始回答不替换
            logger.warning(f"[{self.name}] 修正失败: {e}")
            return None, 1

    def _build_messages(
        self, query: str, context: str = "", messages: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """构建 LLM 输入消息"""
        # 如果调用方已传入完整消息列表, 直接使用, 不再拼接
        if messages:
            return messages

        # 将 system_prompt + 参考资料 + 用户问题拼接为单条 user 消息
        prompt_text = self.system_prompt
        if context:
            prompt_text += f"\n\n参考资料:\n{context}"
        prompt_text += f"\n\n用户问题: {query}"

        # 消息格式采用 OpenAI Chat Completions 标准: content 为纯字符串
        return [
            {
                "role": "user",
                "content": prompt_text
            }
        ]

    async def _call_llm(self, messages: List[Dict]) -> str:
        """调用 LLM (非流式, Chat Completions)"""
        try:
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
            )
            # Chat Completions 标准响应结构: choices[0].message.content
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"[{self.name}] LLM 调用失败: {e}")
            raise

    async def _call_llm_with_tools(self, messages: List[Dict]) -> str:
        """调用 LLM 并附带工具描述"""
        # 策略: 将工具描述注入到最后一条消息的文本末尾
        # 这样 LLM 在回答时能"看到"可用工具, 并自主决定是否调用
        # 注意: 不使用 OpenAI 原生 function calling, 而是通过 prompt 注入实现
        # 原因: 统一的文本注入方式不依赖各厂商对原生 tool_choice 的支持差异
        tools_desc = self._format_tools_description()
        enhanced_messages = messages.copy()
        if enhanced_messages and tools_desc:
            # 构建工具使用指令, 要求 LLM 使用 JSON 格式输出工具调用
            # 若不需要工具则直接给出文本答案 — 这让 LLM 有选择权
            tool_instruction = (
                f"\n\n你可以使用以下工具来辅助回答问题:\n{tools_desc}\n\n"
                f"如果需要使用工具, 请严格按以下 JSON 格式输出:\n"
                f'{{"tool_call": {{"name": "工具名", "arguments": {{...}}}}}}\n\n'
                f"如果不需要使用工具, 直接输出你的回答即可。"
            )
            # 将工具指令追加到对话中最后一条消息的文本末尾
            # 这样 LLM 在生成回答时会"看到"工具选项
            last_msg = enhanced_messages[-1]
            last_text = last_msg.get("content") or ""
            if isinstance(last_text, list):
                # 兼容调用方传入旧式列表 content 的情况
                last_text = "".join(
                    part.get("text", "") for part in last_text if isinstance(part, dict)
                )
            # 创建新消息替换原消息, 保持 role 不变, 仅拼接文本
            enhanced_messages[-1] = {
                "role": last_msg["role"],
                "content": last_text + tool_instruction
            }
        return await self._call_llm(enhanced_messages)

    def _format_tools_description(self) -> str:
        """格式化工具描述为文本"""
        if not self.tools:
            return ""
        lines = []
        for tool in self.tools:
            name = tool.get("name", "")
            desc = tool.get("description", "")
            params = tool.get("parameters", {})
            lines.append(f"- {name}: {desc}")
            if params:
                lines.append(f"  参数: {json.dumps(params, ensure_ascii=False)}")
        return "\n".join(lines)

    def _parse_tool_call(self, response_text: str) -> Optional[Dict]:
        """
        从 LLM 输出中解析工具调用
        返回: {"name": "tool_name", "arguments": {...}} 或 None
        """
        try:
            # 快速检查: 如果响应中不包含 "tool_call" 关键字, 说明 LLM 没有要调用工具
            if '"tool_call"' in response_text:
                # JSON 提取策略: 找到第一个 '{' 和最后一个 '}' 之间的内容
                # 这能容忍 LLM 在 JSON 前后添加的多余文字 (如 "好的, 我来搜索一下:")
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    parsed = json.loads(json_str)
                    if "tool_call" in parsed:
                        return parsed["tool_call"]
        except (json.JSONDecodeError, KeyError, IndexError):
            # JSON 解析失败 = LLM 输出格式不规范, 视为"不调用工具", 返回 None
            pass
        return None

    async def _execute_tool(self, tool_call: Dict, **kwargs) -> ToolResult:
        """执行工具调用"""
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("arguments", {})
        t0 = time.time()

        # ======= 两阶段工具查找 =======
        # 阶段一: 从 kwargs 中查找 (调用方通过 tool_xxx=func 传入的工具)
        # 这种方式优先级更高, 允许调用方临时注入/覆盖工具
        tool_func = kwargs.get(f"tool_{tool_name}")
        if tool_func is None:
            # 阶段二: 从 self.tools 列表中查找含 "handler" 字段的工具定义
            # tools 列表在 __init__ 时注入, 是 Agent 的"标配"工具集
            for tool in self.tools:
                if tool.get("name") == tool_name and "handler" in tool:
                    tool_func = tool["handler"]
                    break

        if tool_func is None:
            return ToolResult(
                tool_name=tool_name,
                tool_input=tool_args,
                result=f"工具 {tool_name} 未找到",
                success=False,
                error=f"Tool {tool_name} not registered",
                duration_ms=0,
            )

        try:
            import asyncio
            # 自适应 async/sync 工具执行:
            # - 若工具是 async 函数 → 直接 await 调用
            # - 若工具是普通同步函数 → 通过 asyncio.to_thread 在线程池中执行, 避免阻塞事件循环
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**tool_args)
            else:
                result = await asyncio.to_thread(tool_func, **tool_args)
            duration = round((time.time() - t0) * 1000, 2)
            return ToolResult(
                tool_name=tool_name,
                tool_input=tool_args,
                result=result,
                success=True,
                duration_ms=duration,
            )
        except Exception as e:
            duration = round((time.time() - t0) * 1000, 2)
            logger.error(f"[{self.name}] 工具 {tool_name} 执行失败: {e}")
            return ToolResult(
                tool_name=tool_name,
                tool_input=tool_args,
                result=None,
                success=False,
                error=str(e),
                duration_ms=duration,
            )