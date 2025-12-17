# agent/react_runner.py
# 基于 LangChain/LangGraph 的 ReAct 封装，提供受限工具集与运行结果解析。

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional

from langchain.tools import tool
from langchain_core.messages import BaseMessage

try:
    # 优先使用 LangGraph 版本
    from langgraph.prebuilt import create_react_agent
except Exception:  # pragma: no cover
    from langchain.agents import create_react_agent

from config.settings import Settings
from llm.client import get_llm_client
from tools.code_runner import run_code
from tools.file_tools import (
    restricted_read,
    restricted_write,
    apply_restricted_patch,
)


@dataclass
class ReActRunResult:
    """封装 ReAct 运行结果，便于上层判断与记录。"""

    success: bool
    exit_code: Optional[int]
    run_output: str
    messages: List[BaseMessage]
    final_output: str
    token_usage: dict


class ReActRunner:
    """负责创建 ReAct agent、管理工具集并解析运行状态。"""

    def __init__(self, settings: Settings, gpu_id: int, logger: logging.Logger, allowed_files: Optional[List[str]] = None):
        self.settings = settings
        self.gpu_id = gpu_id
        self.logger = logger
        self.allowed_files = allowed_files or []
        provider_cfg = settings.get("llm", {}) or {}
        api_keys = settings.get("api_keys", {}) or {}
        provider = provider_cfg.get("provider", "openai")
        base_url = provider_cfg.get("openai_base_url")
        model_env = "GEMINI_MODEL" if provider.lower() == "gemini" else "OPENAI_MODEL"
        model_name = os.environ.get(model_env, "gpt-5.2" if provider.lower() == "openai" else "gemini-3-pro-preview-high")
        self.recursion_limit = int(os.environ.get("REACT_RECURSION_LIMIT", "16"))
        self.model = get_llm_client(
            api_key=api_keys.get("openai") if provider.lower() == "openai" else api_keys.get("gemini"),
            base_url=base_url,
            model_name=model_name,
            temperature=float(os.environ.get("TEMPERATURE", "0.2")),
            provider=provider,
        )
        self.tools = self._build_tools()

    def _build_tools(self):
        """构造带闭包参数的工具集合。"""
        settings = self.settings
        timeout = settings.timeouts.get("code_runner", 300)
        gpu_id = self.gpu_id
        allowed_files = [os.path.abspath(p) for p in (self.allowed_files or [])]

        @tool
        def tool_read(file_path: str) -> str:
            """读取指定代码/配置文件，超长截断。"""
            return restricted_read(file_path, allowed_files)

        @tool
        def tool_write(file_path: str, content: str) -> str:
            """写入指定代码/配置文件（覆盖写）。"""
            return restricted_write(file_path, content, allowed_files)

        @tool
        def tool_patch(unified_diff: str) -> str:
            """对指定代码/配置文件应用补丁。"""
            return apply_restricted_patch(unified_diff, allowed_files)

        return [
            tool_read,
            tool_write,
            tool_patch,
        ]

    def _aggregate_usage_from_messages(self, messages) -> dict:
        """
        聚合消息中的 token 使用信息，兼容 LangGraph/LangChain 元数据。
        """
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        def _acc(prompt, completion, total):
            totals["prompt_tokens"] += prompt
            totals["completion_tokens"] += completion
            totals["total_tokens"] += total

        for msg in messages or []:
            usage = None
            meta = getattr(msg, "response_metadata", None) or {}
            if isinstance(meta, dict):
                usage = meta.get("token_usage") or meta.get("usage")

            if usage is None:
                usage = getattr(msg, "usage_metadata", None)

            if usage is None and hasattr(msg, "additional_kwargs"):
                ak = getattr(msg, "additional_kwargs", {}) or {}
                usage = ak.get("usage_metadata") or ak.get("token_usage") or ak.get("usage")

            if not usage or not isinstance(usage, dict):
                continue

            prompt = usage.get("prompt_tokens") or usage.get("prompt") or usage.get("input_tokens") or 0
            completion = usage.get("completion_tokens") or usage.get("completion") or usage.get("output_tokens") or 0
            total = usage.get("total_tokens") or usage.get("total") or (prompt + completion)

            try:
                _acc(int(prompt or 0), int(completion or 0), int(total or 0))
            except Exception:
                continue

        return totals

    def run(self, system_prompt: str, user_prompt: str, history: Optional[List[BaseMessage]] = None) -> ReActRunResult:
        """执行一次 ReAct 对话，返回结果摘要，可带入历史消息维持上下文。"""
        agent = create_react_agent(
            model=self.model,
            tools=self.tools,
            prompt=system_prompt,
            debug=os.environ.get("AGENT_DEBUG", "0") == "1",
        )
        input_messages = list(history or [])
        input_messages.append({"role": "user", "content": user_prompt})
        inputs = {"messages": input_messages}
        out = agent.invoke(inputs, config={"recursion_limit": self.recursion_limit})
        messages = out.get("messages", [])
        final_output = messages[-1].content if messages else ""
        # command execution moved to orchestrator; success determined outside
        exit_code, run_output = None, ""
        success = False
        token_usage = self._aggregate_usage_from_messages(messages)
        return ReActRunResult(
            success=success,
            exit_code=exit_code,
            run_output=run_output,
            messages=messages,
            final_output=final_output,
            token_usage=token_usage,
        )
