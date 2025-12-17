# agent/paper_agent.py
# 主控 agent：负责读取关系图、调度 ReAct 生成代码、更新知识库。

import os
import time
import logging
from typing import List

from config import prompts
from config.settings import Settings
from llm.client import get_llm_client
from utils.graph_manager import GraphManager
from utils.knowledge_base import KnowledgeBase
from utils.result_logger import ResultLogger
from utils.logger import log_token_usage
from agent.react_runner import ReActRunner, ReActRunResult
from tools.file_tools import read_file, set_allowed_paths
from tools.code_runner import run_code


class PaperAgent:
    """每个 GPU 对应一个 PaperAgent，负责顺序处理分配到的论文节点。"""

    def __init__(self, settings: Settings, gpu_id: int, logger: logging.Logger, status_dict: dict):
        self.settings = settings
        self.gpu_id = gpu_id
        self.logger = logger
        self.status_dict = status_dict
        self._set_status("Idle")

        graph_path = settings.domain_config.get("graph_path", "")
        self.graph_manager = GraphManager(graph_path)
        self.knowledge_base = KnowledgeBase(domain=settings.domain, task=settings.task)
        # allowed_files 会在每个节点运行前动态设置
        self.react_runner = None
        log_dir = settings.paths.get("logs", "logs")
        self.result_logger = ResultLogger(domain=settings.domain, task=settings.task, log_dir=log_dir)

        # LLM 用于经验总结与批次精简
        llm_cfg = settings.get("llm", {}) or {}
        api_keys = settings.get("api_keys", {}) or {}
        provider = llm_cfg.get("provider", "openai")
        base_url = llm_cfg.get("openai_base_url")
        summary_env = "SUMMARY_MODEL"
        default_model = "gpt-5.2" if provider.lower() == "openai" else "gemini-3-pro-preview-high"
        model_name = os.environ.get(summary_env, default_model)
        self.summary_llm = get_llm_client(
            api_key=api_keys.get("openai") if provider.lower() == "openai" else api_keys.get("gemini"),
            base_url=base_url,
            model_name=model_name,
            temperature=float(os.environ.get("SUMMARY_TEMPERATURE", "0.2")),
            provider=provider,
        )

    def _set_status(self, status: str):
        """更新监控板状态。"""
        self.status_dict[self.gpu_id] = status

    def _build_paths(self, method_name: str):
        """根据方法名生成代码与配置路径。"""
        code_dir = self.settings.domain_config.get("code_path", "src")
        cfg_dir = self.settings.domain_config.get("hyperparameter_path", "src/config")
        code_path = os.path.join(code_dir, f"{method_name.lower()}.py")
        config_path = os.path.join(cfg_dir, f"{method_name}.yaml")
        return code_path, config_path

    def _format_prompt(self, node_id: str, node_data: dict, mode: str):
        """填充训练/测试提示词所需的上下文。"""
        context = {"model1": "", "model2": "", "idea1": "", "idea2": "", "relation1": "", "relation2": "", "code2": "", "config2": ""}
        context.update(self.graph_manager.get_neighbors_context(node_id))
        method_name_raw = node_data.get("method_name") or node_id
        method_name = str(method_name_raw).replace(" ", "_")
        code_path, config_path = self._build_paths(method_name)
        prompt_kwargs = {
            "paper_title": node_id,
            "method_name": method_name,
            "idea": node_data.get("idea", ""),
            "method_md": node_data.get("method_md", ""),
            "hyperparam_def": node_data.get("hyperparam_def", ""),
            "code_path": code_path,
            "config_path": config_path,
        }
        prompt_kwargs.update(context)
        if mode == "train":
            user_prompt = prompts.TRAIN_USER_PROMPT.format(**prompt_kwargs)
        else:
            user_prompt = prompts.TEST_USER_PROMPT.format(**prompt_kwargs)
        system_prompt = prompts.REACT_SYSTEM_PROMPT.format(
            learned_knowledge=self.knowledge_base.load()
        )
        return system_prompt, user_prompt, method_name, code_path, config_path

    def _dialogue_snippet(self, messages: List) -> str:
        """提取对话摘要，优先保留方法描述、代码生成与运行日志。"""
        lines = []
        # 保留更多轮次，但截断单条，避免过长
        for msg in messages[-40:]:
            role = getattr(msg, "type", getattr(msg, "role", ""))
            content = msg.content if hasattr(msg, "content") else ""
            content = content if isinstance(content, str) else str(content)
            # 运行输出适当放宽截断，其他更短
            max_len = 1500 if role == "tool" else 1000
            if len(content) > max_len:
                content = content[: max_len // 2] + "\n...<truncated>...\n" + content[-max_len // 2 :]
            lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)

    def _read_snippet(self, path: str, max_len: int = 2000) -> str:
        """读取文件并截断，避免喂给 LLM 过长内容。"""
        content = read_file(path)
        if len(content) > max_len:
            return content[: max_len // 2] + "\n...<truncated>...\n" + content[-max_len // 2 :]
        return content

    def _summarize_experience(self, messages: List, code_path: str, config_path: str) -> str:
        """让 LLM 回顾对话与最终代码，生成经验条目。"""
        prompt = prompts.EXPERIENCE_SUMMARY_PROMPT.format(
            dialogue=self._dialogue_snippet(messages),
            code_content=self._read_snippet(code_path),
            config_content=self._read_snippet(config_path),
        )
        resp = self.summary_llm.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)

    def process_nodes(self, nodes: list, mode: str):
        """顺序处理分配的节点，包含状态回报与结果记录。"""
        self.logger.info(f"Starting {mode} task for {len(nodes)} nodes.")
        for node in nodes:
            start_time = time.time()
            success = False
            result_content = ""
            try:
                self._set_status(f"启动 {node}")
                node_data = self.graph_manager.graph.nodes[node]
                system_prompt, user_prompt, method_name, code_path, config_path = self._format_prompt(node, node_data, mode)
                self.logger.info(f"[{method_name}] System prompt: {system_prompt[:40000]}")
                # 针对该论文限定可读写文件，防止越界透题
                self.react_runner = ReActRunner(
                    settings=self.settings,
                    gpu_id=self.gpu_id,
                    logger=self.logger,
                    allowed_files=[os.path.abspath(code_path), os.path.abspath(config_path)],
                )
                set_allowed_paths([code_path, config_path])

                max_attempts = int(os.environ.get("REACT_MAX_ATTEMPTS", "15"))
                react_result: ReActRunResult = None
                chat_history: List = []

                for attempt in range(1, max_attempts + 1):
                    self._set_status(f"{method_name} 第{attempt}轮生成/运行")
                    self.logger.info(f"[{method_name}] Attempt {attempt} user prompt: {user_prompt[:40000]}")
                    react_result = self.react_runner.run(system_prompt, user_prompt, history=chat_history)
                    # 累加历史，避免覆盖前几轮的上下文
                    chat_history = (chat_history or []) + (react_result.messages or [])
                    result_content = react_result.run_output or react_result.final_output
                    # 记录对话尾部，方便排查
                    self.logger.info(f"[{method_name}] Attempt {attempt} final_output: {react_result.final_output[:6000]}")
                    # 记录 token 消耗
                    if react_result.token_usage:
                        log_token_usage(self.react_runner.model.model_name, react_result.token_usage)

                    # 运行代码（编排式）
                    run_timeout = self.settings.timeouts.get("code_runner", 300)
                    run_result = run_code(
                        model_name=method_name,
                        timeout=run_timeout,
                        gpu_id=self.gpu_id,
                    )
                    exit_code = run_result.get("exit_code")
                    stdout = run_result.get("stdout", "")
                    stderr = run_result.get("stderr", "")
                    run_log = (stdout or "") + ("\n" if stdout and stderr else "") + (stderr or "")
                    if run_result.get("timed_out"):
                        self.logger.warning(f"[{method_name}] run_code timed out after {run_timeout}s")
                    self.logger.info(f"[{method_name}] run_code exit={exit_code}, log tail:\n{run_log[-2000:]}")

                    if exit_code == 0:
                        success = True
                        break

                    user_prompt = prompts.RETRY_FIX_PROMPT.format(
                        last_output=run_log[-2000:],
                        code_path=code_path,
                        config_path=config_path,
                        method_name=method_name,
                    )

                if success and mode == "train":
                    self._set_status(f"写入经验 {method_name}")
                    summary = self._summarize_experience(react_result.messages, code_path, config_path)
                    if summary.strip():
                        # 仅将新经验写入待审核区，后续批次统一筛选入主库
                        self.knowledge_base.append_pending(summary)
            except Exception as e:
                self.logger.error(f"处理节点 {node} 发生异常: {e}", exc_info=True)
                success = False
                result_content = str(e)
            finally:
                elapsed = time.time() - start_time
                status_str = "Success" if success else "Failure"
                error_reason = "" if success else result_content
                self.result_logger.log(node, status_str, elapsed, error_reason)
                self.logger.info(f"Finished node {node} in {elapsed:.2f}s | Status={status_str}")
                self._set_status(f"{node} 完成 | {status_str}")
                # 清空本线程白名单，确保论文间隔离
                set_allowed_paths([])

        self._set_status("Idle")
        self.logger.info("Task finished.")
