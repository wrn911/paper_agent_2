#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
单文件 Paper Reproduction ReAct Agent（LangChain baseline）

输入（放在工作目录根目录）：
  - paper.md                 # 论文原文（md）
  - DATA_CONTRACT.md (可选)  # 你对数据格式/字段/metric 的补充说明（强烈建议）
  - ./data/ (可选)           # 数据目录（agent 会探测）

输出（agent 会生成/更新）：
  - repro_spec.yaml          # 从论文抽取的“复现规格”
  - repro_log.md             # 每次运行命令的日志（自动追加）
  - src/* configs/* logs/* results/*
"""

from __future__ import annotations

import os
# 1) 必填：OpenAI API Key（把你的 key 粘贴进来）
os.environ["OPENAI_API_KEY"] = "sk-1OUmm4rEXt4Hk3eB4HxWrgBD9ImjOINn9pxXIEx6rwxm68QR"
# 2) 可选：如果你要“替换 URL”（例如走代理/兼容端点），就改成你的 URL
os.environ["OPENAI_BASE_URL"] = "https://api2.aigcbest.top/v1"  # <- 换成你的代理地址，如 https://xxx/v1
# 3) 可选：默认模型名（你代码里也会读 OPENAI_MODEL）
os.environ["OPENAI_MODEL"] = "gpt-5.2"

import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

from langchain.tools import tool
from langchain_openai import ChatOpenAI
try:
    # LangGraph <-> LangChain 导入兼容：优先 LangGraph，若不可用再尝试 LangChain
    from langgraph.prebuilt import create_react_agent
except ImportError:  # pragma: no cover - 兼容性兜底
    try:
        from langchain.agents import create_react_agent
    except Exception:
        from langchain.agents import create_agent as create_react_agent


# =========================
# 0) 全局配置：工作目录、输出截断等
# =========================

# 约定：所有读写都限制在 WORKSPACE_ROOT 下，避免路径逃逸
WORKSPACE_ROOT = Path(os.environ.get("WORKSPACE", ".")).resolve()

# 读文件最大字符数（避免一次把大文件塞进上下文导致 token 爆炸）
# 普通文件收紧一些，论文类文件放宽更多上下文
MAX_READ_CHARS = int(os.environ.get("MAX_READ_CHARS", "6000"))
# 论文/长文档优先保留更多上下文
PAPER_MAX_READ_CHARS = int(os.environ.get("PAPER_MAX_READ_CHARS", "40000"))

# run() 工具返回 stdout+stderr 的最大字符数（保留尾部更利于定位报错）
MAX_RUN_OUTPUT_CHARS = int(os.environ.get("MAX_RUN_OUTPUT_CHARS", "4000"))

# 复现日志：每次 run() 自动追加一条记录（即使 agent 忘记写日志也不会丢）
REPRO_LOG = WORKSPACE_ROOT / "repro_log.md"


def _safe_rel(path: Path) -> str:
    """把绝对路径尽量转换成相对路径输出，便于阅读日志。"""
    try:
        return str(path.relative_to(WORKSPACE_ROOT))
    except Exception:
        return str(path)


def _ensure_within_workspace(path: Path) -> Path:
    """
    安全检查：禁止访问工作目录之外的路径。
    - 任何工具读写文件前都要过这层检查
    """
    rp = path.resolve()
    if WORKSPACE_ROOT not in rp.parents and rp != WORKSPACE_ROOT:
        raise ValueError(f"Path escapes workspace: {rp}")
    return rp


def _truncate(text: str, max_chars: int) -> str:
    """
    截断策略：保留开头一半 + 结尾一半，便于既看到上下文又看到报错尾部。
    """
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return head + "\n...<truncated>...\n" + tail


def _max_read_limit_for(path: Path) -> int:
    """
    论文类文件用更大的截断阈值，其余文件保持较小阈值。
    """
    name = path.name.lower()
    if name == "paper.md" or name.startswith("paper."):
        return PAPER_MAX_READ_CHARS
    return MAX_READ_CHARS


def _append_repro_log(section: str) -> None:
    """
    追加写 repro_log.md：run() 工具会调用它自动记录命令与输出尾部。
    """
    REPRO_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with REPRO_LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n\n## {ts}\n")
        f.write(section.rstrip() + "\n")


def _aggregate_usage_from_messages(messages) -> dict:
    """
    兜底：从 LangGraph/LC 消息的 metadata/usage_metadata 中聚合 token 信息。
    适用于 callback 无法正常统计的情况。
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

        # LangGraph 可能把 usage 放在 usage_metadata
        if usage is None:
            usage = getattr(msg, "usage_metadata", None)

        # 也可能挂在 additional_kwargs
        if usage is None and hasattr(msg, "additional_kwargs"):
            ak = getattr(msg, "additional_kwargs", {}) or {}
            usage = ak.get("usage_metadata") or ak.get("token_usage") or ak.get("usage")

        if not usage or not isinstance(usage, dict):
            continue

        prompt = (
            usage.get("prompt_tokens")
            or usage.get("prompt")
            or usage.get("input_tokens")
            or 0
        )
        completion = (
            usage.get("completion_tokens")
            or usage.get("completion")
            or usage.get("output_tokens")
            or 0
        )
        total = usage.get("total_tokens") or usage.get("total") or (prompt + completion)

        try:
            _acc(int(prompt or 0), int(completion or 0), int(total or 0))
        except Exception:
            continue

    return totals

# =========================
# 1) 工具定义：尽量少而强（便于调试）
# =========================

@tool
def list_files(pattern: str) -> str:
    """
    列出工作目录内匹配 glob 的文件（用于探测 data 目录、repo 结构等）
    示例：'**/*.py', 'data/**', 'configs/*.yaml'
    """
    if not pattern or len(pattern) > 200:
        return "ERROR: invalid pattern"

    matches = sorted(WORKSPACE_ROOT.glob(pattern))
    lines = []
    for p in matches[:300]:  # 防止返回过多导致上下文爆炸
        try:
            p = _ensure_within_workspace(p)
            lines.append(_safe_rel(p))
        except Exception:
            continue

    if len(matches) > 300:
        lines.append(f"... <{len(matches) - 300} more>")

    return "\n".join(lines) if lines else "<no matches>"


@tool
def read_text(path: str) -> str:
    """
    读取 UTF-8 文本文件（md/yaml/py/log 等），并自动截断。
    大文件只返回部分内容，避免 token 爆炸；paper.* 使用更大的截断阈值。
    """
    try:
        p = _ensure_within_workspace(WORKSPACE_ROOT / path)
        if not p.exists():
            return f"ERROR: file not found: {path}"

        data = p.read_text(encoding="utf-8", errors="replace")
        max_chars = _max_read_limit_for(p)
        return _truncate(data, max_chars)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def write_text(path: str, content: str, mode: str = "overwrite") -> str:
    """
    写入 UTF-8 文本文件。
    - mode='overwrite' 覆盖写
    - mode='append'    追加写（用于日志或逐步生成）
    """
    try:
        if mode not in ("overwrite", "append"):
            return "ERROR: mode must be overwrite|append"

        p = _ensure_within_workspace(WORKSPACE_ROOT / path)
        p.parent.mkdir(parents=True, exist_ok=True)

        m = "w" if mode == "overwrite" else "a"
        with p.open(m, encoding="utf-8") as f:
            f.write(content)

        return f"OK: wrote {_safe_rel(p)} ({mode})"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def apply_patch(unified_diff: str) -> str:
    """
    应用 unified diff patch（让 agent 做“最小修改”，更可控、更好 review）
    - 如果 workspace 有 .git，优先用 git apply（兼容性好）
    - 否则退化使用 patch -p0
    """
    try:
        if not unified_diff.strip():
            return "ERROR: empty diff"

        # 简单安全检查：禁止 diff 里出现绝对路径（避免改到工作目录外）
        if re.search(r"^\+\+\+\s+/(?!dev/null)", unified_diff, re.M):
            return "ERROR: absolute paths not allowed in diff"

        # 把 diff 写到临时文件，再交给 git apply/patch
        patch_path = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tf:
                tf.write(unified_diff)
                patch_path = tf.name

            git_dir = WORKSPACE_ROOT / ".git"
            if git_dir.exists() and git_dir.is_dir():
                cmd = ["git", "apply", "--whitespace=nowarn", patch_path]
            else:
                cmd = ["patch", "-p0", "-i", patch_path]

            proc = subprocess.run(
                cmd,
                cwd=str(WORKSPACE_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
            )

            out = (proc.stdout or "") + (proc.stderr or "")
            out = _truncate(out, 4000)

            if proc.returncode != 0:
                return f"ERROR: patch failed (code={proc.returncode})\n{out}"

            return f"OK: patch applied\n{out}".strip()
        finally:
            if patch_path and os.path.exists(patch_path):
                try:
                    os.remove(patch_path)
                except OSError:
                    pass
    except FileNotFoundError as e:
        # 如果没装 git/patch，会走到这里（基线场景建议装 git）
        return f"ERROR: missing tool: {e}. Install git or patch."
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# run() 白名单：为了 baseline 好调试 + 安全可控，禁止任意 shell
_ALLOWED_PREFIXES = (
    "python ",
    "pytest ",
)


@tool
def run(cmd: str, timeout_sec: int = 600) -> str:
    """
    在工作目录执行命令（仅允许白名单前缀），返回 stdout+stderr（截断后）。
    额外行为：每次执行都会自动把命令、返回码、输出尾部写进 repro_log.md
    """
    cmd = (cmd or "").strip()
    if not cmd:
        return "ERROR: empty command"

    # 白名单约束：避免 agent 执行 rm -rf 等危险命令，也更利于复现一致性
    if not any(cmd == p or cmd.startswith(p) for p in _ALLOWED_PREFIXES):
        return (
            "ERROR: command not allowed.\n"
            f"Allowed prefixes: {', '.join(_ALLOWED_PREFIXES)}"
        )

    try:
        argv = shlex.split(cmd)
        proc = subprocess.run(
            argv,
            cwd=str(WORKSPACE_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=int(timeout_sec),
        )

        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        combined_trunc = _truncate(combined, MAX_RUN_OUTPUT_CHARS)

        # 自动追加到 repro_log.md：方便你调试 agent 为什么失败/怎么修
        _append_repro_log(
            "\n".join(
                [
                    f"- cmd: `{cmd}`",
                    f"- return_code: {proc.returncode}",
                    "- output_tail:",
                    "```",
                    combined_trunc,
                    "```",
                ]
            )
        )

        # 同时把信息返回给 agent，让它据此修复
        return f"RETURN_CODE={proc.returncode}\n{combined_trunc}"
    except subprocess.TimeoutExpired:
        _append_repro_log(f"- cmd: `{cmd}`\n- result: TIMEOUT after {timeout_sec}s")
        return f"ERROR: TIMEOUT after {timeout_sec}s"
    except Exception as e:
        _append_repro_log(f"- cmd: `{cmd}`\n- result: EXCEPTION {type(e).__name__}: {e}")
        return f"ERROR: {type(e).__name__}: {e}"


# =========================
# 2) Agent Prompt：把 baseline 的“工程规约”写死，避免漂移
# =========================

SYSTEM_PROMPT = """
你是一个论文复现 ReAct agent（LangGraph create_react_agent）。

目标：给定 paper.md（可选 DATA_CONTRACT.md），生成可运行代码并复现论文主实验的闭环。

强规则：
1) 第一件事：创建 repro_spec.yaml，总结 task/dataset/model/train/eval（来自 paper.md + DATA_CONTRACT.md），列出“假设/缺失信息”清单。
2) 永远不要凭空猜数据 schema：优先读取 DATA_CONTRACT.md；没有则用 list_files/read_text 探测 ./data。
3) 优先忠实实现论文描述的模型/损失/优化/数据处理；严禁为了“跑通”而换成未在论文出现的简化 baseline，除非 paper.md 明确不可复现且已在 repro_spec.yaml 标注假设/缺失项并在最终回复中列出偏差。
4) 不要生成/依赖 run.sh；直接用 run() 执行 python 命令完成训练+评估。
5) 固定命令格式（便于批量运行）：统一使用
   - Gate1: `python src/train.py --config configs/default.yaml --smoke --device ${DEVICE}` 然后 `python src/eval.py --config configs/default.yaml --smoke --device ${DEVICE}`
   - Gate2: `python src/train.py --config configs/default.yaml --device ${DEVICE}` 然后 `python src/eval.py --config configs/default.yaml --device ${DEVICE}`
   - DEVICE 优先用 `cuda`（如果 torch.cuda.is_available），否则 `cpu`；可通过环境变量或配置传入，但命令里必须显式包含 `--device ...`
6) 因此你生成的 src/train.py 和 src/eval.py 必须支持 --smoke 和 --device 参数，并在 --smoke 模式下：
   - 使用小数据子集或限制 step 数
   - 运行时间短，且能产出基本的 metrics 输出（哪怕是临时的）
6.1) 调试/验证阶段不要跑完整训练，全程关注“跑通闭环”，把训练轮数/step 控制在很小范围，避免长时间 full train。
6.2) 切勿用 read_text/read_file 之类直接把大型 csv/npy/parquet/raw data 全量读进上下文；如需查看数据格式，优先用 list_files/小片段抽样或代码内按需流式读取，避免上下文爆炸。
7) 每次 run() 后，根据输出制定下一步修复计划。
8) 控制 token：读文件尽量分段、少于 15k 字符；长日志/代码优先 summarize，再按需定位局部；避免重复粘贴相同大段。
9) 不要自行 pip 安装；假设环境已有常用包，若缺依赖只能在回复中提示用户手动安装。
10) 所有执行的命令要在回复中记录，并依赖 run() 自动写 repro_log.md；每轮总结时说明与论文的符合项/偏差项。
11) 如遇缺失信息导致无法按论文实现，优先尝试在 repro_spec.yaml 中列出缺口并提出最小可行假设；若仍无法闭环，应停止并报告阻塞原因（不要静默换成自创方案）。
12) 终止条件：full run 的 eval 能跑完且存在 results/metrics.json；或达到 recursion_limit。
"""


USER_TASK = """请严格按 paper.md 复现。数据在 ./data（如果存在）。
优先跑通闭环：生成代码，直接用固定格式的 python 命令完成训练+评估，输出 results/metrics.json，命令里显式携带 --device（默认 cuda，fallback cpu）。
如果 paper.md 对数据格式不清楚，请优先读取 DATA_CONTRACT.md；没有的话就用 list_files/read_text 探测 data 目录并做最小可运行假设。
如因缺失信息被迫做假设/替代方案，必须在 repro_spec.yaml 和最终回复中明确标注“偏离论文”的部分，避免静默换成自创模型。
"""


# =========================
# 3) 主程序：初始化模型、创建 agent、设置 recursion_limit
# =========================

def main() -> None:
    """
    主入口：
    - 初始化 ChatOpenAI
    - create_react_agent(...) 生成 agent runnable
    - invoke(...) 执行，设置 recursion_limit 防止无限循环
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required（请用环境变量传入，不要硬编码在仓库里）")

    # 模型名可通过环境变量覆盖（比如 gpt-4.1 / gpt-4.1-mini 等）
    model_name = os.environ.get("OPENAI_MODEL", "gpt-5.2")

    # temperature 建议低一些（更稳定、少发散）；timeout/max_retries 便于工程运行
    model = ChatOpenAI(
        model=model_name,
        temperature=float(os.environ.get("TEMPERATURE", "0.1")),
        timeout=float(os.environ.get("OPENAI_TIMEOUT", "100")),
        max_retries=int(os.environ.get("OPENAI_MAX_RETRIES", "5")),
    )

    # 工具集：保持极简，便于调试与可控
    tools = [list_files, read_text, write_text, apply_patch, run]

    # debug=True：会输出 agent 内部执行过程，排查卡住原因很有用
    graph = create_react_agent(
        model=model,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        debug=os.environ.get("AGENT_DEBUG", "1") == "1",
    )

    # 消息格式：LangChain 通用 messages list（role/content）
    inputs = {"messages": [{"role": "user", "content": USER_TASK}]}

    # recursion_limit：硬上限，避免 agent 反复修 bug 无止境
    recursion_limit = int(os.environ.get("RECURSION_LIMIT", "60"))

    # 简化：不依赖回调，直接运行
    out = graph.invoke(inputs, config={"recursion_limit": recursion_limit})

    # 打印最终输出（通常是 agent 的总结/下一步说明）
    print(out["messages"][-1].content)

    # 直接从消息中的 usage_metadata / token_usage 聚合 token
    agg = _aggregate_usage_from_messages(out.get("messages") or [])
    prompt_tokens = agg.get("prompt_tokens", 0)
    completion_tokens = agg.get("completion_tokens", 0)
    total_tokens = agg.get("total_tokens", 0)
    if prompt_tokens or completion_tokens or total_tokens:
        print("\nToken usage (from messages):")
        print(f"- prompt_tokens: {prompt_tokens}")
        print(f"- completion_tokens: {completion_tokens}")
        print(f"- total_tokens: {total_tokens}")


if __name__ == "__main__":
    main()
