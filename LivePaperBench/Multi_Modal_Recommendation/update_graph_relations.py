#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量调用大模型，生成 graph.json 中每条边的 relation_text。

使用方式示例：
python update_graph_relations.py --graph graph.json --model gpt-4o-mini --only-missing

注意：
- 默认直接写回原文件，先备份或使用 --output 输出到新文件。
- 可通过 --api-base/--api-key 显式指定第三方推理服务地址与密钥。
- 可加 --print-raw 查看模型原始输出，便于排查格式问题。
- 可加 --request-timeout 设置单次请求超时时间，避免长时间卡住。
- 默认使用 Chat Completions 的 response_format=json_object 强约束输出；如模型不支持可加 --no-response-format。
- 需已安装 openai 库并配置好 OPENAI_API_KEY；若仅想预览提示词，可加 --dry-run。
"""
import argparse
import json
import pathlib
import re
import sys
from typing import Any, Dict, Optional, Tuple

try:
    # 可选依赖，运行时再检查
    from openai import OpenAI
except ImportError:  # pragma: no cover - 环境缺失 openai 库时走备用路径
    OpenAI = None


def load_graph(path: pathlib.Path) -> Dict[str, Any]:
    """加载图文件。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_graph(data: Dict[str, Any], path: pathlib.Path) -> None:
    """安全写回图文件，先写临时文件再替换。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def pick_method_text(node: Dict[str, Any]) -> str:
    """选择方法全文，优先 method_md，其次 idea，不做截断。"""
    return node.get("method_md") or node.get("idea") or ""


def build_prompt(src_node: Dict[str, Any], tgt_node: Dict[str, Any]) -> str:
    """根据新模板构造提示词，附上源/目标论文全文。"""
    src_text = pick_method_text(src_node)
    tgt_text = pick_method_text(tgt_node)
    src_meta = f"Title: {src_node.get('paper_title')}\nYear: {src_node.get('year')}\nDomain: {src_node.get('domain')}\nMethod Name: {src_node.get('method_name')}\n\nFull text:\n{src_text}"
    tgt_meta = f"Title: {tgt_node.get('paper_title')}\nYear: {tgt_node.get('year')}\nDomain: {tgt_node.get('domain')}\nMethod Name: {tgt_node.get('method_name')}\n\nFull text:\n{tgt_text}"
    return f"""You are a researcher familiar with recommender systems and multimodal recommendation.

I will provide two full papers as attachments.

[Task]
Analyze the key similarities and differences between two papers

[Guidelines]
1. Be as concise as possible; keep only high-signal points.
2. Focus on design-level and conceptual differences, including:
   - learning objective and problem formulation
   - representation design and modeling granularity
   - architectural choices that reflect design philosophy
   - training paradigm and supervision signals (e.g., masking, augmentation, contrastive learning, negative sampling)
   - usage and fusion of multimodal information
3. Ignore obvious or trivial similarities (e.g., same field, same task, same evaluation protocol).
4. Prefer differences that would matter for understanding, re-implementing, or choosing between the two methods.
5. Each bullet should be short and specific (at most one sentence).

[Attachments]
Paper P (source):
{src_meta}

Paper Q (target):
{tgt_meta}

[Output format — MUST FOLLOW]
Return only the following JSON in a code block, with no extra text:

{{
  "relation_text": "Similarities:\\n1. ...\\n2. ...\\n\\nDifferences:\\n1. ...\\n2. ..."
}}

[Additional constraints on relation_text]
- Use exactly two sections: Similarities and Differences
- Use numbered lists (1., 2., ...)
- Keep 2–4 items per section unless absolutely necessary"""


def extract_json_from_response(resp_text: str) -> Dict[str, Any]:
    """
    从模型回复中提取 JSON。
    优先解析 code block 内 JSON，若不存在则回退到首个大括号段。
    """
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", resp_text, re.S)
    if code_block:
        text = code_block.group(1)
    else:
        brace = re.search(r"(\{.*\})", resp_text, re.S)
        if not brace:
            raise ValueError("未找到 JSON 片段")
        text = brace.group(1)
    return json.loads(text)


def coerce_content(value: Any) -> str:
    """将 content 字段转成字符串，兼容列表结构。"""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(item.get("text") or item.get("content") or "")
            elif isinstance(item, str):
                parts.append(item)
        return "".join(p for p in parts if p).strip()
    return ""


def response_to_dict(resp: Any) -> Dict[str, Any]:
    """尽可能把响应转为字典结构，便于后续解析。"""
    if isinstance(resp, dict):
        return resp
    for attr in ("model_dump", "to_dict"):
        if hasattr(resp, attr):
            try:
                data = getattr(resp, attr)()
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
    try:
        choices_obj = getattr(resp, "choices", None)
        if choices_obj:
            choices_list = []
            for ch in choices_obj:
                if isinstance(ch, dict):
                    choices_list.append(ch)
                    continue
                ch_dict: Dict[str, Any] = {}
                for key in ("message", "content", "text"):
                    val = getattr(ch, key, None)
                    if val is not None:
                        ch_dict[key] = val
                choices_list.append(ch_dict)
            return {"choices": choices_list}
    except Exception:
        pass
    return {}


def extract_content_from_resp(resp: Any) -> str:
    """从响应（或其 dict 形式）提取正文文本。"""
    data = response_to_dict(resp)
    if not data:
        return ""
    choices = data.get("choices") or []
    if choices:
        c0 = choices[0] or {}
        if isinstance(c0, dict):
            msg = c0.get("message") or {}
            if isinstance(msg, dict):
                content = coerce_content(msg.get("content"))
                if content:
                    return content
            content = coerce_content(c0.get("content"))
            if content:
                return content
            content = coerce_content(c0.get("text"))
            if content:
                return content
    return ""


class LLMClient:
    """简单封装 openai ChatCompletion，支持 dry-run。"""

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        dry_run: bool,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        use_response_format: bool = True,
        timeout: Optional[float] = None,
        debug_print_response: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.dry_run = dry_run
        self.base_url = base_url
        self.api_key = api_key
        self.use_response_format = use_response_format
        self.timeout = timeout
        self.debug_print_response = debug_print_response
        if not dry_run:
            if OpenAI is None:
                raise RuntimeError("未安装 openai 库，请先 `pip install openai`")
            client_kwargs: Dict[str, Any] = {}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            self.client = OpenAI(**client_kwargs)

    def chat(self, prompt: str) -> Tuple[Dict[str, Any], str]:
        """
        返回 (结构化结果, 原始文本)。
        dry-run 模式下返回示例结构，方便检查提示词。
        """
        if self.dry_run:
            dummy = {
                "relation_text": "Similarities:\n1. 占位相似点\n2. 占位相似点\n\nDifferences:\n1. 占位差异点\n2. 占位差异点"
            }
            return dummy, json.dumps(dummy, ensure_ascii=False)

        resp_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.use_response_format:
            resp_kwargs["response_format"] = {"type": "json_object"}
        if self.timeout and self.timeout > 0:
            resp_kwargs["timeout"] = self.timeout

        while True:
            try:
                resp = self.client.chat.completions.create(**resp_kwargs)
                break
            except TypeError as exc:  # pragma: no cover - 运行时保护
                if "timeout" in resp_kwargs and "timeout" in str(exc):
                    print(f"[warn] 接口不接受 timeout 参数，回退无超时: {exc}", file=sys.stderr)
                    resp_kwargs.pop("timeout", None)
                    continue
                raise
            except Exception as exc:  # pragma: no cover - 运行时保护
                if self.use_response_format and "response_format" in resp_kwargs:
                    print(f"[warn] response_format 调用失败，回退无格式请求: {exc}", file=sys.stderr)
                    resp_kwargs.pop("response_format", None)
                    continue
                raise
        resp_data = response_to_dict(resp)
        if self.debug_print_response:
            try:
                print(f"[resp] {json.dumps(resp_data, ensure_ascii=False)}", file=sys.stderr, flush=True)
            except Exception:
                print(f"[resp] {resp_data}", file=sys.stderr, flush=True)

        content = extract_content_from_resp(resp_data)
        if not content:
            content = coerce_content(getattr(resp, "content", None))
        try:
            parsed = extract_json_from_response(content)
        except Exception as exc:  # pragma: no cover - 运行时保护
            # 回退：直接使用原文，避免整批任务中断
            print(f"[warn] 解析模型回复失败，使用原文回退: {exc}", file=sys.stderr)
            parsed = {"relation_text": (content or "").strip()}
        return parsed, content


def process_edge(edge: Dict[str, Any], nodes: Dict[str, Dict[str, Any]], args, llm: LLMClient) -> None:
    """处理单条边，调用大模型并写回 relation_text。"""
    src = nodes.get(edge["source"])
    tgt = nodes.get(edge["target"])
    if not src or not tgt:
        raise KeyError(f"找不到节点：{edge}")

    prompt = build_prompt(src, tgt)
    if getattr(args, "print_raw", False):
        print(f"[call] edge {edge['source']} -> {edge['target']}", file=sys.stderr, flush=True)
    result, raw = llm.chat(prompt)
    if getattr(args, "print_raw", False):
        print(f"[raw] edge {edge['source']} -> {edge['target']}:\n{raw}\n", file=sys.stderr, flush=True)
    # 仅保留字符串，不新增其他字段
    edge["relation_text"] = result.get("relation_text", "").strip()


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="批量生成 graph.json 边的 relation_text")
    p.add_argument("--graph", type=pathlib.Path, default=pathlib.Path("graph.json"), help="输入图文件路径")
    p.add_argument("--output", type=pathlib.Path, default=None, help="输出路径，默认覆盖输入文件")
    p.add_argument("--model", type=str, default="gpt-5", help="大模型名称")
    p.add_argument("--api-base", type=str, default="https://api2.aigcbest.top/v1", help="大模型服务地址，如 https://api.example.com/v1")
    p.add_argument("--api-key", type=str, default="sk-1OUmm4rEXt4Hk3eB4HxWrgBD9ImjOINn9pxXIEx6rwxm68QR", help="大模型服务密钥，留空则使用环境变量")
    p.add_argument("--no-response-format", action="store_true", help="禁用 response_format=json_object（模型不支持时可用）")
    p.add_argument("--request-timeout", type=float, default=60.0, help="单次请求超时时间（秒），<=0 表示不设")
    p.add_argument("--temperature", type=float, default=0.2, help="采样温度")
    p.add_argument("--max-tokens", type=int, default=8000, help="回复最大 token 数")
    p.add_argument("--print-raw", action="store_true", help="打印模型原始输出（stderr），便于调试格式问题")
    p.add_argument("--print-response", action="store_true", help="打印完整接口响应（stderr，可能较长）")
    p.add_argument("--only-missing", action="store_true", help="仅处理缺少 relation_text 的边")
    p.add_argument("--limit", type=int, default=0, help="最多处理多少条边，0 表示全部")
    p.add_argument("--dry-run", action="store_true", help="不调用大模型，返回占位数据以调试提示词")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    graph_path = args.graph
    out_path = args.output or graph_path

    graph = load_graph(graph_path)
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    edges = graph.get("edges", [])

    llm = LLMClient(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        dry_run=args.dry_run,
        base_url=args.api_base,
        api_key=args.api_key,
        use_response_format=not args.no_response_format,
        timeout=args.request_timeout if args.request_timeout and args.request_timeout > 0 else None,
        debug_print_response=args.print_response,
    )

    processed = 0
    for edge in edges:
        # 仅处理两个端点都有示例代码的边（has_example_code 为 true），其他不改动
        src_node = nodes.get(edge["source"])
        tgt_node = nodes.get(edge["target"])
        if not src_node or not tgt_node:
            continue
        if not src_node.get("has_example_code") or not tgt_node.get("has_example_code"):
            continue
        if args.only_missing and edge.get("relation_text"):
            continue
        process_edge(edge, nodes, args, llm)
        processed += 1
        if args.limit and processed >= args.limit:
            break

    save_graph(graph, out_path)
    print(f"完成处理 {processed} 条边，输出已写入 {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
