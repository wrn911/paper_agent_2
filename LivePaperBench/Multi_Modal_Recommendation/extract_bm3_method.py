#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 BM3.md 提取方法部分，更新 graph.json 中 BM3 节点的 method_md 字段。

默认按章节标题提取：
- 起始：以 “# 3” 开头的章节（例：# 3BOOTSTRAPPED MULTI-MODAL MODEL）
- 结束：遇到下一个以 “# 4” 开头的章节或文件结束

使用示例：
python extract_bm3_method.py --graph graph.json --paper BM3.md --output graph.json
"""
import argparse
import json
import pathlib
import re
from typing import Dict, Any, Optional


def load_text(path: pathlib.Path) -> str:
    """读取 Markdown 原文。"""
    return path.read_text(encoding="utf-8")


def extract_method(md_text: str) -> str:
    """
    截取 BM3 方法章节（第3节），从 “# 3” 到下一个 “# 4” 之间。
    若未找到起始，则返回全文以免丢失信息。
    """
    # 查找起始位置
    start_match = re.search(r"^#\s*3[^\n]*$", md_text, re.M)
    if not start_match:
        return md_text.strip()
    start_idx = start_match.start()
    # 查找结束位置
    end_match = re.search(r"^#\s*4[^\n]*$", md_text[start_idx:], re.M)
    if end_match:
        end_idx = start_idx + end_match.start()
        return md_text[start_idx:end_idx].strip()
    return md_text[start_idx:].strip()


def load_graph(path: pathlib.Path) -> Dict[str, Any]:
    """加载 graph.json。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_graph(data: Dict[str, Any], path: pathlib.Path) -> None:
    """写回 graph.json（先写临时文件再替换）。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def find_bm3_node(graph: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    在 nodes 中查找 BM3 对应节点。
    优先按 method_name 精确匹配 “BM3”，否则按 id 前缀 “BM3” 模糊匹配。
    """
    for n in graph.get("nodes", []):
        if n.get("method_name") == "BM3":
            return n
    for n in graph.get("nodes", []):
        if str(n.get("id", "")).startswith("BM3"):
            return n
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="提取 BM3 方法并写入 graph.json 的 method_md")
    parser.add_argument("--graph", type=pathlib.Path, default=pathlib.Path("graph.json"), help="输入 graph.json 路径")
    parser.add_argument("--paper", type=pathlib.Path, default=pathlib.Path("BM3.md"), help="BM3 论文 Markdown 路径")
    parser.add_argument("--output", type=pathlib.Path, default=None, help="输出文件路径，默认覆盖 graph.json")
    args = parser.parse_args()

    md_text = load_text(args.paper)
    method_section = extract_method(md_text)

    graph = load_graph(args.graph)
    node = find_bm3_node(graph)
    if not node:
        raise SystemExit("未找到 BM3 节点，请检查 graph.json")

    node["method_md"] = method_section

    out_path = args.output or args.graph
    save_graph(graph, out_path)
    print(f"已更新 BM3 节点的 method_md，输出写入 {out_path}")


if __name__ == "__main__":
    main()
