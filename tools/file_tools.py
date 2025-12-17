# tools/file_tools.py
# 文件读写相关工具，包含对路径的严格限制。

import os
import re
import subprocess
import tempfile
from typing import List

from config.settings import Settings


def set_allowed_paths(paths: List[str]):
    """
    兼容占位：当前基于显式文件列表 + 目录检查，不需要预设。
    """
    return None


def _is_allowed_path(file_path: str, allowed_files: List[str] = None) -> bool:
    """
    安全检查：优先匹配显式允许的文件列表；否则检查是否位于 code/config 目录。
    """
    abs_path = os.path.abspath(file_path)
    if allowed_files:
        if any(abs_path == os.path.abspath(p) for p in allowed_files):
            return True

    settings = Settings()
    domain_config = settings.domain_config or {}
    allowed_dirs = [
        os.path.abspath(domain_config.get("code_path", "")),
        os.path.abspath(domain_config.get("hyperparameter_path", "")),
    ]
    for base in allowed_dirs:
        if not base:
            continue
        try:
            common = os.path.commonpath([abs_path, base])
            if common == base:
                return True
        except ValueError:
            continue
    return False


def write_file(file_path: str, content: str, allowed_files: List[str] = None) -> str:
    """
    覆盖写文件，限制只能写入 code_path / hyperparameter_path 下；可选显式允许列表。
    """
    if not _is_allowed_path(file_path, allowed_files):
        return f"Error: Access denied. Writing to {file_path} is not permitted."

    try:
        abs_file_path = os.path.abspath(file_path)
        os.makedirs(os.path.dirname(abs_file_path), exist_ok=True)
        with open(abs_file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing to file {file_path}: {e}"


def read_file(file_path: str) -> str:
    """
    读取文件（供内部使用，无路径限制）。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except Exception as e:
        return f"Error reading file {file_path}: {e}"


def restricted_read(file_path: str, allowed_files: List[str] = None) -> str:
    """
    读取允许路径下的文件（code/config 目录），超过 4000 字符会截断。
    """
    if not _is_allowed_path(file_path, allowed_files):
        return f"Error: Access denied. Reading {file_path} is not permitted."
    content = read_file(file_path)
    if isinstance(content, str) and len(content) > 4000:
        return content[:2000] + "\n...<truncated>...\n" + content[-2000:]
    return content


def restricted_write(file_path: str, content: str, allowed_files: List[str] = None) -> str:
    """
    写入允许路径下的文件（覆盖写）。
    """
    return write_file(file_path, content, allowed_files) if _is_allowed_path(file_path, allowed_files) else f"Error: Access denied. Writing to {file_path} is not permitted."


def apply_restricted_patch(unified_diff: str, allowed_files: List[str] = None) -> str:
    """
    应用补丁，补丁中的文件必须位于允许路径（code/config 目录）内。
    """
    if not unified_diff.strip():
        return "Error: empty patch"

    if re.search(r"^\+\+\+\s+/(?!dev/null)", unified_diff, re.M):
        return "Error: absolute paths are not allowed in patch"

    target_paths = []
    for line in unified_diff.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            parts = line.split()
            if len(parts) < 2:
                continue
            path = parts[1]
            if path.endswith("dev/null"):
                continue
            path = path.lstrip("ab/")
            target_paths.append(os.path.abspath(path))

    if target_paths and not all(_is_allowed_path(p, allowed_files) for p in target_paths):
        return "Error: patch touches files outside allowed list"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tf:
            tf.write(unified_diff)
            tmp_path = tf.name

        cmd = ["patch", "-p0", "-i", tmp_path]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            return f"Error: patch failed ({proc.returncode})\n{out}"
        return "OK: patch applied\n" + out
    except FileNotFoundError:
        return "Error: patch command not available in environment"
    except Exception as e:
        return f"Error applying patch: {e}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
