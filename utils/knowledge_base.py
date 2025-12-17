# utils/knowledge_base.py
import os
import threading


class KnowledgeBase:
    """
    领域+任务级别的知识库管理：
    - 主知识库：LivePaperBench/<domain>/<task>/kb.txt（存放精简后的共性知识）
    - 待审核区：LivePaperBench/<domain>/<task>/pending.txt（存放刚提取的经验，待汇总）
    线程安全，通过目录级锁防止并发写冲突。
    """

    _locks = {}
    _class_lock = threading.Lock()

    def __init__(self, domain: str, task: str):
        """
        初始化指定 domain+task 的知识库路径，并为该目录创建互斥锁。
        """
        base_kb_path = os.path.join("LivePaperBench", domain, task)
        os.makedirs(base_kb_path, exist_ok=True)  # Ensure base directory exists

        self.base_dir = base_kb_path
        self.kb_path = os.path.join(base_kb_path, "kb.txt")
        self.pending_path = os.path.join(base_kb_path, "pending.txt")

        # Create a lock for this specific task path if it doesn't exist
        with self._class_lock:
            if self.base_dir not in self._locks:
                self._locks[self.base_dir] = threading.Lock()
        self.lock = self._locks[self.base_dir]

        print(f"KnowledgeBase initialized for domain='{domain}', task='{task}'. Path: {self.kb_path}")

    def load(self) -> str:
        """
        读取主知识库内容（kb.txt），若不存在返回空字符串。
        """
        with self.lock:
            try:
                with open(self.kb_path, "r", encoding="utf-8") as f:
                    return f.read()
            except FileNotFoundError:
                return ""

    def load_pending(self) -> str:
        """
        读取待审核区内容（pending.txt），若不存在返回空字符串。
        """
        with self.lock:
            try:
                with open(self.pending_path, "r", encoding="utf-8") as f:
                    return f.read()
            except FileNotFoundError:
                return ""

    def save(self, content: str):
        """
        覆盖写入主知识库（kb.txt）。
        """
        with self.lock:
            os.makedirs(os.path.dirname(self.kb_path), exist_ok=True)
            with open(self.kb_path, "w", encoding="utf-8") as f:
                f.write(content)

    def save_pending(self, content: str):
        """
        覆盖写入待审核区（pending.txt）。
        """
        with self.lock:
            os.makedirs(os.path.dirname(self.pending_path), exist_ok=True)
            with open(self.pending_path, "w", encoding="utf-8") as f:
                f.write(content)

    def append_pending(self, content: str):
        """
        追加新提取的经验到待审核区，保持 task 级别隔离。
        """
        if not content.strip():
            return
        with self.lock:
            os.makedirs(os.path.dirname(self.pending_path), exist_ok=True)
            with open(self.pending_path, "a", encoding="utf-8") as f:
                if os.path.exists(self.pending_path) and os.path.getsize(self.pending_path) > 0:
                    f.write("\n")
                f.write(content)
