# main.py
# Entry point and parallel orchestrator for the application.
import argparse
import concurrent.futures
import logging
import threading
import os
import time
from datetime import datetime
from multiprocessing import Manager

from config.settings import Settings
from agent.paper_agent import PaperAgent
from utils.graph_manager import GraphManager
from utils.logger import setup_thread_logger
from utils.knowledge_base import KnowledgeBase
from config import prompts
from llm.client import get_llm_client


def _split_kb_sections(text: str):
    """
    根据标题拆分“Generalizable”和“Pending”两部分。
    - 若未识别到标题，则将原文整体视为通用部分。
    """
    general_lines, pending_lines = [], []
    current = None
    for line in (text or "").splitlines():
        lower = line.strip().lower()
        if lower.startswith("generalizable"):
            current = "general"
            continue
        if lower.startswith("pending"):
            current = "pending"
            continue
        if current == "general":
            general_lines.append(line)
        elif current == "pending":
            pending_lines.append(line)
    general = "\n".join(general_lines).strip()
    pending = "\n".join(pending_lines).strip()
    if not general and not pending:
        general = (text or "").strip()
    return general, pending


def agent_worker(gpu_id: int, nodes_to_process: list, mode: str, domain: str, task: str, timestamp: str, status_dict: dict):
    """线程池的工作入口：为指定 GPU 构建独立的 agent 与日志器"""
    thread_name = f"GPU-{gpu_id}"
    threading.current_thread().name = thread_name
    
    # --- Per-Thread Logger Setup ---
    settings = Settings(domain=domain, task=task)
    log_dir = settings.paths.get('logs', 'logs')
    log_file_name = f"run_{domain}_{task}_{timestamp}__{thread_name}.log"
    log_file_path = os.path.join(log_dir, log_file_name)
    logger = setup_thread_logger(thread_name, log_file_path)
    # --- End Logger Setup ---

    logger.info(f"Worker starting with {len(nodes_to_process)} nodes.")
    try:
        # 每个线程持有自己的 settings/agent/logger/status_dict
        agent = PaperAgent(settings=settings, gpu_id=gpu_id, logger=logger, status_dict=status_dict)
        agent.process_nodes(nodes_to_process, mode)
    except Exception as e:
        logger.error("An exception occurred in worker:", exc_info=e)
        raise


def status_monitor(status_dict: dict, gpu_ids: list, stop_event: threading.Event):
    """状态监控线程，定期刷新所有 GPU 的运行状态"""
    while not stop_event.is_set():
        # Clear console screen (works on Windows, Linux, macOS)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print(f"--- Agent Status Dashboard (Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
        for gpu_id in sorted(gpu_ids):
            status = status_dict.get(gpu_id, "Initializing...")
            print(f"[GPU-{gpu_id}] Status: {status}")
        
        # Wait for 15 seconds before the next update, but check for the stop event every second
        for _ in range(15):
            if stop_event.is_set():
                break
            time.sleep(1)


def summarize_global_kb(domain: str, task: str, settings: Settings, old_kb_snapshot: str):
    """批次结束后：读取主库+待审核，压缩去重，只将共性知识写回主库"""
    kb = KnowledgeBase(domain=domain, task=task)
    main_kb = kb.load()
    pending_kb = kb.load_pending()
    if not pending_kb.strip() and main_kb.strip() == old_kb_snapshot.strip():
        logging.info("Knowledge base unchanged and no pending items; skip summarization.")
        return

    llm_cfg = settings.get('llm', {}) or {}
    api_keys = settings.get('api_keys', {}) or {}
    base_url = llm_cfg.get('openai_base_url')
    model = get_llm_client(
        api_key=api_keys.get('openai'),
        base_url=base_url,
        model_name=os.environ.get("SUMMARY_MODEL", "gpt-5.2"),
        temperature=float(os.environ.get("SUMMARY_TEMPERATURE", "0.2")),
    )
    prompt = prompts.BATCH_KB_SUMMARY_PROMPT.format(old_kb=main_kb, new_kb=pending_kb)
    logging.info("Summarizing global knowledge base...")
    resp = model.invoke(prompt)
    merged = resp.content if hasattr(resp, "content") else str(resp)
    general, pending = _split_kb_sections(merged)
    kb.save(general)
    kb.save_pending(pending)
    logging.info("Knowledge base summarized and saved.")


def main():
    """主入口：解析参数、分发节点、并行运行 agent"""
    parser = argparse.ArgumentParser(description='Graph Scientist Agent - Parallel Runner')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help='Mode to run the agents in')
    parser.add_argument('--domain', type=str, required=True, help='The domain for the task')
    parser.add_argument('--task', type=str, required=True, help='The specific task to run')
    parser.add_argument('--batch-size', type=int, default=None, help='Training mode concurrency (defaults to config or GPU count)')
    args = parser.parse_args()

    # Configure a simple logger for the main orchestrator thread's console output
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [Orchestrator] - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    settings = Settings(domain=args.domain, task=args.task)
    gpu_ids = settings.get('gpu_ids', [])

    if not gpu_ids:
        logging.error("No GPU IDs configured in config/config.yaml under 'gpu_ids'.")
        return

    graph_manager = GraphManager(settings.domain_config['graph_path'])
    all_nodes = graph_manager.get_nodes(node_type=args.mode, task=args.task)

    if not all_nodes:
        logging.info("No nodes found for the specified mode and task. Exiting.")
        return

    # 计算并发度：train 模式按 batch size；test 模式按 GPU 数
    if args.mode == 'train':
        default_batch = settings.get('train_batch_size', len(gpu_ids))
        batch_size = args.batch_size or default_batch
    else:
        batch_size = len(gpu_ids)
    batch_size = max(1, min(batch_size, len(gpu_ids)))
    active_gpus = gpu_ids[:batch_size]

    logging.info(f"Configured {len(all_nodes)} nodes. Batch agent count={batch_size}, GPUs={active_gpus}")
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    kb = KnowledgeBase(domain=args.domain, task=args.task) if args.mode == 'train' else None
    kb_snapshot = kb.load() if kb else ""

    with Manager() as manager:
        status_dict = manager.dict()
        stop_event = threading.Event()

        # Start the monitoring thread
        monitor = threading.Thread(target=status_monitor, args=(status_dict, active_gpus, stop_event), daemon=True)
        monitor.start()

        # 分批执行：每批最多 batch_size 个 GPU，每个 GPU 处理 1 篇论文
        remaining = list(all_nodes)
        batch_idx = 0
        while remaining:
            batch_idx += 1
            current_gpu_list = active_gpus[: min(len(active_gpus), len(remaining), batch_size)]
            nodes_this_batch = [remaining.pop(0) for _ in range(len(current_gpu_list))]
            node_distribution = [[n] for n in nodes_this_batch]

            logging.info(f"Starting batch {batch_idx}: nodes={nodes_this_batch}, gpus={current_gpu_list}")

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(current_gpu_list)) as executor:
                future_to_gpu = {
                    executor.submit(agent_worker, gpu_id, nodes, args.mode, args.domain, args.task, run_timestamp, status_dict): gpu_id
                    for gpu_id, nodes in zip(current_gpu_list, node_distribution) if nodes
                }

                for future in concurrent.futures.as_completed(future_to_gpu):
                    gpu = future_to_gpu[future]
                    try:
                        future.result()
                        logging.info(f"Worker for GPU {gpu} finished successfully.")
                    except Exception as exc:
                        logging.error(f"Worker for GPU {gpu} generated an exception: {exc}", exc_info=False)

            # 每批结束后压缩一次知识库
            if args.mode == 'train' and kb:
                try:
                    summarize_global_kb(args.domain, args.task, settings, kb_snapshot)
                    kb_snapshot = kb.load()
                except Exception as e:
                    logging.error(f"Failed to summarize knowledge base after batch {batch_idx}: {e}", exc_info=True)

        # Signal the monitor to stop and wait for it to finish
        stop_event.set()
        monitor.join(timeout=5)

    logging.info("All tasks completed.")


if __name__ == '__main__':
    main()
