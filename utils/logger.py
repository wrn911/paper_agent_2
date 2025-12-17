# utils/logger.py
# This file provides utilities for setting up thread-specific loggers.

import logging
import os
import csv
import threading
from datetime import datetime
from typing import Dict, Any
from config.settings import Settings

# A lock for the token usage CSV file to ensure thread-safe writes
_token_log_lock = threading.Lock()

def setup_thread_logger(name: str, log_file_path: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Sets up and returns an independent logger for a specific thread.
    This logger writes to its own file and does not propagate to the root logger.
    
    Args:
        name (str): The unique name for the logger (e.g., 'GPU-0').
        log_file_path (str): The full path to the log file for this logger.
        log_level (int): The logging level.
        
    Returns:
        logging.Logger: A configured, independent logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent messages from being passed to the root logger
    logger.propagate = False

    # Clear existing handlers to prevent duplicate logging on re-configuration
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a new formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # File handler for this specific thread
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    f_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    f_handler.setLevel(log_level)
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)
    
    return logger

def log_token_usage(model_name: str, usage_stats: Dict[str, Any]):
    """Appends token usage statistics to a CSV file in a thread-safe manner."""
    settings = Settings()
    log_dir = settings.paths.get('logs', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, 'token_usage.csv')

    write_header = not os.path.exists(file_path)

    with _token_log_lock:
        try:
            with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['timestamp', 'model_name', 'prompt_tokens', 'completion_tokens', 'total_tokens']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if write_header:
                    writer.writeheader()
                
                writer.writerow({
                    'timestamp': datetime.now().isoformat(),
                    'model_name': model_name,
                    'prompt_tokens': usage_stats.get('prompt_tokens', 0),
                    'completion_tokens': usage_stats.get('completion_tokens', 0),
                    'total_tokens': usage_stats.get('total_tokens', 0)
                })
        except IOError as e:
            # Use the root logger for this error as it's a shared utility
            logging.getLogger().error(f"Failed to write to token usage log: {e}")
