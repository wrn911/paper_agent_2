# utils/result_logger.py
import csv
import os
from datetime import datetime

class ResultLogger:
    """
    Logs the results of node processing to a CSV file.
    """
    def __init__(self, domain: str, task: str, log_dir: str = 'logs'):
        """
        Initializes the ResultLogger for a specific domain and task.

        Args:
            domain (str): The domain of the current run.
            task (str): The task of the current run.
            log_dir (str): The directory to store the log file. Defaults to 'logs'.
        """
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{domain}_{task}_{timestamp}.csv"
        self.filepath = os.path.join(log_dir, filename)
        self.fieldnames = ['node_id', 'status', 'processing_time_seconds', 'error_reason']
        self._initialize_csv()
        print(f"Result logger initialized. Results will be saved to: {self.filepath}")

    def _initialize_csv(self):
        """Creates the CSV file and writes the header."""
        with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, node_id: str, status: str, processing_time: float, error_reason: str = ''):
        """
        Appends a result row to the CSV file.

        Args:
            node_id (str): The ID of the node being processed.
            status (str): 'Success' or 'Failure'.
            processing_time (float): The time taken in seconds.
            error_reason (str): The error message if the status is 'Failure'.
        """
        with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({
                'node_id': node_id,
                'status': status,
                'processing_time_seconds': round(processing_time, 2),
                'error_reason': error_reason.replace('\n', ' ').replace('\r', '') # Clean up newlines for CSV
            })
