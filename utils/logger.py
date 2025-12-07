"""
Logging Utilities for Deepfake Detection Project

This module provides centralized logging configuration and utilities
for consistent logging across the entire project.

Author: Deepfake Detection Team
Date: 2024
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import os

def setup_logger(name: str, 
                level: int = logging.INFO,
                log_file: Optional[str] = None,
                console_output: bool = True) -> logging.Logger:
    """
    Setup a logger with consistent formatting and output options
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional path to log file
        console_output: Whether to output to console
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def setup_project_logging(project_root: str = None,
                         log_level: int = logging.INFO) -> dict:
    """
    Setup comprehensive logging for the entire project
    
    Args:
        project_root: Root directory of the project
        log_level: Logging level for all loggers
        
    Returns:
        Dictionary of configured loggers
    """
    if project_root is None:
        project_root = os.getcwd()
    
    # Create logs directory
    logs_dir = Path(project_root) / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for log files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Configure different loggers for different components
    loggers = {}
    
    # Main application logger
    loggers['main'] = setup_logger(
        'deepfake_detection',
        level=log_level,
        log_file=str(logs_dir / f'main_{timestamp}.log'),
        console_output=True
    )
    
    # Data preprocessing logger
    loggers['preprocessing'] = setup_logger(
        'data_preprocessing',
        level=log_level,
        log_file=str(logs_dir / f'preprocessing_{timestamp}.log'),
        console_output=True
    )
    
    # Model training logger
    loggers['training'] = setup_logger(
        'model_training',
        level=log_level,
        log_file=str(logs_dir / f'training_{timestamp}.log'),
        console_output=True
    )
    
    # Web application logger
    loggers['webapp'] = setup_logger(
        'webapp',
        level=log_level,
        log_file=str(logs_dir / f'webapp_{timestamp}.log'),
        console_output=True
    )
    
    # Evaluation logger
    loggers['evaluation'] = setup_logger(
        'evaluation',
        level=log_level,
        log_file=str(logs_dir / f'evaluation_{timestamp}.log'),
        console_output=True
    )
    
    # Error logger (for critical errors only)
    loggers['error'] = setup_logger(
        'error',
        level=logging.ERROR,
        log_file=str(logs_dir / f'errors_{timestamp}.log'),
        console_output=True
    )
    
    return loggers

class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class
    """
    
    def __init__(self, logger_name: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if logger_name:
            self.logger = logging.getLogger(logger_name)
        else:
            self.logger = logging.getLogger(self.__class__.__name__)
    
    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def log_debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def log_critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)

def log_function_call(func):
    """
    Decorator to log function calls with parameters and return values
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        
        # Log function call
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Log successful completion
            logger.debug(f"{func.__name__} completed successfully")
            
            return result
            
        except Exception as e:
            # Log error
            logger.error(f"{func.__name__} failed with error: {str(e)}")
            raise
    
    return wrapper

def log_execution_time(func):
    """
    Decorator to log function execution time
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    import time
    
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.4f} seconds with error: {str(e)}")
            raise
    
    return wrapper

def create_log_summary(log_file: str) -> dict:
    """
    Create a summary of log file contents
    
    Args:
        log_file: Path to log file
        
    Returns:
        Dictionary with log summary statistics
    """
    summary = {
        'total_lines': 0,
        'info_count': 0,
        'warning_count': 0,
        'error_count': 0,
        'debug_count': 0,
        'critical_count': 0,
        'first_timestamp': None,
        'last_timestamp': None
    }
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                summary['total_lines'] += 1
                
                # Count log levels
                if ' - INFO - ' in line:
                    summary['info_count'] += 1
                elif ' - WARNING - ' in line:
                    summary['warning_count'] += 1
                elif ' - ERROR - ' in line:
                    summary['error_count'] += 1
                elif ' - DEBUG - ' in line:
                    summary['debug_count'] += 1
                elif ' - CRITICAL - ' in line:
                    summary['critical_count'] += 1
                
                # Extract timestamps
                try:
                    timestamp = line.split(' - ')[0]
                    if summary['first_timestamp'] is None:
                        summary['first_timestamp'] = timestamp
                    summary['last_timestamp'] = timestamp
                except:
                    pass
    
    except FileNotFoundError:
        summary['error'] = 'Log file not found'
    
    return summary

def cleanup_old_logs(logs_dir: str, days_to_keep: int = 30):
    """
    Clean up old log files
    
    Args:
        logs_dir: Directory containing log files
        days_to_keep: Number of days to keep log files
    """
    import time
    from pathlib import Path
    
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        return
    
    current_time = time.time()
    cutoff_time = current_time - (days_to_keep * 24 * 60 * 60)
    
    deleted_count = 0
    
    for log_file in logs_path.glob('*.log'):
        if log_file.stat().st_mtime < cutoff_time:
            try:
                log_file.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"Failed to delete {log_file}: {e}")
    
    if deleted_count > 0:
        print(f"Deleted {deleted_count} old log files")

# Example usage
if __name__ == "__main__":
    # Setup project logging
    loggers = setup_project_logging()
    
    # Test different loggers
    loggers['main'].info("Main application started")
    loggers['preprocessing'].info("Data preprocessing completed")
    loggers['training'].info("Model training started")
    loggers['webapp'].info("Web application initialized")
    
    # Test error logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        loggers['error'].error(f"An error occurred: {e}")
    
    # Create log summary
    logs_dir = Path('logs')
    if logs_dir.exists():
        for log_file in logs_dir.glob('*.log'):
            summary = create_log_summary(str(log_file))
            print(f"Summary for {log_file.name}: {summary}")
