#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logging Utility Module
Provides unified logging management functionality
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
    log_dir: Optional[str] = None,
    enable_file_logging: bool = False
) -> logging.Logger:
    """
    Set up logger
    
    Args:
        name: Logger name
        log_file: Log filename, if None uses default naming
        level: Log level
        log_dir: Log directory, if None uses default directory
        enable_file_logging: Whether to enable file logging
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Only create file when file logging is explicitly enabled
    if enable_file_logging:
        # Determine log directory
        if log_dir is None:
            log_dir = Path(__file__).parent.parent.parent / "logs"
        else:
            log_dir = Path(log_dir)
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine log filename
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{name}_{timestamp}.log"
        
        log_path = log_dir / log_file
        
        # File handler
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler (always add)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get configured logger
    
    Args:
        name: Logger name
        
    Returns:
        Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # If no handlers, use default settings
        return setup_logger(name)
    return logger

def log_function_call(func):
    """
    Decorator: Log function calls
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.info(f"Calling function: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Function {func.__name__} executed successfully")
            return result
        except Exception as e:
            logger.error(f"Function {func.__name__} execution failed: {e}")
            raise
    return wrapper

def log_execution_time(func):
    """
    Decorator: Log function execution time
    """
    import time
    
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"Function {func.__name__} execution time: {execution_time:.2f} seconds")
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.error(f"Function {func.__name__} execution failed (took {execution_time:.2f} seconds): {e}")
            raise
            
    return wrapper

class PipelineLogger:
    """Pipeline dedicated logger"""
    
    def __init__(self, pipeline_name: str, domain: str, stage: str, enable_file_logging: bool = True):
        self.pipeline_name = pipeline_name
        self.domain = domain 
        self.stage = stage
        self.logger_name = f"{pipeline_name}_{domain}_{stage}"
        self.logger = setup_logger(self.logger_name, enable_file_logging=enable_file_logging)
        
    def info(self, message: str):
        """Log info message"""
        self.logger.info(f"[{self.domain}|{self.stage}] {message}")
        
    def error(self, message: str):
        """Log error message"""
        self.logger.error(f"[{self.domain}|{self.stage}] {message}")
        
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(f"[{self.domain}|{self.stage}] {message}")
        
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(f"[{self.domain}|{self.stage}] {message}")
        
    def log_stage_start(self):
        """Log stage start"""
        self.info(f"=== Starting {self.stage} stage ===")
        
    def log_stage_end(self, success: bool = True):
        """Log stage end"""
        status = "succeeded" if success else "failed"
        self.info(f"=== {self.stage} stage {status} ===")
        
    def log_step(self, step_name: str, message: str):
        """Log step information"""
        self.info(f"[Step: {step_name}] {message}")

if __name__ == "__main__":
    # Test logging functionality
    logger = setup_logger("test")
    logger.info("This is an info log")
    logger.warning("This is a warning log")
    logger.error("This is an error log")
    
    # Test Pipeline logger
    pipeline_logger = PipelineLogger("test_pipeline", "medicine", "stage1")
    pipeline_logger.log_stage_start()
    pipeline_logger.log_step("Knowledge Graph Retrieval", "Starting medical domain knowledge graph retrieval")
    pipeline_logger.info("Testing Pipeline logging functionality")
    pipeline_logger.log_stage_end(True)
