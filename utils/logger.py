"""
Logging utility for the RAG system.

Provides structured logging with colors and consistent formatting.
"""
import logging
import sys
from typing import Optional


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a configured logger with colored output.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Console handler with formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Format: timestamp - module - level - message
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger


# Pre-configured loggers for main modules
def get_db_logger():
    return setup_logger("RAG.Database")

def get_handler_logger():
    return setup_logger("RAG.Handler")

def get_pipeline_logger():
    return setup_logger("RAG.Pipeline")

def get_api_logger():
    return setup_logger("RAG.API")
