import logging
import json
from typing import Any, Dict, Optional
from datetime import datetime
from pydantic import BaseModel

class AgentLogger:
    """Custom logger for agent operations with pretty printing support."""
    
    def __init__(self, name: str = "smolantic_ai"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create console handler with pretty formatting
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(message)s')  # Only show the message
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def log_step(self, step_type: str, step_data: str, indent: int = 0) -> None:
        """Log a step with pretty printing."""
        indent_str = "  " * indent
        # Add a single separator line before each new step
        separator = "─" * 80
        message = f"\n{separator}\n{indent_str}{step_data}"
        self.logger.info(message)
        # Force immediate output by flushing the stream
        for handler in self.logger.handlers:
            if hasattr(handler.stream, 'flush'):
                handler.stream.flush()
    
    def log_planning(self, plan: Dict[str, Any]) -> None:
        """Log planning steps."""
        self.log_step("Planning", json.dumps(plan, indent=2), indent=0)
    
    def log_action(self, action: Dict[str, Any]) -> None:
        """Log action steps."""
        self.log_step("Action", json.dumps(action, indent=2), indent=1)
    
    def log_result(self, result: Dict[str, Any]) -> None:
        """Log result steps."""
        self.log_step("Result", json.dumps(result, indent=2), indent=2)
    
    def error(self, message: str, exc_info: bool = True) -> None:
        """Log error messages."""
        self.log_step("Error", message, indent=1)
        if exc_info:
            import traceback
            self.log_step("Error", traceback.format_exc(), indent=2)
    
    def log_error(self, error: Exception) -> None:
        """Log error steps."""
        self.error(f"Error occurred: {str(error)}", exc_info=True)

    def debug(self, message: str, *args, **kwargs) -> None:
        """Log messages with severity 'DEBUG'."""
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """Log messages with severity 'INFO'."""
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """Log messages with severity 'WARNING'."""
        self.logger.warning(message, *args, **kwargs)

# Create a default logger instance
default_logger = AgentLogger()

def get_logger(name: Optional[str] = None) -> AgentLogger:
    """Get a logger instance."""
    if name:
        return AgentLogger(name)
    return default_logger 