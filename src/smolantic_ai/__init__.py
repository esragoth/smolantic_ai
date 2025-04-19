"""
SmolAntic AI - A small but powerful AI agent framework
"""

from .code_agent import CodeAgent
from .multistep_agent import MultistepAgent
from .executors import PythonExecutor, LocalPythonExecutor, E2BExecutor, DockerExecutor, CodeExecutionResult
from .models import ActionStep, CodeResult, PlanningStep, Message, MessageRole, AgentMemory, MultistepResult, TaskStep, FinalAnswerStep
__all__ = [
    "CodeAgent",
    "BaseAgent",
    "CodeResult", 
    "MultistepAgent",
    "PythonExecutor",
    "LocalPythonExecutor",
    "E2BExecutor",
    "DockerExecutor",
    "CodeExecutionResult",
    "ActionStep",
    "CodeResult",
    "PlanningStep",
    "Message",
    "MessageRole",
    "AgentMemory",
    "MultistepResult",
    "TaskStep",
    "FinalAnswerStep",
]

__version__ = "0.1.0" 