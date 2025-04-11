from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
from pydantic_ai import Tool, RunContext

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class Message(BaseModel):
    """Represents a message in the conversation."""
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class ActionStep(BaseModel):
    """Represents a single action step taken by the agent."""
    input_messages: List[Message]
    output_messages: List[Message]
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    tool_outputs: List[Dict[str, Any]] = Field(default_factory=list)
    observations: Optional[str] = None
    error: Optional[str] = None

class AgentMemory(BaseModel):
    """Tracks the agent's conversation history and action steps."""
    action_steps: List[ActionStep] = Field(default_factory=list)
    state: Dict[str, Any] = Field(default_factory=dict)

    def reset(self):
        """Reset the agent's memory."""
        self.action_steps = []
        self.state = {}

    def add_step(self, step: ActionStep):
        """Add a new action step to memory."""
        self.action_steps.append(step)

    def update_state(self, key: str, value: Any):
        """Update the agent's state."""
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a value from the agent's state."""
        return self.state.get(key, default)

class ToolResult(BaseModel):
    """Represents the result of a tool call."""
    name: str
    arguments: Dict[str, Any]
    result: Any
    error: Optional[str] = None

class AgentResult(BaseModel):
    """Base class for agent results."""
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ToolCallingResult(AgentResult):
    """Result of a tool calling operation."""
    tool_results: List[ToolResult] = Field(default_factory=list)
    final_answer: Optional[str] = None

class CodeExecutionResult(AgentResult):
    """Result of a code execution operation."""
    code: str
    output: Any
    execution_logs: str
    is_final_answer: bool = False 