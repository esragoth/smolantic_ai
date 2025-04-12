from typing import Any, Dict, List, Optional, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field
from pydantic_ai import Tool, RunContext
import json # Import json for formatting tool calls/outputs

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
    step_type: Literal["action"] = "action"

    def to_string_summary(self) -> str:
        """Return a concise string summary of the action step."""
        summary_parts = [f"  Type: {self.step_type}"]
        # Include thought/input that led to the action
        if self.input_messages:
            # Try to get the last relevant message (often Assistant thought or User input)
            last_input = self.input_messages[-1]
            role = "Thought" if last_input.role == MessageRole.ASSISTANT else "Input"
            summary_parts.append(f"  {role}: {last_input.content[:200]}{'...' if len(last_input.content) > 200 else ''}")

        if self.tool_calls:
            summary_parts.append("  Tool Calls:")
            for call in self.tool_calls:
                args_str = json.dumps(call.get('args', {}))
                summary_parts.append(f"    - {call.get('name', 'N/A')}({args_str[:100]}{'...' if len(args_str)>100 else ''})")
                
        # Prefer observations field if available and populated
        if self.observations and self.observations.strip():
             summary_parts.append(f"  Observations: {self.observations[:200]}{'...' if len(self.observations) > 200 else ''}")
        # Fallback to tool_outputs if observations is empty/None
        elif self.tool_outputs:
            summary_parts.append("  Tool Outputs:")
            for output in self.tool_outputs:
                output_content = str(output.get('output', 'N/A'))
                summary_parts.append(f"    - Tool {output.get('name','N/A')}: {output_content[:100]}{'...' if len(output_content) > 100 else ''}")
                
        if self.error:
            summary_parts.append(f"  Error: {self.error}")
            
        return "\n".join(summary_parts)

class PlanningStep(BaseModel):
    """Represents a planning step in the multi-step process."""
    facts_survey: str = Field(description="Survey of known facts and facts to discover")
    action_plan: str = Field(description="Step-by-step plan to solve the task")
    input_messages: List[Message] = Field(default_factory=list)
    output_messages: List[Message] = Field(default_factory=list)
    step_type: Literal["planning"] = "planning"

    def to_string_summary(self) -> str:
        """Return a concise string summary of the planning step."""
        summary_parts = [
            f"  Type: {self.step_type}",
            f"  Facts Survey: {self.facts_survey[:200]}{'...' if len(self.facts_survey) > 200 else ''}",
            f"  Action Plan: {self.action_plan[:300]}{'...' if len(self.action_plan) > 300 else ''}"
        ]
        return "\n".join(summary_parts)

class AgentMemory(BaseModel):
    """Tracks the agent's conversation history and action steps."""
    action_steps: List[Union[ActionStep, PlanningStep]] = Field(default_factory=list)
    state: Dict[str, Any] = Field(default_factory=dict)

    def reset(self):
        """Reset the agent's memory."""
        self.action_steps = []
        self.state = {}

    def add_step(self, step: Union[ActionStep, PlanningStep]):
        """Add a new action or planning step to memory."""
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

class TaskStep(BaseModel):
    """Represents a task execution step."""
    task: str
    status: str = Field(default="pending", description="Status: pending, in_progress, completed, failed")
    result: Optional[Any] = None
    error: Optional[str] = None

class FinalAnswerStep(BaseModel):
    """Represents the final answer step."""
    answer: str
    explanation: str

class MultistepResult(AgentResult):
    """Result model for multi-step operations."""
    planning: PlanningStep
    tasks: List[TaskStep] = Field(default_factory=list)
    final_answer: FinalAnswerStep
    memory: AgentMemory = Field(default_factory=AgentMemory) 