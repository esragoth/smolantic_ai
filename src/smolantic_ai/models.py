from typing import Any, Dict, List, Optional, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field
from pydantic_ai import Tool, RunContext, Agent
from pydantic_ai.messages import ToolCallPart, TextPart
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

class NodeType(str, Enum):
    USER_PROMPT = "user_prompt"
    CALL_TOOLS = "call_tools"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    PLANNING = "planning"
    END = "end"

class Node(BaseModel):
    """Base class for all nodes in the agent's process."""
    input_messages: List[Message] = Field(default_factory=list)
    output_messages: List[Message] = Field(default_factory=list)
    node_type: NodeType
    error: Optional[str] = None

    def to_string_summary(self) -> str:
        """Return a concise string summary of the node's contents."""
        if not self:
            return "None"
            
        def truncate(text: str, max_length: int = 100) -> str:
            """Truncate text to max_length and add ellipsis if needed."""
            if len(text) > max_length:
                return text[:max_length] + "..."
            return text

        # Start with node type
        summary = [f"Node Type: {self.node_type.value}"]
        
        # Add error if present
        if self.error:
            summary.append(f"Error: {self.error}")
        
        # Add input messages summary
        if self.input_messages:
            summary.append("Input Messages:")
            for msg in self.input_messages:
                summary.append(f"  {msg.role}: {truncate(msg.content)}")
        
        # Add output messages summary
        if self.output_messages:
            summary.append("Output Messages:")
            for msg in self.output_messages:
                summary.append(f"  {msg.role}: {truncate(msg.content)}")
        
        return "\n".join(summary)

class UserPromptNode(Node):
    """Represents a user prompt node."""
    user_prompt: str = Field(description="The user prompt")

    def to_string_summary(self) -> str:
        """Return a concise string summary of the user prompt node."""
        base_summary = super().to_string_summary()
        return f"{base_summary}\nUser Prompt: {self.user_prompt[:100]}{'...' if len(self.user_prompt) > 100 else ''}"

class CallToolsNode(Node):
    """Represents a call tools node."""
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)

    def to_string_summary(self) -> str:
        """Return a concise string summary of the tool calls node."""
        base_summary = super().to_string_summary()
        if not self.tool_calls:
            return base_summary
            
        tool_summary = ["Tool Calls:"]
        for tool_call in self.tool_calls:
            name = tool_call.get('name', 'Unknown')
            args = tool_call.get('args', {})
            args_str = str(args)
            tool_summary.append(f"  {name}: {args_str[:100]}{'...' if len(args_str) > 100 else ''}")
            
        return f"{base_summary}\n" + "\n".join(tool_summary)

class ModelRequestNode(Node):
    """Represents a model request node."""
    system_prompt: Optional[str] = Field(description="The system prompt part of the request")
    user_prompt: Optional[str] = Field(description="The user prompt part of the request")
    tool_return: Optional[List[Dict[str, Any]]] = Field(description="The tool return part of the request")
    retry_prompt: Optional[str] = Field(description="The retry prompt part of the request")

    def to_string_summary(self) -> str:
        """Return a concise string summary of the model request node."""
        base_summary = super().to_string_summary()
        sections = []
        
        if self.user_prompt:
            sections.append(f"User Prompt: {self.user_prompt[:100]}{'...' if len(self.user_prompt) > 100 else ''}")
        if self.system_prompt:
            sections.append(f"System Prompt: {self.system_prompt[:100]}{'...' if len(self.system_prompt) > 100 else ''}")
        if self.tool_return:
            sections.append("Tool Returns:")
            for tool_call in self.tool_return:
                tool_name = tool_call.get('tool_name', 'unknown')
                tool_content = tool_call.get('content', '')
                sections.append(f"  {tool_name}: {tool_content[:100]}{'...' if len(tool_content) > 100 else ''}")
        if self.retry_prompt:
            sections.append(f"Retry Prompt: {self.retry_prompt[:100]}{'...' if len(self.retry_prompt) > 100 else ''}")
            
        return f"{base_summary}\n" + "\n".join(sections) if sections else base_summary

class ModelResponseNode(Node):
    """Represents a model response node."""
    model_response: str = Field(description="The model response")

    def to_string_summary(self) -> str:
        """Return a concise string summary of the model response node."""
        base_summary = super().to_string_summary()
        return f"{base_summary}\nModel Response: {self.model_response[:100]}{'...' if len(self.model_response) > 100 else ''}"

class PlanningNode(Node):
    """Represents a planning node."""
    facts_survey: str = Field(description="Survey of known facts and facts to discover")
    action_plan: str = Field(description="Step-by-step plan to solve the task")

    def to_string_summary(self) -> str:
        """Return a concise string summary of the planning node."""
        base_summary = super().to_string_summary()
        return f"{base_summary}\nFacts Survey: {self.facts_survey[:100]}{'...' if len(self.facts_survey) > 100 else ''}\nAction Plan: {self.action_plan[:100]}{'...' if len(self.action_plan) > 100 else ''}"

class EndNode(Node):
    """Represents an end node."""
    end_reason: str = Field(description="The reason for the end of the process")

    def to_string_summary(self) -> str:
        """Return a concise string summary of the end node."""
        base_summary = super().to_string_summary()
        return f"{base_summary}\nEnd Reason: {self.end_reason}"

class Step(BaseModel):
    """Base class for all step types in the agent's process."""
    input_messages: List[Message] = Field(default_factory=list)
    output_messages: List[Message] = Field(default_factory=list)
    step_type: Literal["action", "planning"]
    step_number: int = Field(description="The step number of the step")
    error: Optional[str] = None

    def to_string_summary(self) -> str:
        """Base implementation of string summary. Should be overridden by subclasses."""
        summary_parts = [f"  Type: {self.step_type}"]
        if self.error:
            summary_parts.append(f"  Error: {self.error}")
        return "\n".join(summary_parts)

class ActionStep(Step):
    """Represents a single action step taken by the agent."""
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    tool_outputs: List[Dict[str, Any]] = Field(default_factory=list)
    observations: Optional[str] = None
    step_type: Literal["action"] = "action"

    def to_string_summary(self) -> str:
        """Return a concise string summary of the action step."""
        summary_parts = super().to_string_summary().split("\n")
        
        # Include thought/input that led to the action
        if self.input_messages:
            last_input = self.input_messages[-1]
            role = "Thought" if last_input.role == MessageRole.ASSISTANT else "Input"
            summary_parts.append(f"  {role}: {last_input.content[:200]}{'...' if len(last_input.content) > 200 else ''}")

        if self.tool_calls:
            summary_parts.append("  Tool Calls:")
            for call in self.tool_calls:
                args_str = json.dumps(call.get('args', {}))
                summary_parts.append(f"    - {call.get('name', 'N/A')}({args_str[:100]}{'...' if len(args_str)>100 else ''})")
                
        if self.observations and self.observations.strip():
            summary_parts.append(f"  Observations: {self.observations[:200]}{'...' if len(self.observations) > 200 else ''}")
        elif self.tool_outputs:
            summary_parts.append("  Tool Outputs:")
            for output in self.tool_outputs:
                output_content = str(output.get('output', 'N/A'))
                summary_parts.append(f"    - Tool {output.get('name','N/A')}: {output_content[:100]}{'...' if len(output_content) > 100 else ''}")
                
        return "\n".join(summary_parts)

class PlanningStep(Step):
    """Represents a planning step in the multi-step process."""
    facts_survey: str = Field(description="Survey of known facts and facts to discover")
    action_plan: str = Field(description="Step-by-step plan to solve the task")
    step_type: Literal["planning"] = "planning"

    def to_string_summary(self) -> str:
        """Return a concise string summary of the planning step."""
        summary_parts = super().to_string_summary().split("\n")
        summary_parts.extend([
            f"  Facts Survey: {self.facts_survey[:200]}{'...' if len(self.facts_survey) > 200 else ''}",
            f"  Action Plan: {self.action_plan[:300]}{'...' if len(self.action_plan) > 300 else ''}"
        ])
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

class CodeResult(BaseModel):
    """Result of code execution."""
    code: str = Field(..., description="The executed code")
    result: Optional[Any] = Field(None, description="The result of the code execution")
    explanation: str = Field(..., description="Explanation of the code and result")
    execution_logs: str = Field(..., description="Logs from the code execution")
    answer: Optional[Any] = Field(None, description="The final answer from the code execution")

    def __str__(self) -> str:
        return f"Result: {self.result}"

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

def reconstruct_node(pydantic_node: Any) -> Optional[Node]:
    """Reconstruct a pydantic node into a smolantic_ai Node class.
    
    Args:
        pydantic_node: The pydantic node to reconstruct
        
    Returns:
        Optional[Node]: The reconstructed smolantic_ai Node, or None if reconstruction fails
    """
    try:
        # Extract common fields
        input_messages = []
        output_messages = []
        
        # Handle input messages
        if hasattr(pydantic_node, 'input_messages'):
            for msg in pydantic_node.input_messages:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    input_messages.append(Message(
                        role=MessageRole(msg.role),
                        content=str(msg.content)
                    ))
        
        # Handle output messages
        if hasattr(pydantic_node, 'output_messages'):
            for msg in pydantic_node.output_messages:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    output_messages.append(Message(
                        role=MessageRole(msg.role),
                        content=str(msg.content)
                    ))
        
        # Determine node type and reconstruct accordingly
        if Agent.is_user_prompt_node(pydantic_node):
            prompt_content = "(Prompt content not readily extractable)"
            if hasattr(pydantic_node, 'user_prompt'):
                if isinstance(pydantic_node.user_prompt, str):
                    prompt_content = pydantic_node.user_prompt
                elif isinstance(pydantic_node.user_prompt, (list, tuple)):
                    # Handle sequence of UserContent
                    content_parts = []
                    for content in pydantic_node.user_prompt:
                        if hasattr(content, 'content'):
                            content_parts.append(str(content.content))
                    prompt_content = "\n".join(content_parts)
                elif hasattr(pydantic_node.user_prompt, 'content'):
                    prompt_content = str(pydantic_node.user_prompt.content)
                else:
                    prompt_content = str(pydantic_node.user_prompt)
            
            return UserPromptNode(
                input_messages=input_messages,
                output_messages=output_messages,
                node_type=NodeType.USER_PROMPT,
                user_prompt=prompt_content
            )
        elif Agent.is_call_tools_node(pydantic_node):
            tool_calls = []
            if hasattr(pydantic_node, 'model_response') and hasattr(pydantic_node.model_response, 'parts'):
                for part in pydantic_node.model_response.parts:
                    if isinstance(part, ToolCallPart):
                        tool_calls.append({
                            'name': part.tool_name,
                            'args': part.args,
                            'tool_call_id': part.tool_call_id
                        })
            
            return CallToolsNode(
                input_messages=input_messages,
                output_messages=output_messages,
                node_type=NodeType.CALL_TOOLS,
                tool_calls=tool_calls
            )
        elif Agent.is_model_request_node(pydantic_node):
            system_prompt = None
            user_prompt = None
            tool_return = None
            retry_prompt = None

            if hasattr(pydantic_node, 'request') and hasattr(pydantic_node.request, 'parts'):
                for part in pydantic_node.request.parts:
                    if hasattr(part, 'part_kind'):
                        if part.part_kind == 'system-prompt':
                            system_prompt = str(part.content)
                        elif part.part_kind == 'user-prompt':
                            user_prompt = str(part.content)
                        elif part.part_kind == 'tool-return':
                            # Convert tool return content to a proper dictionary
                            if tool_return is None:
                                tool_return = []
                            tool_return.append({
                                'tool_name': getattr(part, 'tool_name', 'unknown'),
                                'content': str(part.content),
                                'tool_call_id': getattr(part, 'tool_call_id', None)
                            })
                        elif part.part_kind == 'retry-prompt':
                            retry_prompt = str(part.content)
                return ModelRequestNode(
                    input_messages=input_messages,
                    output_messages=output_messages,
                    node_type=NodeType.MODEL_REQUEST,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    tool_return=tool_return,
                    retry_prompt=retry_prompt
                )
        elif isinstance(pydantic_node, PlanningStep):
            facts_survey = ""
            action_plan = ""
            step_number = 0
            
            if hasattr(pydantic_node, 'facts_survey'):
                facts_survey = str(pydantic_node.facts_survey)
            if hasattr(pydantic_node, 'action_plan'):
                action_plan = str(pydantic_node.action_plan)
            if hasattr(pydantic_node, 'step_number'):
                step_number = int(pydantic_node.step_number)
            
            return PlanningNode(
                input_messages=input_messages,
                output_messages=output_messages,
                node_type=NodeType.PLANNING,
                facts_survey=facts_survey,
                action_plan=action_plan,
            )
        elif Agent.is_end_node(pydantic_node):
            end_reason = "Agent execution graph finished"
            if hasattr(pydantic_node, 'end_reason'):
                end_reason = str(pydantic_node.end_reason)
            
            return EndNode(
                input_messages=input_messages,
                output_messages=output_messages,
                node_type=NodeType.END,
                end_reason=end_reason
            )
        else:
            # Fallback to base Node if type is unknown
            return Node(
                input_messages=input_messages,
                output_messages=output_messages,
                node_type=NodeType.USER_PROMPT  # Default type
            )
    except Exception as e:
        # Log error and return None if reconstruction fails
        import logging
        logging.error(f"Failed to reconstruct node: {str(e)}")
        return None 
    
def pretty_print_node(node: Node, print_user_prompt: bool = False) -> str:
    """Generate a pretty-printed string representation of a node with truncation and nice separators."""
    if not node:
        return "None"
    if isinstance(node, UserPromptNode) and not print_user_prompt:
        return ""
    def truncate(text: str, max_length: int = 200) -> str:
        """Truncate text to max_length and add ellipsis if needed."""
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text
    
    def format_section(title: str, content: str) -> str:
        """Format a section with a nice separator."""
        separator = "─" * 80  # Using thin line character
        return f"\n{separator}\n{title}\n{separator}\n{content}"
    
    def format_node_title(node_type: NodeType) -> str:
        """Format node type into a nice title."""
        title_map = {
            NodeType.USER_PROMPT: "User Prompt",
            NodeType.CALL_TOOLS: "Tool Calls",
            NodeType.MODEL_REQUEST: "Model Request",
            NodeType.MODEL_RESPONSE: "Model Response",
            NodeType.PLANNING: "Planning",
            NodeType.END: "End"
        }
        return title_map.get(node_type, str(node_type).title())
    
    def format_messages_section(messages: List[Message], section_title: str) -> str:
        """Format a messages section."""
        if not messages:
            return ""
        section = [f"\n{section_title}:"]
        for msg in messages:
            section.append(f"  - {msg.role}: {truncate(msg.content)}")
        return "\n".join(section)
    
    # Start with thick separator
    output = ["=" * 80]
    
    # Node Type Section
    output.append(format_section(format_node_title(node.node_type), ""))
    
    # Error Section (if any)
    if node.error:
        output.append(format_section("Error", node.error))
    
    # Messages Section
    messages_section = []
    if node.input_messages:
        messages_section.append(format_messages_section(node.input_messages, "Input Messages"))
    if node.output_messages:
        messages_section.append(format_messages_section(node.output_messages, "Output Messages"))
    
    if messages_section:
        output.append("\n".join(messages_section))
    
    # Node Type Specific Sections
    if isinstance(node, UserPromptNode):
        if print_user_prompt:
            output.append(format_section("Content", truncate(node.user_prompt)))
    elif isinstance(node, CallToolsNode):
        if node.tool_calls:
            tools_section = []
            for tool_call in node.tool_calls:
                name = tool_call.get('name', 'Unknown')
                args = tool_call.get('args', {})
                tools_section.append(f"{name}: {truncate(str(args))}")
            output.append("\n".join(tools_section))
        else:
            output.append("(No tool calls found)")
    
    elif isinstance(node, ModelRequestNode):
        sections = []
        if node.user_prompt:
            sections.append(f"User Prompt: {truncate(node.user_prompt)}")
        if node.tool_return:
            for tool_call in node.tool_return:
                tool_name = tool_call.get('tool_name', 'unknown')
                tool_content = tool_call.get('content', '')
                sections.append(f"{tool_name.title()}: {truncate(tool_content)}")
        if node.retry_prompt:
            sections.append(f"Retry Prompt: {truncate(node.retry_prompt)}")
        if node.system_prompt:
            sections.append(f"System Prompt: {truncate(node.system_prompt)}")
        
        if sections:
            output.append("\n".join(sections))
        else:
            output.append("(No request content found)")
    elif isinstance(node, ModelResponseNode):
        output.append(truncate(node.model_response))
    
    elif isinstance(node, PlanningNode):
        planning_section = [
            f"Facts Survey: {truncate(node.facts_survey)}",
            f"Action Plan: {truncate(node.action_plan)}"
        ]
        output.append("\n".join(planning_section))
    
    elif isinstance(node, EndNode):
        output.append(node.end_reason)
    
    # End with thick separator
    output.append("=" * 80)
    
    return "\n".join(output)
