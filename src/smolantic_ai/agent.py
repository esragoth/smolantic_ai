from typing import Any, Dict, List, Optional, TypeVar, Literal
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool
from .models import Message, MessageRole, ActionStep, AgentMemory
from .executors import PythonExecutor, LocalPythonExecutor, E2BExecutor, DockerExecutor
from .config import settings

T = TypeVar('T', bound=BaseModel)

# Model type definitions
ModelProvider = Literal["openai", "anthropic", "gemini"]
ModelName = str

def get_model_string(provider: ModelProvider, model_name: ModelName) -> str:
    """Get the correct model string format for the given provider."""
    return f"{provider}:{model_name}"

class PlanningStep(BaseModel):
    """Represents a planning step in the multi-step process."""
    plan: str
    steps: List[str]
    reasoning: str

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

class MultistepResult(BaseModel):
    """Result model for multi-step operations."""
    planning: PlanningStep
    tasks: List[TaskStep]
    final_answer: FinalAnswerStep
    memory: AgentMemory

class ToolCallingResult(BaseModel):
    """Result model for tool calling operations."""
    result: Any
    explanation: str

class CodeResult(BaseModel):
    """Result model for code operations."""
    code: str
    result: Any
    explanation: str

class MultistepAgent(Agent[MultistepResult]):
    """Base agent specialized for handling multi-step tasks with planning and execution."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        planning_interval: Optional[int] = None,
        max_steps: int = 20,
        **kwargs
    ):
        super().__init__(
            model=model or settings.multistep_model.model_string,
            system_prompt="""You are a helpful assistant that follows a structured approach to solving problems.
            For each task:
            1. Create a detailed plan with clear steps
            2. Execute each step sequentially
            3. Track progress and handle errors
            4. Provide a final answer with explanation
            
            Always maintain context and explain your reasoning at each step.""",
            tools=tools or [],
            result_type=MultistepResult,
            **kwargs
        )
        self.planning_interval = planning_interval
        self.max_steps = max_steps
        self.memory = AgentMemory()

    async def run(self, task: str, **kwargs) -> MultistepResult:
        """Run the agent with the given task."""
        result = await super().run(task, **kwargs)
        return result

class ToolCallingAgent(Agent[ToolCallingResult]):
    """Agent specialized for tool calling operations."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        **kwargs
    ):
        super().__init__(
            model=model or settings.tool_calling_model.model_string,
            result_type=ToolCallingResult,
            system_prompt="""You are a helpful assistant that uses tools to solve problems.
            When using tools:
            1. Plan which tools to use and in what order
            2. Execute each tool with the correct arguments
            3. Combine the results to form a final answer""",
            tools=tools or [],
            **kwargs
        )

class CodeAgent(Agent[CodeResult]):
    """Agent specialized for code generation and execution."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        executor_type: str = "local",
        authorized_imports: Optional[List[str]] = None,
        max_print_outputs_length: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            model=model or settings.code_model.model_string,
            result_type=CodeResult,
            system_prompt="""You are a coding assistant that:
            1. Analyzes the requirements
            2. Plans the code structure
            3. Writes and tests the code
            4. Explains the solution
            
            Always write clean, well-documented code and explain your approach.""",
            **kwargs
        )
        
        self.executor = self._create_executor(
            executor_type,
            authorized_imports or [],
            max_print_outputs_length
        )

    def _create_executor(
        self,
        executor_type: str,
        authorized_imports: List[str],
        max_print_outputs_length: Optional[int]
    ) -> PythonExecutor:
        """Create the appropriate executor based on type."""
        if executor_type == "local":
            return LocalPythonExecutor(
                authorized_imports=authorized_imports,
                max_print_outputs_length=max_print_outputs_length
            )
        elif executor_type == "e2b":
            return E2BExecutor(
                authorized_imports=authorized_imports,
                max_print_outputs_length=max_print_outputs_length
            )
        elif executor_type == "docker":
            return DockerExecutor(
                authorized_imports=authorized_imports,
                max_print_outputs_length=max_print_outputs_length
            )
        else:
            raise ValueError(f"Unsupported executor type: {executor_type}")

    async def run(self, task: str, **kwargs) -> CodeResult:
        """Run the agent with the given task."""
        result = await super().run(task, **kwargs)
        return result 