from typing import Any, Optional, List
from pydantic import BaseModel
from pydantic_ai import Agent, Tool
from .config import settings
from .prompts import TOOL_CALLING_SYSTEM_PROMPT, TOOL_CALLING_PLANNING_INITIAL
from .logging import get_logger
from .agent import MultistepAgent

class ToolCallingResult(BaseModel):
    """Result model for tool calling operations."""
    result: Any
    explanation: str

class ToolCallingAgent(MultistepAgent):
    """Agent that can call tools to accomplish tasks."""

    def __init__(
        self,
        model: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        planning_interval: int = 3,
        **kwargs
    ):
        """Initialize the tool calling agent.
        
        Args:
            model: Model to use for the agent
            tools: List of tools available to the agent
            planning_interval: Number of steps between planning
            **kwargs: Additional arguments passed to base Agent
        """
        super().__init__(
            model=model or settings.tool_calling_model.model_string,
            tools=tools or [],
            system_prompt=TOOL_CALLING_SYSTEM_PROMPT,
            planning_prompt=TOOL_CALLING_PLANNING_INITIAL,
            planning_interval=planning_interval,
            **kwargs
        )
        self.logger = get_logger("tool_calling_agent")
        
    async def run(self, task: str, **kwargs) -> ToolCallingResult:
        """Run the agent with the given task."""
        self.logger.log_planning({"task": task, "tools": [tool.name for tool in self.tools]})
        try:
            result = await super().run(task, **kwargs)
            self.logger.log_result({"result": result.model_dump()})
            return result
        except Exception as e:
            self.logger.log_error(e)
            raise 