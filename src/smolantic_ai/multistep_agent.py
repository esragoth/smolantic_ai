from typing import Any, Dict, List, Optional, TypeVar, cast, Type
from pydantic import Field
from pydantic_ai import  RunContext, Tool  
from pydantic_ai import UserPromptNode, ModelRequestNode, CallToolsNode
from pydantic_graph import End
from .models import (
    Message,
    MessageRole,
    ActionStep,
    AgentMemory,
    AgentResult,
)
from .config import settings_manager
from .logging import get_logger
from .prompts import (
    MULTISTEP_AGENT_SYSTEM_PROMPT,
    MULTISTEP_AGENT_PLANNING_INITIAL,
    MULTISTEP_AGENT_PLANNING_UPDATE_PRE,
    MULTISTEP_AGENT_PLANNING_UPDATE_POST,
)
from jinja2 import Template
from pydantic_ai import messages as pydantic_ai_messages
from dataclasses import dataclass
from smolantic_ai.agent import BaseAgent
import json

logger = get_logger(__name__)

T = TypeVar('T')
DepsT = TypeVar('DepsT')

@dataclass
class MultistepAgentResult(AgentResult):
    """Result of a multistep agent run."""

    steps: List[ActionStep] = Field(default_factory=list)
    """List of steps taken by the agent."""

    @classmethod
    def from_agent_result(cls, result: AgentResult) -> "MultistepAgentResult":
        """Create a MultistepAgentResult from an AgentResult."""
        return cls(
            result=result.result,
            steps=result.steps,
            error=result.error,
            error_traceback=result.error_traceback,
        )

class MultistepAgent(BaseAgent):
    """Agent that can execute multiple steps in sequence."""

    # --- Required Abstract Method Implementations ---
    @property
    def default_system_prompt_template(self) -> str:
        """Return the default system prompt template string for the subclass."""
        return MULTISTEP_AGENT_SYSTEM_PROMPT

    @property
    def initial_planning_template(self) -> str:
        """Return the Jinja template string for initial planning."""
        return MULTISTEP_AGENT_PLANNING_INITIAL

    @property
    def update_planning_template_pre(self) -> str:
        """Return the Jinja template string for the pre-history part of replanning."""
        return MULTISTEP_AGENT_PLANNING_UPDATE_PRE

    @property
    def update_planning_template_post(self) -> str:
        """Return the Jinja template string for the post-history part of replanning."""
        return MULTISTEP_AGENT_PLANNING_UPDATE_POST

    def __init__(
        self,
        model: Any,
        tools: List[Tool],
        result_type: Type[T],
        logger_name: Optional[str] = None,
        max_steps: int = 10,
        planning_interval: Optional[int] = None,
    ):
        """Initialize the agent.

        Args:
            model: The model to use for generating responses.
            tools: The tools available to the agent.
            result_type: The type of result to return.
            logger_name: The name of the logger to use.
            max_steps: The maximum number of steps to take.
            planning_interval: The number of steps between planning steps.
        """
        super().__init__(
            model=model,
            tools=tools,
            result_type=result_type,
            logger_name=logger_name,
            max_steps=max_steps,
            planning_interval=planning_interval,
        )
        self.tools = tools
        self.memory = AgentMemory()

    def add_tool(self, tool: Tool) -> None:
        """Add a single tool to the agent."""
        self.tools.append(tool)
        self.logger.log_action({"action": "add_tool", "tool": tool.name})

    def add_tools(self, tools: List[Tool]) -> None:
        """Add multiple tools to the agent."""
        for tool in tools:
            self.add_tool(tool)

    def write_memory_to_messages(self, summary_mode: bool = False) -> List[Dict[str, str]]:
        """Convert memory steps to messages for the model."""
        messages = []
        for step in self.memory.action_steps:
            if not summary_mode or step.step_type == "final_answer":
                messages.extend(step.to_messages())
        return messages

    def _get_final_tools(self) -> List[Tool]:
        """Get the final set of tools to use."""
        return self.tools

    def _format_system_prompt(self, system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        """Format the system prompt with the given arguments."""
        template = system_prompt or self.default_system_prompt_template
        jinja_template = Template(template)
        return jinja_template.render(**kwargs)

    async def _process_run_result(self, agent_run: AgentResult) -> T:
        """Process the run result and return the final result."""
        if hasattr(agent_run, 'result') and agent_run.result is not None:
            if hasattr(agent_run.result, 'data'):
                actual_result_data = agent_run.result.data
            else:
                actual_result_data = agent_run.result

            self.logger.info(f"Agent run finished. Final result data type: {type(actual_result_data).__name__}")

            if isinstance(actual_result_data, self.result_type):
                return actual_result_data
            else:
                self.logger.error(f"Agent run finished, but extracted data type {type(actual_result_data).__name__} does not match expected {self.result_type.__name__}.")
                explanation_text = f"Agent finished with unexpected result type: {type(actual_result_data).__name__}. Content: {str(actual_result_data)}"
                if hasattr(self.result_type, 'model_fields'):
                    if 'error' in self.result_type.model_fields:
                        return self.result_type(error="Type mismatch", explanation=explanation_text)
                    elif 'explanation' in self.result_type.model_fields:
                        return self.result_type(explanation=explanation_text)
                return self.result_type()  # Try to create a default instance
        else:
            self.logger.error("Agent run completed, but no result object found on agent_run.")
            explanation_text = "Agent finished without producing a final result or error."
            if hasattr(self.result_type, 'model_fields'):
                if 'error' in self.result_type.model_fields:
                    return self.result_type(error="No result", explanation=explanation_text)
                elif 'explanation' in self.result_type.model_fields:
                    return self.result_type(explanation=explanation_text)
            return self.result_type()  # Try to create a default instance 