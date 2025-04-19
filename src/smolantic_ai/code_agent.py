# TODO: Verify compatibility and integration with the refactored MultistepAgent base class.
from typing import Any, Optional, List, Union, Dict, TypeVar, Type, Callable, AsyncIterator
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool, UserPromptNode, ModelRequestNode, CallToolsNode, messages
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models import ModelResponse
from pydantic_graph import End
from .executors import PythonExecutor, LocalPythonExecutor, E2BExecutor, DockerExecutor, CodeExecutionResult
from .config import settings_manager
from .prompts import CODE_AGENT_SYSTEM_PROMPT, CODE_AGENT_PLANNING_INITIAL, CODE_AGENT_PLANNING_UPDATE_PRE, CODE_AGENT_PLANNING_UPDATE_POST
from .logging import get_logger
from .models import ActionStep, CodeResult, PlanningStep
from .utils import parse_code_blobs, fix_final_answer_code, extract_thought_action_observation
import traceback
import re
import json
import os
from dotenv import load_dotenv
import inspect
from jinja2 import Template
from .agent import BaseAgent
# Load .env at the module level to ensure environment variables are available
load_dotenv()

# Define the generic type variables
DepsT = TypeVar('DepsT')
ResultT = TypeVar('ResultT', bound=BaseModel)

class CodeAgent(BaseAgent[DepsT, Union[CodeResult, ResultT]]):
    """Agent specialized for code generation and execution.
    
    This agent uses a code-based approach where the LLM produces Python code
    that gets executed in the environment. It supports different executor types
    for code isolation and safety.
    
    Attributes:
        executor: The Python executor to use for running code
        authorized_imports: List of allowed module imports
    """
    
    def __init__(
        self,
        result_type: Optional[Type[ResultT]] = None,
        deps_type: Type[DepsT] = type(None),
        model: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        executor_type: str = "local",
        authorized_imports: Optional[List[str]] = None,
        max_print_outputs_length: Optional[int] = None,
        planning_interval: Optional[int] = None,
        logger_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_steps: int = 20,
        verbose: bool = False,
        **kwargs
    ):
        # Store CodeAgent-specific attributes
        self.authorized_imports = authorized_imports or []
        self.max_print_outputs_length = max_print_outputs_length
        self.executor_type = executor_type
        self.tools = tools
        self.tools = self._get_final_tools()
        if '*' in self.authorized_imports:
            self.logger.warning("Caution: '*' in authorized_imports allows any package to be imported")
        
        # Initialize base agent
        super().__init__(
            result_type=result_type or CodeResult,
            deps_type=deps_type,
            model=model,
            tools=self.tools,
            planning_interval=planning_interval,
            logger_name=logger_name,
            system_prompt=system_prompt,
            max_steps=max_steps,
            verbose=verbose,
            **kwargs
        )
        
        # Create executor after super().__init__ has run
        self.executor = self._create_executor(
            executor_type,
            self.authorized_imports,
            max_print_outputs_length
        )

        # Inject tools into the executor state
        if self.tools and hasattr(self.executor, "state") and isinstance(self.executor.state, dict):
            tool_dict = {tool.name: tool.function for tool in self.tools}
            self.executor.state.update(tool_dict)
            self.logger.info(f"Injected tools {[t.name for t in self.tools]} into executor state.")
        elif self.tools and not (hasattr(self.executor, "state") and isinstance(getattr(self.executor, "state", None), dict)):
             self.logger.warning(f"Executor type {type(self.executor).__name__} does not support state injection for tools.")

    # --- Required Abstract Method Implementations ---
    @property
    def default_system_prompt_template(self) -> str:
        return CODE_AGENT_SYSTEM_PROMPT

    @property
    def initial_planning_template(self) -> str:
        return CODE_AGENT_PLANNING_INITIAL

    @property
    def update_planning_template_pre(self) -> str:
        return CODE_AGENT_PLANNING_UPDATE_PRE

    @property
    def update_planning_template_post(self) -> str:
        return CODE_AGENT_PLANNING_UPDATE_POST

    def _get_final_tools(self) -> List[Tool]:
        """Return the final list of tools (base + python_interpreter)."""
        # Create the python_interpreter tool
        self.python_interpreter_tool = Tool(
            name="python_interpreter",
            description="Executes a string of Python code in the current environment. Returns stdout, stderr, and the final expression result as a string.",
            function=self._execute_code_tool_func,
        )
        # Combine user-provided tools with the internal one
        return self.tools + [self.python_interpreter_tool]

    def _format_system_prompt(self, template: str) -> str:
        """Format the system prompt template with code-specific context."""
        auth_imports_str = "*" if "*" in self.authorized_imports else str(self.authorized_imports)
        tools_str = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
        try:
            return template.format(
                authorized_imports=auth_imports_str,
                tools=tools_str
            )
        except KeyError as e:
            self.logger.error(f"System prompt template missing key: {e}. Using raw template.")
            self.logger.error(f"Template: {template}")
            return template

    async def _process_run_result(self, agent_run: AgentRunResult) -> Union[CodeResult, ResultT]:
        """Process the final result from the agent run."""
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
                        return CodeResult(
                            code="# Unexpected result type", result=str(actual_result_data),
                            explanation=explanation_text, error="Type mismatch",
                            execution_logs=""
                        )
        else:
                    self.logger.error("Agent run completed, but no result object found on agent_run.")
                    explanation_text = "Agent finished without producing a final result or error."
                    error_text = "No result object"
                    try:
                        if hasattr(self.result_type, 'model_fields') and \
                           'answer' in self.result_type.model_fields and \
                           'explanation' in self.result_type.model_fields:
                             return self.result_type(answer=error_text, explanation=explanation_text)
                        else:
                             try:
                                  return self.result_type(explanation=explanation_text, error=error_text)
                             except TypeError:
                                  self.logger.warning(f"Could not instantiate {self.result_type.__name__} with error details, falling back to CodeResult.")
                                  return CodeResult(
                                     code="# No result generated", result=None,
                                     explanation=explanation_text, error=error_text,
                                     execution_logs=""
                                  )
                    except Exception as e_create:
                        self.logger.error(f"Failed to create fallback result {self.result_type.__name__}: {e_create}")
                        return CodeResult(
                            code="# No result generated", result=None,
                            explanation=explanation_text + f" (Error creating result object: {e_create})",
                            error=error_text,
                            execution_logs=""
                        )

    # --- CodeAgent-specific Methods ---
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

    async def _execute_code(self, code: str, ctx: Any = None) -> CodeExecutionResult:
        """Execute the code using the executor."""
        try:
            self.logger.info(f"Executing code:\n---\n{code}\n---")
            if hasattr(self.executor, '__call__') and len(inspect.signature(self.executor.__call__).parameters) == 2:
                result = self.executor(code, ctx)
            else:
                result = self.executor(code)

            if isinstance(result, CodeExecutionResult):
                if result.execution_logs:
                    self.logger.info(f"Execution logs:\n{result.execution_logs}")
                if result.output is not None:
                    self.logger.info(f"Execution output: {result.output}")
                if result.error:
                    self.logger.error(f"Execution error reported: {result.error}")
                return result
            else:
                self.logger.error(f"Executor returned unexpected type: {type(result)}. Expected CodeExecutionResult.")
                return CodeExecutionResult(
                    success=False,
                    code=code,
                    output=None,
                    execution_logs=f"Error: Executor returned unexpected type {type(result)}.",
                    error=f"Executor returned unexpected type {type(result)}."
                )

        except Exception as e:
            error_msg = f"Error during code execution: {type(e).__name__}: {str(e)}"
            tb = traceback.format_exc()
            self.logger.error(error_msg)
            self.logger.error(tb)
            return CodeExecutionResult(
                success=False,
                code=code,
                output=None,
                execution_logs=f"{error_msg}\n{tb}",
                error=str(e)
            )

    async def _execute_code_tool_func(self, code: str) -> str:
        """Function to be called by the python_interpreter tool."""
        exec_result = await self._execute_code(code)

        # Format the result into a string observation
        observation_parts = []
        if exec_result.execution_logs:
            observation_parts.append(f"Logs:\n{exec_result.execution_logs}")
        if exec_result.output is not None:
            observation_parts.append(f"Output:\n{exec_result.output}")
        if exec_result.error:
             observation_parts.append(f"Execution Error:\n{exec_result.error}")

        if not observation_parts:
            return "Code executed successfully with no output or logs."
        else:
            return "\n".join(observation_parts)

    def _extract_explanation(self, text: str) -> str:
        """Extract explanation/thought from the model's response text."""
        # Use regex to find thought/explanation before code blocks or final answer patterns
        # Prioritize sections explicitly marked as Thought, Reasoning, Explanation
        thought_pattern = r"^(?:Thought|Reasoning|Explanation):?\s*(.*?)(?=\n```python|\nAction:|\nFinal Answer:|$)"
        match = re.search(thought_pattern, text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()

        # Fallback: capture text before the first code block if no explicit marker found
        code_block_start = text.find("```python")
        if code_block_start > 0:
             # Capture text before the code block, assuming it's the thought
             potential_thought = text[:code_block_start].strip()
             # Avoid capturing just prompt remnants or instructions
             if len(potential_thought) > 10 and not potential_thought.lower().startswith("here is the python code"):
                 return potential_thought

        return "No detailed thought process extracted." # Default if nothing suitable found

    async def _handle_run_error(self, error: Exception, error_msg: str, traceback_str: str) -> Union[CodeResult, ResultT]:
        """Handle errors during agent run with code-specific error handling."""
        try:
            if hasattr(self.result_type, 'model_fields'):
                if 'error' in self.result_type.model_fields:
                    return self.result_type(error=error_msg, explanation=traceback_str)
                elif 'explanation' in self.result_type.model_fields:
                    return self.result_type(explanation=f"{error_msg}\n{traceback_str}")
            # Fallback to CodeResult if we can't create the expected result type
            return CodeResult(
                code="# Critical error",
                result=None,
                explanation=error_msg,
                execution_logs=traceback_str,
                error=str(error)
            )
        except Exception as e_create:
            self.logger.error(f"Failed to create error result {self.result_type.__name__}: {e_create}")
            # Ultimate fallback to CodeResult
            return CodeResult(
                code="# Critical error",
                result=None,
                explanation=f"{error_msg}\n(Error creating result object: {e_create})",
                execution_logs=traceback_str,
                error=str(error)
            )

    def _get_planning_context(self, task: str, is_first_step: bool) -> Dict[str, Any]:
        """Prepare the context dictionary for rendering planning prompts with code-specific context."""
        context = super()._get_planning_context(task, is_first_step)
        # Add code-specific context
        context["authorized_imports"] = "*" if "*" in self.authorized_imports else str(self.authorized_imports)
        return context
