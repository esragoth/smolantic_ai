# TODO: Verify compatibility and integration with the refactored MultistepAgent base class.
from typing import Any, Optional, List, Union, Dict, TypeVar, Type, Callable, AsyncIterator
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool, UserPromptNode, ModelRequestNode, CallToolsNode, messages
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models import ModelResponse
from pydantic_graph import End
from .executors import PythonExecutor, LocalPythonExecutor, E2BExecutor, DockerExecutor, CodeExecutionResult
from .config import settings_manager
from .prompts import CODE_AGENT_SYSTEM_PROMPT, CODE_AGENT_PLANNING_INITIAL, MULTISTEP_AGENT_PLANNING_INITIAL, MULTISTEP_AGENT_PLANNING_UPDATE_PRE, MULTISTEP_AGENT_PLANNING_UPDATE_POST
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

# Load .env at the module level to ensure environment variables are available
load_dotenv()

# Define the generic type variable bound to BaseModel
T = TypeVar('T', bound=BaseModel)

# --- Define Input Schema for the Code Execution Tool ---

class CodeAgent(Agent[None, Union[CodeResult, T]]):
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
        result_type: Optional[Type[T]] = None,
        model: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        executor_type: str = "local",
        authorized_imports: Optional[List[str]] = None,
        max_print_outputs_length: Optional[int] = None,
        planning_interval: Optional[int] = None,
        logger_name: Optional[str] = None,
        system_prompt: Optional[str] = None, # Keep the passed-in prompt separate
        max_steps: int = 20, # Added max_steps here
        **kwargs
    ):
        # Load settings
        settings = settings_manager.settings
        
        # Set attributes needed before super().__init__ or for formatting
        self.authorized_imports = authorized_imports or []
        self.max_print_outputs_length = max_print_outputs_length
        self.planning_interval = planning_interval
        self.step_count = 0
        self.planning_step_count = 0
        self.max_steps = max_steps # Store max_steps
        self.tools = tools or [] # Initialize tools list here
        self.logger = get_logger(logger_name or self.__class__.__name__)
        
        if self.planning_interval == 1:
            raise ValueError("planning_interval cannot be 1")
        
        if '*' in self.authorized_imports:
            self.logger.warning("Caution: '*' in authorized_imports allows any package to be imported")
        
        # --- Create the Internal Code Execution Tool --- #
        # Note: The function passed to Tool must be synchronous or properly wrapped if async within the base Agent.
        # For simplicity, let's assume the executor itself handles async if needed, or make this tool func async
        # and ensure the base Agent handles awaiting tool calls correctly.
        # Making the tool func async seems more robust with async executors.
        self.python_interpreter_tool = Tool(
            name="python_interpreter",
            description="Executes a string of Python code in the current environment. Returns stdout, stderr, and the final expression result as a string.",
            function=self._execute_code_tool_func, # Reference the method
        )
        # --- End Tool Creation --- #

        # Combine user-provided tools with the internal one
        self.tools = self.tools + [self.python_interpreter_tool]

        # Format the prompt template string directly
        auth_imports_str = "*" if "*" in self.authorized_imports else str(self.authorized_imports)
        # Prepare a simple string representation of tools for the system prompt
        tools_str = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
        # Determine the prompt template string to use
        prompt_template = system_prompt or CODE_AGENT_SYSTEM_PROMPT
        try:
            formatted_system_prompt = prompt_template.format(
                authorized_imports=auth_imports_str,
                tools=tools_str # Provide the tools string
                # Add other potential format variables if needed
            )
        except KeyError as e:
             self.logger.error(f"System prompt template missing key: {e}. Using raw template.")
             self.logger.error(f"Template: {prompt_template}")
             formatted_system_prompt = prompt_template # Fallback to unformatted

        # Initialize base agent with the formatted prompt and other args
        super().__init__(
            model=model or f"{settings.model_provider}:{settings.model_name}",
            result_type=result_type or CodeResult,
            deps_type=type(None),
            tools=self.tools, # Pass the combined list of tools
            system_prompt=formatted_system_prompt,
            **kwargs
        )
        
        # Create executor after super().__init__ has run
        self.executor = self._create_executor(
            executor_type,
            self.authorized_imports,
            max_print_outputs_length
        )

        # Inject tools into the executor state (ensure executor supports it)
        if self.tools and hasattr(self.executor, "state") and isinstance(self.executor.state, dict):
            tool_dict = {tool.name: tool.function for tool in self.tools}
            self.executor.state.update(tool_dict)
            self.logger.info(f"Injected tools {[t.name for t in self.tools]} into executor state.")
        elif self.tools and not (hasattr(self.executor, "state") and isinstance(getattr(self.executor, "state", None), dict)):
             self.logger.warning(f"Executor type {type(self.executor).__name__} does not support state injection for tools.")

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
             # Ensure E2BExecutor accepts logger if needed
            return E2BExecutor(
                authorized_imports=authorized_imports,
                max_print_outputs_length=max_print_outputs_length
            )
        elif executor_type == "docker":
             # Ensure DockerExecutor accepts logger if needed
            return DockerExecutor(
                authorized_imports=authorized_imports,
                max_print_outputs_length=max_print_outputs_length
            )
        else:
            raise ValueError(f"Unsupported executor type: {executor_type}")

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

    async def _execute_code(self, code: str, ctx: RunContext) -> CodeExecutionResult:
        """Execute the code using the executor."""
        try:
            self.logger.info(f"Executing code:\n---\n{code}\n---")
            # Pass context if executor needs it, otherwise call without it
            # Assuming the executor interface is executor(code) or executor(code, context)
            if hasattr(self.executor, '__call__') and len(inspect.signature(self.executor.__call__).parameters) == 2:
                 result = self.executor(code, ctx) # type: ignore
            else:
                 result = self.executor(code) # type: ignore

            if isinstance(result, CodeExecutionResult):
                if result.execution_logs:
                    self.logger.info(f"Execution logs:\n{result.execution_logs}")
                if result.output is not None:
                     self.logger.info(f"Execution output: {result.output}")
                if result.error:
                     self.logger.error(f"Execution error reported: {result.error}")
                return result
            else:
                 # Handle unexpected return type from executor
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

    async def run(self, task: str, **kwargs) -> Union[CodeResult, T]:
        """Run the agent with the given task and return the final result."""
        # Reset counters
        self.step_count = 0
        self.planning_step_count = 0
        current_step_logged = False
        
        # Log agent settings
        settings = settings_manager.settings
        settings_info = (
            f"Starting CodeAgent run.\n" 
            f"  Task: {task[:100]}...\n"
            f"  Model: {settings.model_provider}:{settings.model_name}\n"
            f"  Executor: {type(self.executor).__name__}\n"
            f"  Planning Interval: {self.planning_interval}\n"
            f"  Authorized Imports: {'*' if '*' in self.authorized_imports else self.authorized_imports}\n"
            f"  Max Steps: {self.max_steps}\n"
            f"  Tools: {[tool.name for tool in self.tools]}"
        )
        self.logger.info(settings_info)

        try:
            # --- Initial Planning --- 
            if self.planning_interval: # Check if planning is enabled (interval is set)
                planning_step = await self._create_planning_step(task, is_first_step=True)
                # TODO: Decide how to store/use the planning step (e.g., add to memory if implemented)
                # self.memory.add_step(planning_step) # Example if memory exists

            final_result_obj = None 
            # Use the base agent's iterator
            async with self.iter(task, **kwargs) as agent_run:
                async for step in agent_run:
                    current_step_logged = False # Reset flag for each node

                    # --- Node Logging & Step Counting --- 
                    if Agent.is_user_prompt_node(step):
                        self._log_user_prompt_node(step)
                    elif Agent.is_call_tools_node(step):
                        self._log_call_tools_node(step) # Logs LLM thought/action request
                    elif Agent.is_model_request_node(step):
                        # Check if this node represents the return from tools
                        if hasattr(step.request, 'parts') and any(isinstance(p, messages.ToolReturnPart) for p in step.request.parts):
                            # Attempt to reconstruct and log the full T-A-O step
                            history = agent_run.ctx.state.message_history
                            if history and isinstance(history[-1], messages.ModelResponse):
                                reconstructed_step_info = await self._reconstruct_and_log_action_step(step.request, history[-1])
                                if reconstructed_step_info:
                                    current_step_logged = True # Mark that a full step was logged
                                    # TODO: Store reconstructed_step_info in memory if needed
                            else:
                                # Fallback logging if reconstruction isn't possible
                                self._log_tool_return(step) 
                        else:
                            # Log initial model request (before first LLM call)
                            self._log_initial_model_request_node(step)
                    elif Agent.is_end_node(step):
                        self._log_end_node(step)
                    # --- End Node Logging --- 

                    # --- Replanning Check --- 
                    # Check interval, make sure planning is enabled, step count > 0, 
                    # and we haven't just logged a reconstructed step (to avoid double counting/immediate replan)
                    if (self.planning_interval is not None and
                        self.step_count > 0 and 
                        self.step_count % self.planning_interval == 0 and 
                        not current_step_logged): # Avoid replanning immediately after a T-A-O step log
                        self.logger.info(f"--- Triggering Replanning (Step Count: {self.step_count}) ---")
                        planning_step = await self._create_planning_step(task, is_first_step=False)
                        # TODO: Decide how to store/use the replanning step
                        # self.memory.add_step(planning_step) # Example

                # --- Post-iteration Result Determination (Existing Logic) --- 
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

        except Exception as e:
            error_msg = f"Critical error during agent run: {type(e).__name__}: {str(e)}"
            tb = traceback.format_exc()
            self.logger.error(error_msg)
            self.logger.error(tb)
            return CodeResult(
                code="# Critical error",
                result=None,
                explanation=error_msg,
                execution_logs=tb,
                error=str(e)
            )

    # --- Tool Function Implementation --- #
    async def _execute_code_tool_func(self, code: str) -> str:
        """Function to be called by the python_interpreter tool."""
        # Assuming RunContext might be needed by executor or logging later
        # If not strictly needed now, it can be omitted or passed optionally
        # For now, creating a dummy context if needed, or passing None
        # ctx = RunContext() # Or get it from somewhere if available in this scope

        exec_result = await self._execute_code(code, ctx=None) # Pass None or dummy ctx

        # Format the result into a string observation
        observation_parts = []
        if exec_result.execution_logs:
            # Limit log length for observation? Maybe not, let LLM handle it.
            observation_parts.append(f"Logs:\n{exec_result.execution_logs}")
        if exec_result.output is not None:
            observation_parts.append(f"Output:\n{exec_result.output}")
        if exec_result.error:
             observation_parts.append(f"Execution Error:\n{exec_result.error}")

        if not observation_parts:
            return "Code executed successfully with no output or logs."
        else:
            return "\n".join(observation_parts)
    # --- End Tool Function --- #

    # <<< Logging Helper Methods (Adapted from MultistepAgent) >>>
    def _log_user_prompt_node(self, node: UserPromptNode) -> None:
        """Logs information from a UserPromptNode."""
        # Ensure content extraction is robust
        prompt_content = "(Prompt content not readily extractable)"
        if hasattr(node, 'user_prompt'):
            if isinstance(node.user_prompt, str):
                 prompt_content = node.user_prompt
            elif hasattr(node.user_prompt, 'content'): # Handle potential nested structure
                 prompt_content = str(node.user_prompt.content)
            else:
                 prompt_content = str(node.user_prompt)

        log_lines = [
            "\n--- User Prompt Node ---",
            f"Processing Prompt: {prompt_content[:150]}...",
            "------------------------"
        ]
        # Use self.logger, assuming it has methods like log_step or just info
        self.logger.info("\n".join(log_lines))

    def _log_initial_model_request_node(self, node: ModelRequestNode) -> None:
        """Logs information for an initial/intermediate ModelRequestNode."""
        log_lines = [
            "\n--- Model Request Node (Initial/Intermediate) ---",
            "Sending request to LLM with parts:"
        ]
        if hasattr(node.request, 'parts') and isinstance(node.request.parts, list):
            for i, part in enumerate(node.request.parts):
                part_content_str = str(getattr(part, 'content', '(No Content)'))[:150]
                if len(str(getattr(part, 'content', ''))) > 150: part_content_str += "..."
                log_lines.append(f"  Part {i+1}: {type(part).__name__} - Content: {part_content_str}")
        else:
            req_str = str(node.request)[:200] + ("..." if len(str(node.request)) > 200 else "")
            log_lines.append(f"  Request (no parts attr or not list): {req_str}")
        log_lines.append("---------------------------------------------")
        self.logger.info("\n".join(log_lines))

    def _log_call_tools_node(self, node: CallToolsNode) -> None:
        """Logs information from a CallToolsNode (LLM response with thought/actions)."""
        log_lines = ["\n--- Call Tools Node (LLM Response Received) ---"]
        thought = ""
        actions_log = [] # Formatted action strings

        if hasattr(node.model_response, 'parts') and isinstance(node.model_response.parts, list):
            for part in node.model_response.parts:
                if isinstance(part, messages.TextPart):
                    thought += part.content + "\n"
                elif isinstance(part, messages.ToolCallPart):
                    tool_name = part.tool_name
                    args_data = part.args
                    # Basic argument formatting for logging
                    args_str = str(args_data)
                    if len(args_str) > 150: args_str = args_str[:150] + "..."

                    actions_log.append(f"  Tool Call: {tool_name}(...) Args Preview: {args_str}")
                    # Special handling for python_interpreter to log the code
                    if tool_name == 'python_interpreter' and isinstance(args_data, dict) and 'code' in args_data:
                        code_preview = args_data['code'].strip()[:200] # Show start of code
                        if len(args_data['code'].strip()) > 200: code_preview += "..."
                        actions_log.append(f"    Code to execute:\n------\n{code_preview}\n------")

        log_lines.append("LLM Thought/Text Response:")
        log_lines.append("-"*20)
        log_lines.append(thought.strip() or "(No explicit thought text)")
        log_lines.append("-"*20)

        if actions_log:
            log_lines.extend(actions_log)

        log_lines.append("---------------------------------------------")
        self.logger.info("\n".join(log_lines))

    def _log_tool_return(self, node: ModelRequestNode) -> None:
        """Logs the observation (tool return) being sent back to the LLM."""
        # This node contains the *request* for the *next* LLM call,
        # which includes the ToolReturnPart from the previous step.
        log_lines = ["\n--- Tool Return Node (Observation) ---"]
        observation = "(No tool return part found)"
        tool_name = "Unknown"

        if hasattr(node.request, 'parts') and isinstance(node.request.parts, list):
            for part in node.request.parts:
                if isinstance(part, messages.ToolReturnPart):
                    tool_name = part.tool_name
                    observation_content = str(part.content).strip()
                    if len(observation_content) > 200: observation_content = observation_content[:200] + "..."
                    observation = observation_content
                    break # Assuming one tool return per request node for logging

        log_lines.append(f"Observation from Tool '{tool_name}':")
        log_lines.append("-"*20)
        log_lines.append(observation)
        log_lines.append("-"*20)
        log_lines.append("---------------------------------------")
        self.logger.info("\n".join(log_lines))

    def _log_end_node(self, node: End) -> None:
        """Logs information from the End node."""
        log_lines = [
            "\n--- End Node ---",
            "Agent execution graph finished.",
            "----------------"
        ]
        # Check if the final result is attached to the End node's context or predecessor
        # This might vary based on pydantic-ai implementation details
        final_result_preview = "(Final result location/access method TBD)"
        if hasattr(node, 'result') and node.result: # Check if End node itself has result (unlikely but possible)
             final_result_preview = str(node.result)[:150] + "..."
        # Add logic here if result is stored differently, e.g., in context

        log_lines.append(f"Final Result Preview: {final_result_preview}")
        self.logger.info("\n".join(log_lines))
    # <<< End Logging Helper Methods >>> 

    # --- T-A-O Step Reconstruction (Adapted from MultistepAgent) ---
    async def _reconstruct_and_log_action_step(
        self,
        tool_return_request: messages.ModelRequest,
        assistant_response: messages.ModelResponse
    ) -> Optional[Dict]: # Return basic info dict, not full ActionStep model for now
        """Reconstructs, logs, and counts a complete Thought-Action-Observation step."""
        try:
            # Extract Thought from the assistant response that *requested* the tool call
            thought = ""
            if hasattr(assistant_response, 'parts'):
                for part in assistant_response.parts:
                    if isinstance(part, messages.TextPart):
                        thought += part.content + "\n"
                thought = thought.strip() or "(No explicit thought text)"

            # Extract Tool Calls from the assistant response
            tool_calls_log = []
            if hasattr(assistant_response, 'parts'):
                for part in assistant_response.parts:
                    if isinstance(part, messages.ToolCallPart):
                        tool_name = part.tool_name
                        args_data = part.args
                        args_str = str(args_data)
                        if len(args_str) > 100: args_str = args_str[:100] + "..."
                        tool_calls_log.append(f"  Tool: {tool_name}({args_str}) ID: {part.tool_call_id}")

            # Extract Observations (Tool Returns) from the *next* request parts
            observations_log = []
            if hasattr(tool_return_request, 'parts'):
                for part in tool_return_request.parts:
                    if isinstance(part, messages.ToolReturnPart):
                        output_content = str(part.content).strip()
                        if len(output_content) > 150: output_content = output_content[:150] + "..."
                        observations_log.append(f"  Tool Return ({part.tool_name}): {output_content}")
            
            # Increment step count *after* successfully reconstructing a step
            self.step_count += 1 

            # Log the reconstructed step details
            log_lines = [
                "\n" + "=" * 80,
                f"Action Step {self.step_count}",
                "=" * 80,
                "\nThought:",
                "-" * 40,
                thought,
                "-" * 40
            ]
            if tool_calls_log:
                log_lines.append("\nAction:")
                log_lines.append("-" * 40)
                log_lines.extend(tool_calls_log)
                log_lines.append("-" * 40)
            if observations_log:
                log_lines.append("\nObservation:")
                log_lines.append("-" * 40)
                log_lines.extend(observations_log)
                log_lines.append("-" * 40)
            log_lines.append("=" * 80)

            self.logger.log_step("Action Step", "\n".join(log_lines))
            
            # Return minimal info needed for run loop logic (e.g., just confirmation step happened)
            return {"step_counted": True} 
        
        except Exception as log_err:
            self.logger.error(f"Error during T-A-O step reconstruction/logging: {log_err}")
            self.logger.error(traceback.format_exc())
            return None
    # --- End T-A-O Step Reconstruction ---

    # --- Planning Methods (Adapted from MultistepAgent) ---
    def _log_planning_step(self, step: PlanningStep) -> None:
        """Log a planning step with pretty printing."""
        formatted_data = []
        formatted_data.append("\n" + "=" * 80)  # Step boundary
        formatted_data.append(f"Planning Step {self.planning_step_count}")  # Use planning step count
        formatted_data.append("=" * 80)  # Step boundary
        
        formatted_data.append("\nFacts Survey:")
        formatted_data.append("-" * 40)
        formatted_data.append(step.facts_survey.strip())
        formatted_data.append("-" * 40)
        
        formatted_data.append("\nAction Plan:")
        formatted_data.append("-" * 40)
        formatted_data.append(step.action_plan.strip())
        formatted_data.append("-" * 40)
        
        formatted_data.append("=" * 80)  # Step boundary
        
        self.logger.log_step("Planning", "\n".join(formatted_data))

    async def _create_planning_step(self, task: str, is_first_step: bool = True) -> PlanningStep:
        """Create a planning step using self.model.request() with rendered templates."""
        
        # Import necessary prompts
        from .prompts import (
            CODE_AGENT_PLANNING_INITIAL,
            CODE_AGENT_PLANNING_UPDATE_PRE,
            CODE_AGENT_PLANNING_UPDATE_POST
        )
        
        # Prepare base template context
        context = {
            "task": task,
            "tools": self.tools, # Use the combined list including interpreter
            "authorized_imports": "*" if "*" in self.authorized_imports else str(self.authorized_imports)
        }
        
        rendered_user_prompt = ""
        prompt_template_for_step = ""
        self.logger.info(f"Context before rendering initial plan: {context!r}")
        if is_first_step:
            self.logger.log_action({"action": "plan", "status": "using_initial_template"})
            template = Template(CODE_AGENT_PLANNING_INITIAL)
            self.logger.debug(f"Context before rendering initial plan (keywords): task={context.get('task')!r}, tools_len={len(context.get('tools', []))}")
            # Try rendering with explicit keywords
            rendered_user_prompt = template.render(
                task=context.get('task', 'TASK_NOT_FOUND_IN_CONTEXT'), 
                tools=context.get('tools', []), 
                authorized_imports=context.get('authorized_imports', 'IMPORTS_NOT_FOUND')
            )
            # self.logger.info(f"Planning with prompt template: {rendered_user_prompt}") # Keep user's added log
            prompt_template_for_step = "CODE_AGENT_PLANNING_INITIAL"
        else:
            self.logger.log_action({"action": "replan", "status": "using_update_templates"})
            
            # Prepare history and remaining steps for replanning context
            # TODO: Implement actual history retrieval if memory is added
            # history_str = "\n".join(
            #     f"Step {i+1}:\n{step.to_string_summary()}" 
            #     for i, step in enumerate(self.memory.action_steps) # Assumes memory exists
            # )
            history_str = "(History not yet implemented for CodeAgent)"
            remaining_steps = max(0, self.max_steps - self.step_count) # Assumes self.max_steps exists
            
            context["history"] = history_str
            context["remaining_steps"] = remaining_steps
            
            template_pre = Template(CODE_AGENT_PLANNING_UPDATE_PRE)
            template_post = Template(CODE_AGENT_PLANNING_UPDATE_POST)
            
            # Try rendering with explicit keywords
            rendered_pre = template_pre.render(
                task=context.get('task', 'TASK_NOT_FOUND'), 
                history=context.get('history', 'HISTORY_NOT_FOUND')
            )
            rendered_post = template_post.render(
                task=context.get('task', 'TASK_NOT_FOUND'), 
                history=context.get('history', 'HISTORY_NOT_FOUND'), 
                tools=self.tools, # Use the combined list here too
                remaining_steps=context.get('remaining_steps', -1)
            )
            
            rendered_user_prompt = f"{rendered_pre}\n\n{rendered_post}"
            prompt_template_for_step = "CODE_AGENT_PLANNING_UPDATE_PRE/POST"
            self.logger.info(f"Replanning with prompt template: {rendered_pre}")
            
        planning_step = PlanningStep(facts_survey="", action_plan="", input_messages=[], output_messages=[])

        try:
            from pydantic_ai.models import ModelSettings, ModelRequestParameters
            from pydantic_ai.messages import ModelRequest, UserPromptPart
            from pydantic_ai import messages as pydantic_ai_messages # For response checking
            
            message = ModelRequest(
                parts=[
                    UserPromptPart(content=rendered_user_prompt)
                ]
            )
            # Use minimal settings for planning - no tools expected, just text
            planning_settings = ModelSettings(stop_sequences=["<end_plan>"])
            request_params = ModelRequestParameters(
                function_tools=[], 
                allow_text_result=True, 
                result_tools=[]
            )
            
            # >>> ADDED LOGGING HERE <<<
            self.logger.debug(f"Rendered Planning Prompt:\n---\n{rendered_user_prompt}\n---")
            
            # Call the model instance's request method (using self.model)
            response: Any = await self.model.request(
                messages=[message], 
                model_settings=planning_settings, 
                model_request_parameters=request_params 
            )
            
            # --- Robust Response Text Extraction --- 
            response_str = ""
            if isinstance(response, tuple) and len(response) > 0:
                model_response = response[0]
                if hasattr(model_response, 'parts') and isinstance(model_response.parts, list) and len(model_response.parts) > 0:
                    first_part = model_response.parts[0]
                    if isinstance(first_part, pydantic_ai_messages.TextPart) and hasattr(first_part, 'content') and first_part.content is not None:
                        response_str = str(first_part.content)
                    else:
                        self.logger.log_action({"action": "planning_response_parse", "warning": "First part not TextPart or has no content", "part_type": type(first_part).__name__})
                        response_str = str(model_response)
                else:
                    self.logger.log_action({"action": "planning_response_parse", "warning": "ModelResponse has no parts or parts is empty", "response_type": type(model_response).__name__})
                    response_str = str(model_response)
            elif isinstance(response, str):
                response_str = response
            elif hasattr(response, 'content') and response.content is not None:
                response_str = str(response.content)
            elif isinstance(response, dict) and response.get('content') is not None:
                response_str = str(response['content'])
            else:
                response_str = str(response)
                self.logger.log_action({"action": "planning_response_parse", "warning": "Could not extract content/text attribute directly, using str(response)", "response_type": type(response).__name__})

            response_str = response_str.strip()
            if response_str.endswith("<end_plan>"):
                response_str = response_str[:-len("<end_plan>")].strip()
            # --- End Extraction ---

            # --- Parse Facts and Plan (Handles initial and updated headings) --- 
            facts = "(Facts section not found)"
            plan = "(Plan section not found)"

            # Define possible markers, case-insensitive search
            initial_facts_marker = "## 1. Facts survey"
            updated_facts_marker = "## 1. Updated facts survey"
            plan_marker = "## 2. Plan"

            response_lower = response_str.lower()
            facts_marker_to_use = None
            facts_start_index = -1

            # Check for updated facts marker first
            updated_facts_start = response_lower.find(updated_facts_marker.lower())
            if updated_facts_start != -1:
                facts_marker_to_use = updated_facts_marker
                facts_start_index = updated_facts_start
            else:
                # Check for initial facts marker
                initial_facts_start = response_lower.find(initial_facts_marker.lower())
                if initial_facts_start != -1:
                    facts_marker_to_use = initial_facts_marker
                    facts_start_index = initial_facts_start

            plan_start_index = response_lower.find(plan_marker.lower())

            if facts_marker_to_use and facts_start_index != -1 and plan_start_index != -1:
                # Found both facts and plan markers
                facts_content_start = facts_start_index + len(facts_marker_to_use)
                # Ensure plan marker comes after facts marker
                if plan_start_index > facts_content_start:
                    facts = response_str[facts_content_start:plan_start_index].strip()
                    plan = response_str[plan_start_index + len(plan_marker):].strip()
                else: # Markers found but in wrong order
                    facts = f"(Plan marker found before {facts_marker_to_use})"
                    # Attempt to extract plan anyway if it starts after the facts marker text
                    if plan_start_index + len(plan_marker) < len(response_str):
                         plan = response_str[plan_start_index + len(plan_marker):].strip()
                    else:
                         plan = "(Could not extract plan after wrongly placed marker)"
            elif plan_start_index != -1:
                # Found plan marker, but not a recognized facts marker before it
                facts = "(Facts section marker not found or not recognized)"
                plan = response_str[plan_start_index + len(plan_marker):].strip()
            elif facts_marker_to_use and facts_start_index != -1:
                 # Found facts marker, but not plan marker after it
                 facts_content_start = facts_start_index + len(facts_marker_to_use)
                 facts = response_str[facts_content_start:].strip()
                 plan = "(Plan section marker not found after facts section)"
            else:
                # Neither facts nor plan markers reliably found
                facts = "(Facts/Plan structure markers not detected)"
                plan = response_str # Fallback: assume whole response is plan
            # --- End Parsing --- 

            planning_step.facts_survey = facts
            planning_step.action_plan = plan
            # Store messages as dicts for potential serialization later
            planning_step.input_messages = [{"role": "user", "content": rendered_user_prompt}] 
            planning_step.output_messages = [{"role": "assistant", "content": response_str}]
            
        except Exception as e:
            self.logger.error(f"Error during planning call: {type(e).__name__}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            planning_step.facts_survey = "Error generating planning response."
            planning_step.action_plan = str(e)
            planning_step.input_messages = [{"role": "user", "content": rendered_user_prompt}]
            planning_step.output_messages = [{"role": "assistant", "content": f"Error: {e}"}]

        self.planning_step_count += 1
        self._log_planning_step(planning_step)
        
        return planning_step
    # --- End Planning Methods --- 