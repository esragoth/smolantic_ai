import abc
from typing import Any, Dict, List, Optional, TypeVar, Union, Type, AsyncIterator
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool, UserPromptNode, ModelRequestNode, CallToolsNode, messages
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models import ModelResponse, ModelSettings, ModelRequestParameters
from pydantic_graph import End
from jinja2 import Template
import traceback
import inspect
from .models import Message, MessageRole, ActionStep, AgentMemory, MultistepResult, PlanningStep, TaskStep, FinalAnswerStep, CodeResult
from .config import settings_manager
from .logging import get_logger

# Define the generic type variable bound to BaseModel
T = TypeVar('T', bound=BaseModel)

class BaseAgent(Agent[None, T], abc.ABC):
    """Abstract Base Class for agents with shared planning and logging capabilities."""

    def __init__(
        self,
        result_type: Type[T],
        model: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        planning_interval: Optional[int] = None,
        logger_name: Optional[str] = None,
        system_prompt: Optional[str] = None, # Raw template expected
        max_steps: int = 20,
        **kwargs
    ):
        settings = settings_manager.settings
        self.planning_interval = planning_interval
        self.step_count = 0
        self.planning_step_count = 0
        self.max_steps = max_steps
        self.base_tools = tools or [] # Store base tools provided by user
        self.logger = get_logger(logger_name or self.__class__.__name__)
        self.memory = AgentMemory() # Initialize memory

        if self.planning_interval is not None and self.planning_interval < 2:
            raise ValueError("planning_interval must be None or >= 2")

        # --- Subclass must provide formatted prompt and final tools ---
        formatted_system_prompt = self._format_system_prompt(system_prompt or self.default_system_prompt_template)

        super().__init__(
            model=model or f"{settings.model_provider}:{settings.model_name}",
            result_type=result_type,
            deps_type=type(None),
            tools=tools,
            system_prompt=formatted_system_prompt,
            **kwargs
        )
        
        # Log initial setup after super().__init__
        self.logger.info(f"Initialized {self.__class__.__name__} with:")
        self.logger.info(f"  Model: {self.model}")
        self.logger.info(f"  Result Type: {self.result_type.__name__}")
        self.logger.info(f"  Planning Interval: {self.planning_interval}")
        self.logger.info(f"  Max Steps: {self.max_steps}")
        self.logger.info(f"  Tools: {[tool.name for tool in tools]}")

    # --- Abstract methods/properties for subclasses ---
    @property
    @abc.abstractmethod
    def default_system_prompt_template(self) -> str:
        """Return the default system prompt template string for the subclass."""
        pass

    @property
    @abc.abstractmethod
    def initial_planning_template(self) -> str:
        """Return the Jinja template string for initial planning."""
        pass

    @property
    @abc.abstractmethod
    def update_planning_template_pre(self) -> str:
        """Return the Jinja template string for the pre-history part of replanning."""
        pass

    @property
    @abc.abstractmethod
    def update_planning_template_post(self) -> str:
        """Return the Jinja template string for the post-history part of replanning."""
        pass

    @abc.abstractmethod
    def _get_final_tools(self) -> List[Tool]:
        """Return the final list of tools (base + specific)."""
        pass

    @abc.abstractmethod
    def _format_system_prompt(self, template: str) -> str:
        """Format the system prompt template with subclass-specific context."""
        pass

    @abc.abstractmethod
    async def _process_run_result(self, agent_run: AgentRunResult) -> T:
        """Process the final result from the agent run."""
        pass
    # --- End Abstract methods ---

    async def _run_impl(self, task: str, **kwargs) -> T:
        """Internal implementation of the run method."""
        # Reset counters
        self.step_count = 0
        self.planning_step_count = 0
        current_step_logged = False
        
        # Log agent settings
        settings = settings_manager.settings
        settings_info = (
            f"Starting {self.__class__.__name__} run.\n" 
            f"  Task: {task[:100]}...\n"
            f"  Model: {settings.model_provider}:{settings.model_name}\n"
            f"  Planning Interval: {self.planning_interval}\n"
            f"  Max Steps: {self.max_steps}\n"
            f"  Tools: {[tool.name for tool in self.tools]}"
        )
        self.logger.info(settings_info)

        try:
            # --- Initial Planning --- 
            if self.planning_interval: # Check if planning is enabled (interval is set)
                planning_step = await self._create_planning_step(task, is_first_step=True)

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
                    if (self.planning_interval is not None and
                        self.step_count > 0 and 
                        self.step_count % self.planning_interval == 0 and 
                        not current_step_logged): # Avoid replanning immediately after a T-A-O step log
                        self.logger.info(f"--- Triggering Replanning (Step Count: {self.step_count}) ---")
                        planning_step = await self._create_planning_step(task, is_first_step=False)

                # Process the final result
                return await self._process_run_result(agent_run)

        except Exception as e:
            error_msg = f"Critical error during agent run: {type(e).__name__}: {str(e)}"
            tb = traceback.format_exc()
            self.logger.error(error_msg)
            self.logger.error(tb)
            return await self._handle_run_error(e, error_msg, tb)

    async def run(self, task: str, **kwargs) -> T:
        """Run the agent with the given task and return the final result."""
        return await self._run_impl(task, **kwargs)

    async def _handle_run_error(self, error: Exception, error_msg: str, traceback_str: str) -> T:
        """Handle errors during agent run. Subclasses can override to provide custom error handling."""
        try:
            if hasattr(self.result_type, 'model_fields'):
                if 'error' in self.result_type.model_fields:
                    return self.result_type(error=error_msg, explanation=traceback_str)
                elif 'explanation' in self.result_type.model_fields:
                    return self.result_type(explanation=f"{error_msg}\n{traceback_str}")
            return self.result_type()  # Try to create a default instance
        except Exception as e_create:
            self.logger.error(f"Failed to create error result {self.result_type.__name__}: {e_create}")
            raise error  # Re-raise original error if we can't create a result

    async def _reconstruct_and_log_action_step(
        self,
        tool_return_request: messages.ModelRequest,
        assistant_response: messages.ModelResponse
    ) -> Optional[Dict]:
        """Reconstruct and log a complete Thought-Action-Observation step.
        Subclasses can override to provide custom step reconstruction."""
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
            
            # Return minimal info needed for run loop logic
            return {"step_counted": True} 
        
        except Exception as log_err:
            self.logger.error(f"Error during T-A-O step reconstruction/logging: {log_err}")
            self.logger.error(traceback.format_exc())
            return None

    # --- Logging Helper Methods (Common) ---
    def _log_user_prompt_node(self, node: UserPromptNode) -> None:
        prompt_content = "(Prompt content not readily extractable)"
        if hasattr(node, 'user_prompt'):
            if isinstance(node.user_prompt, str):
                 prompt_content = node.user_prompt
            elif hasattr(node.user_prompt, 'content'):
                 prompt_content = str(node.user_prompt.content)
            else:
                 prompt_content = str(node.user_prompt)
        log_lines = ["\n--- User Prompt Node ---", f"Processing Prompt: {prompt_content[:150]}...", "------------------------"]
        self.logger.info("\n".join(log_lines))

    def _log_initial_model_request_node(self, node: ModelRequestNode) -> None:
        log_lines = ["\n--- Model Request Node (Initial/Intermediate) ---", "Sending request to LLM with parts:"]
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
        log_lines = ["\n--- Call Tools Node (LLM Response Received) ---"]
        thought = ""
        actions_log = []
        if hasattr(node.model_response, 'parts') and isinstance(node.model_response.parts, list):
            for part in node.model_response.parts:
                if isinstance(part, messages.TextPart):
                    thought += part.content + "\n"
                elif isinstance(part, messages.ToolCallPart):
                    tool_name = part.tool_name
                    args_data = part.args
                    args_str = str(args_data)
                    if len(args_str) > 150: args_str = args_str[:150] + "..."
                    actions_log.append(f"  Tool Call: {tool_name}(...) Args Preview: {args_str}")
                    # Potentially add subclass-specific logging for certain tools here if needed
        log_lines.append("LLM Thought/Text Response:"); log_lines.append("-"*20)
        log_lines.append(thought.strip() or "(No explicit thought text)"); log_lines.append("-"*20)
        if actions_log: log_lines.extend(actions_log)
        log_lines.append("---------------------------------------------")
        self.logger.info("\n".join(log_lines))

    def _log_tool_return(self, node: ModelRequestNode) -> None:
        log_lines = ["\n--- Tool Return Node (Observation) ---"]
        observation = "(No tool return part found)"
        tool_name = "Unknown"
        if hasattr(node.request, 'parts') and isinstance(node.request.parts, list):
            for part in node.request.parts:
                if isinstance(part, messages.ToolReturnPart):
                    tool_name = part.tool_name
                    observation_content = str(part.content).strip()
                    if len(observation_content) > 200: observation_content = observation_content[:200] + "..."
                    observation = observation_content; break
        log_lines.append(f"Observation from Tool '{tool_name}':"); log_lines.append("-"*20)
        log_lines.append(observation); log_lines.append("-"*20)
        log_lines.append("---------------------------------------")
        self.logger.info("\n".join(log_lines))

    def _log_end_node(self, node: End) -> None:
        log_lines = ["\n--- End Node ---", "Agent execution graph finished.", "----------------"]
        # Final result logging might be better handled in _process_run_result
        self.logger.info("\n".join(log_lines))
    # --- End Logging Helpers ---

    # --- Planning Methods (Common Framework) ---
    def _log_planning_step(self, step: PlanningStep) -> None:
        formatted_data = [f"\n{'='*80}", f"Planning Step {self.planning_step_count}", f"{'='*80}",
                          "\nFacts Survey:", "-"*40, step.facts_survey.strip(), "-"*40,
                          "\nAction Plan:", "-"*40, step.action_plan.strip(), "-"*40,
                          "="*80]
        self.logger.log_step("Planning", "\n".join(formatted_data)) # Assuming logger has log_step

    async def _create_planning_step(self, task: str, is_first_step: bool = True) -> PlanningStep:
        """Create a planning step using self.model.request() with rendered templates."""
        context = self._get_planning_context(task, is_first_step) # Delegate context gathering

        rendered_user_prompt = ""
        prompt_template_source = ""

        if is_first_step:
            self.logger.log_action({"action": "plan", "status": "using_initial_template"})
            template = Template(self.initial_planning_template)
            rendered_user_prompt = template.render(**context)
            prompt_template_source = "initial_planning_template"
        else:
            self.logger.log_action({"action": "replan", "status": "using_update_templates"})
            template_pre = Template(self.update_planning_template_pre)
            template_post = Template(self.update_planning_template_post)
            rendered_pre = template_pre.render(**context)
            rendered_post = template_post.render(**context)
            rendered_user_prompt = f"{rendered_pre}\n\n{rendered_post}"
            prompt_template_source = "update_planning_template_pre/post"

        planning_step = PlanningStep(facts_survey="", action_plan="", input_messages=[], output_messages=[])

        try:
            message = messages.ModelRequest(parts=[messages.UserPromptPart(content=rendered_user_prompt)])
            planning_settings = ModelSettings(stop_sequences=["<end_plan>"])
            request_params = ModelRequestParameters(function_tools=[], allow_text_result=True, result_tools=[])

            self.logger.debug(f"Rendered Planning Prompt (from {prompt_template_source}):\n---\n{rendered_user_prompt}\n---")

            response: Any = await self.model.request(
                messages=[message],
                model_settings=planning_settings,
                model_request_parameters=request_params
            )

            response_str = self._extract_text_from_response(response)
            response_str = response_str.strip()
            if response_str.endswith("<end_plan>"):
                response_str = response_str[:-len("<end_plan>")].strip()

            facts, plan = self._parse_planning_response(response_str)

            planning_step.facts_survey = facts
            planning_step.action_plan = plan
            planning_step.input_messages = [Message(role=MessageRole.USER, content=rendered_user_prompt)]
            planning_step.output_messages = [Message(role=MessageRole.ASSISTANT, content=response_str)]

        except Exception as e:
            self.logger.error(f"Error during planning call: {type(e).__name__}: {e}")
            self.logger.error(traceback.format_exc())
            planning_step.facts_survey = "Error generating planning response."
            planning_step.action_plan = str(e)
            planning_step.input_messages = [Message(role=MessageRole.USER, content=rendered_user_prompt)]
            planning_step.output_messages = [Message(role=MessageRole.ASSISTANT, content=f"Error: {e}")]

        self.planning_step_count += 1
        self._log_planning_step(planning_step)
        return planning_step

    def _get_planning_context(self, task: str, is_first_step: bool) -> Dict[str, Any]:
        """Prepare the context dictionary for rendering planning prompts."""
        context = {
            "task": task,
            "tools": self.tools, # Pass the final list of tools
        }
        # Add subclass-specific context items if needed by overriding this method
        # e.g., context["authorized_imports"] = self.authorized_imports

        if not is_first_step:
            # Use memory to build history string
            history_parts = []
            for i, step in enumerate(self.memory.action_steps):
                 # Only include ActionSteps in history for replanning? Or planning steps too?
                 # Let's include both for now using their summary.
                 step_num = i + 1 # Use index + 1 for step numbering
                 # Differentiate between action and planning steps in history
                 step_type_prefix = "Action" if isinstance(step, ActionStep) else "Planning"
                 history_parts.append(f"Step {step_num} ({step_type_prefix}):\n{step.to_string_summary()}")

            context["history"] = "\n\n".join(history_parts) if history_parts else "(No history recorded yet)"
            context["remaining_steps"] = max(0, self.max_steps - self.step_count)
            
        return context

    def _extract_text_from_response(self, response: Any) -> str:
         """Extracts text content robustly from various response structures."""
         response_str = ""
         if isinstance(response, tuple) and len(response) > 0:
             model_response = response[0]
             if hasattr(model_response, 'parts') and isinstance(model_response.parts, list) and len(model_response.parts) > 0:
                 first_part = model_response.parts[0]
                 if isinstance(first_part, messages.TextPart) and hasattr(first_part, 'content') and first_part.content is not None:
                     response_str = str(first_part.content)
                 else: response_str = str(model_response) # Fallback
             else: response_str = str(model_response) # Fallback
         elif isinstance(response, str): response_str = response
         elif hasattr(response, 'content') and response.content is not None: response_str = str(response.content)
         elif isinstance(response, dict) and response.get('content') is not None: response_str = str(response['content'])
         else: response_str = str(response) # Ultimate fallback
         return response_str

    def _parse_planning_response(self, response_str: str) -> tuple[str, str]:
        """Parses the facts and plan from the planning LLM response."""
        facts = "(Facts section not found)"
        plan = "(Plan section not found)"
        initial_facts_marker = "## 1. Facts survey"
        updated_facts_marker = "## 1. Updated facts survey"
        plan_marker = "## 2. Plan"
        response_lower = response_str.lower()
        facts_marker_to_use = None
        facts_start_index = -1

        updated_facts_start = response_lower.find(updated_facts_marker.lower())
        if updated_facts_start != -1:
            facts_marker_to_use = updated_facts_marker; facts_start_index = updated_facts_start
        else:
            initial_facts_start = response_lower.find(initial_facts_marker.lower())
            if initial_facts_start != -1:
                facts_marker_to_use = initial_facts_marker; facts_start_index = initial_facts_start

        plan_start_index = response_lower.find(plan_marker.lower())

        if facts_marker_to_use and facts_start_index != -1 and plan_start_index != -1:
            facts_content_start = facts_start_index + len(facts_marker_to_use)
            if plan_start_index > facts_content_start:
                facts = response_str[facts_content_start:plan_start_index].strip()
                plan = response_str[plan_start_index + len(plan_marker):].strip()
            else: # Markers out of order
                facts = f"(Plan marker found before {facts_marker_to_use})"
                if plan_start_index + len(plan_marker) < len(response_str):
                    plan = response_str[plan_start_index + len(plan_marker):].strip()
                else: plan = "(Could not extract plan after wrongly placed marker)"
        elif plan_start_index != -1: # Plan found, no preceding facts
            facts = "(Facts section marker not found or not recognized)"
            plan = response_str[plan_start_index + len(plan_marker):].strip()
        elif facts_marker_to_use and facts_start_index != -1: # Facts found, no plan after
             facts_content_start = facts_start_index + len(facts_marker_to_use)
             facts = response_str[facts_content_start:].strip()
             plan = "(Plan section marker not found after facts section)"
        else: # Neither found reliably
            facts = "(Facts/Plan structure markers not detected)"
            plan = response_str # Fallback: assume whole response is plan
        return facts, plan
    # --- End Planning Methods ---

