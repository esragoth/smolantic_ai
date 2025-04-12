from typing import Any, Dict, List, Optional, TypeVar, Generic, Callable, Union
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai import UserPromptNode, ModelRequestNode, CallToolsNode
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings # Import ModelSettings
from pydantic_ai.messages import ModelRequest,SystemPromptPart,UserPromptPart
from pydantic_graph import End # <-- Keep End import
from .models import Message, MessageRole, ActionStep, AgentMemory, MultistepResult, PlanningStep, TaskStep, FinalAnswerStep
from .config import settings # Import settings
from .logging import get_logger
from .prompts import (
    MULTISTEP_AGENT_SYSTEM_PROMPT,
    MULTISTEP_AGENT_PLANNING_INITIAL,
    MULTISTEP_AGENT_PLANNING_UPDATE_PRE,
    MULTISTEP_AGENT_PLANNING_UPDATE_POST,
)
import json
from jinja2 import Template # Import Jinja2 Template
from pydantic_ai import messages as pydantic_ai_messages

class PlanningResponse(BaseModel):
    """Response model for planning steps."""
    facts: str = Field(description="Survey of known facts and facts to discover")
    plan: str = Field(description="Step-by-step plan to solve the task")

T = TypeVar('T', bound=BaseModel)

class MultistepAgent(Agent[None, T]):
    """Base agent specialized for handling multi-step tasks with planning and execution."""
    
    # The base Agent class provides self.model after initialization
    model: Model 

    def __init__(
        self,
        model: Union[str, Model] = None, # Accept str or Model instance
        tools: Optional[List[Tool]] = None,
        planning_interval: Optional[int] = None,
        planning_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_steps: int = 20,
        result_type: Optional[type[T]] = None,
        logger_name: Optional[str] = None,
        on_step: Optional[Callable[[ActionStep], None]] = None,
        request_limit: Optional[int] = None,
        **kwargs
    ):
        
        # Determine the effective model string or instance for super()
        if model is None:
            # Use default unified model from settings
            provider = settings.model.provider
            name = settings.model.model_name
            model_input = f"{provider}:{name}"
        else:
            # Pass the provided string or Model instance directly
            model_input = model

        # Validate planning interval
        if planning_interval == 1:
            raise ValueError("planning_interval cannot be 1")
        
        # Set up usage limits
        if request_limit is not None:
            from pydantic_ai.usage import UsageLimits
            kwargs['usage_limits'] = UsageLimits(request_limit=request_limit)
        
        # Initialize base agent - Pass result_type directly
        # Also pass result_tool_name/description if provided
        super().__init__(
            model=model_input,
            result_type=result_type,
            tools=tools or [],
            system_prompt=system_prompt or MULTISTEP_AGENT_SYSTEM_PROMPT,
            **kwargs
        )
        
        # NO MORE manual adapter instantiation here
        # self.model is now correctly populated by super().__init__
        
        # Store other multistep-specific attributes
        self.planning_interval = planning_interval
        self.planning_prompt = planning_prompt or MULTISTEP_AGENT_PLANNING_INITIAL
        self.max_steps = max_steps
        self.memory = AgentMemory()
        self.logger = get_logger(logger_name)
        self.step_count = 0
        self.planning_step_count = 0
        self.on_step = on_step
        self.system_prompt = system_prompt or MULTISTEP_AGENT_SYSTEM_PROMPT
        self.tools = list(self._function_tools.values()) # Get tools registered by super

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

    def _log_step(self, step: ActionStep) -> None:
        """Log a single step with pretty printing."""
        # Get thought from the reconstructed input message
        thought = step.input_messages[0].content if step.input_messages else "No thought recorded."

        formatted_data = []
        formatted_data.append("\n" + "=" * 80)
        formatted_data.append(f"Step {self.step_count}")
        formatted_data.append("=" * 80)

        formatted_data.append("\nThought:")
        formatted_data.append("-" * 40)
        formatted_data.append(thought.strip())
        formatted_data.append("-" * 40)

        if step.tool_calls:
            formatted_data.append("\nAction:")
            formatted_data.append("-" * 40)
            for tool_call in step.tool_calls:
                tool_name = tool_call.get('name', 'unknown')
                args = tool_call.get('args', {})
                # Format args as key=value string pairs
                args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                formatted_data.append(f"  Tool: {tool_name}({args_str})")
            formatted_data.append("-" * 40)

        # Use the detailed tool_outputs list from the reconstructed step
        if step.tool_outputs:
            formatted_data.append("\nObservation:")
            formatted_data.append("-" * 40)
            for output in step.tool_outputs:
                tool_name = output.get('name', 'unknown')
                output_content = output.get('output', 'N/A')
                is_error = output.get('is_error', False)
                # Limit output length for cleaner logs
                output_content_str = str(output_content).strip()
                max_len = 200
                if len(output_content_str) > max_len:
                    output_content_str = output_content_str[:max_len] + "..."

                if is_error:
                    formatted_data.append(f"  Tool {tool_name} ERROR: {output_content_str}")
                else:
                    formatted_data.append(f"  Tool {tool_name} OK: {output_content_str}")
            formatted_data.append("-" * 40)
        elif step.observations and step.observations != "No tool outputs.": # Fallback for safety
             formatted_data.append("\nObservation:")
             formatted_data.append("-" * 40)
             formatted_data.append(step.observations.strip()[:200] + ("..." if len(step.observations.strip()) > 200 else ""))
             formatted_data.append("-" * 40)

        formatted_data.append("=" * 80)

        # Call the step callback if provided
        if self.on_step:
            self.on_step(step)

        # Log the formatted step data
        self.logger.log_step("Reasoning", "\n".join(formatted_data))

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
        
        # Prepare template context
        context = {
            "task": task,
            "tools": self.tools,
            "managed_agents": [] # Placeholder, add actual managed agents if applicable
        }
        
        rendered_user_prompt = ""
        prompt_template_for_step = ""

        if is_first_step:
            template = Template(MULTISTEP_AGENT_PLANNING_INITIAL)
            rendered_user_prompt = template.render(context)
            prompt_template_for_step = MULTISTEP_AGENT_PLANNING_INITIAL # Store original template for logging/reference
        else:
            # Prepare history and remaining steps for replanning context
            history_str = "\n".join(
                f"Step {i+1}:\n{step.to_string_summary()}" 
                for i, step in enumerate(self.memory.action_steps)
            )
            remaining_steps = self.max_steps - self.step_count
            
            context["history"] = history_str
            context["remaining_steps"] = remaining_steps
            
            # For replanning, we might use a combination of PRE and POST prompts,
            # or just send the combined context to a single replan template if designed that way.
            # Let's use PRE and POST as distinct parts for now.
            # The PRE part sets up the history, the POST asks for the new plan.
            # We'll combine them into the user prompt.
            # The PRE part sets up the history, the POST asks for the new plan.
            # We'll combine them into the user prompt.
            template_pre = Template(MULTISTEP_AGENT_PLANNING_UPDATE_PRE)
            template_post = Template(MULTISTEP_AGENT_PLANNING_UPDATE_POST)
            
            rendered_pre = template_pre.render(context)
            rendered_post = template_post.render(context)
            
            rendered_user_prompt = f"{rendered_pre}\n\n{rendered_post}"
            prompt_template_for_step = f"{MULTISTEP_AGENT_PLANNING_UPDATE_PRE}\n\n{MULTISTEP_AGENT_PLANNING_UPDATE_POST}" # Store original templates

            self.logger.log_action({"action": "replan", "status": "using_templated_prompts"})
            
        planning_step = PlanningStep(facts_survey="", action_plan="", input_messages=[], output_messages=[])

        try:
            # Note: The detailed structure (system vs user) might need adjustment based on how
            # the new prompts expect to be used. Assuming the entire rendered content
            # should go into the UserPromptPart for now. A SystemPromptPart might be empty or generic.
            message = ModelRequest(
                parts=[
                    # SystemPromptPart(content="You are a planning expert."), # Optional generic system prompt
                    UserPromptPart(content=rendered_user_prompt)
                ]
            )
            # 2. Define model settings including stop sequences
            planning_settings = ModelSettings(stop_sequences=["<end_plan>"])
            
            # 3. Create ModelRequestParameters instance with required args
            request_params = ModelRequestParameters(
                function_tools=[], 
                allow_text_result=True, 
                result_tools=[]
            )
            
            # 4. Call the model instance's request method (using self.model)
            response: Any = await self.model.request(
                messages=[message], 
                model_settings=planning_settings, 
                model_request_parameters=request_params 
            )
            
            # 5. Extract raw text content (existing robust extraction logic)
            response_str = ""
            # Check if response is a tuple and has at least one element
            if isinstance(response, tuple) and len(response) > 0:
                model_response = response[0] # Get the ModelResponse object
                # Check if model_response has 'parts' and it's a list
                if hasattr(model_response, 'parts') and isinstance(model_response.parts, list) and len(model_response.parts) > 0:
                    # Get the first part (assuming it's the main text response)
                    first_part = model_response.parts[0]
                    # Check if the part has a 'content' attribute
                    if hasattr(first_part, 'content') and first_part.content is not None:
                        response_str = str(first_part.content)
                    else:
                        self.logger.log_action({"action": "planning_response_parse", "warning": "First part has no 'content' attribute", "part_type": type(first_part).__name__})
                        response_str = str(model_response) # Fallback to string representation of ModelResponse
                else:
                    self.logger.log_action({"action": "planning_response_parse", "warning": "ModelResponse has no 'parts' or 'parts' is empty", "response_type": type(model_response).__name__})
                    response_str = str(model_response) # Fallback to string representation of ModelResponse

            elif isinstance(response, str): # Handle direct string response (less likely now but good fallback)
                response_str = response
            elif hasattr(response, 'content') and response.content is not None: # Handle object with content attribute
                response_str = str(response.content)
            elif isinstance(response, dict) and response.get('content') is not None: # Handle dict with content key
                response_str = str(response['content'])
            else:
                # Fallback to string representation if specific extraction fails
                response_str = str(response)
                self.logger.log_action({"action": "planning_response_parse", "warning": "Could not extract content/text attribute directly, using str(response)", "response_type": type(response).__name__})

            response_str = response_str.strip()

            # Stop sequence handling
            if response_str.endswith("<end_plan>"):
                response_str = response_str[:-len("<end_plan>")].strip()

            # Debug Print (Optional: useful for verifying rendered prompt and response)
            # print("\nRendered Planning Prompt:")
            # print("-" * 80)
            # print(rendered_user_prompt)
            # print("-" * 80)
            # print("\nRaw Planning Response Text (from model.request with templates):")
            # print("=" * 80)
            # print(response_str)
            # print("=" * 80)
            
            # 6. Parse the response string (using existing logic, should still work with headings)
            facts_marker = "## 1. Facts survey"
            plan_marker = "## 2. Plan"
            
            # Handle both initial and updated fact survey headings
            updated_facts_marker = "## 1. Updated facts survey" 
            effective_facts_marker = facts_marker
            if updated_facts_marker in response_str:
                effective_facts_marker = updated_facts_marker

            facts_start = response_str.find(effective_facts_marker)
            plan_start = response_str.find(plan_marker)
            
            if facts_start != -1 and plan_start != -1:
                facts_content_start = facts_start + len(effective_facts_marker)
                facts = response_str[facts_content_start:plan_start].strip()
                plan = response_str[plan_start + len(plan_marker):].strip()
            # Fallback if markers aren't found exactly as expected
            elif response_str.startswith(effective_facts_marker): 
                 # Find where the plan starts relative to the facts marker
                plan_start_relative = response_str.find(plan_marker, len(effective_facts_marker))
                if plan_start_relative != -1:
                    facts = response_str[len(effective_facts_marker):plan_start_relative].strip()
                    plan = response_str[plan_start_relative + len(plan_marker):].strip()
                else: # Assume rest is plan if plan marker not found after facts
                    facts = "(Plan marker not found after facts marker)"
                    plan = response_str[len(effective_facts_marker):].strip() 
            else:
                facts = "(Facts section marker not found)"
                plan = response_str # Assume the whole response is the plan if no structure found

            # Update PlanningStep object
            planning_step.facts_survey = facts
            planning_step.action_plan = plan
            # Log the *rendered* prompt as input
            planning_step.input_messages = [Message(role=MessageRole.USER, content=rendered_user_prompt)] 
            planning_step.output_messages = [Message(role=MessageRole.ASSISTANT, content=response_str)]
            
        except Exception as e:
            self.logger.error(f"Error during templated planning call: {type(e).__name__}: {e}")
            planning_step.facts_survey = "Error generating planning response via templated call."
            planning_step.action_plan = str(e)
            # Log the *rendered* prompt even in case of error
            planning_step.input_messages = [Message(role=MessageRole.USER, content=rendered_user_prompt)] 
            planning_step.output_messages = [Message(role=MessageRole.ASSISTANT, content=f"Error: {e}")]

        # Log the planning step (using existing method)
        self.planning_step_count += 1 # Increment count here now that step is complete
        self._log_planning_step(planning_step)
        
        return planning_step

    async def _handle_messages(self, messages: List[Message], context: RunContext) -> Any:
        """Override the base agent's message handling to add step logging."""
        # Create an ActionStep to track this interaction
        step = ActionStep(
            input_messages=messages,
            output_messages=[],
            tool_calls=[],
            tool_outputs=[]
        )
        
        # Get response from base agent
        response = await super()._handle_messages(messages, context)
        
        # Update step with response data
        if isinstance(response, dict):
            # Extract thought process from messages
            if 'messages' in response:
                step.input_messages.extend(response['messages'])
            
            # Extract tool calls
            if 'tool_calls' in response:
                step.tool_calls = response['tool_calls']
            
            # Extract tool outputs
            if 'tool_outputs' in response:
                step.tool_outputs = response['tool_outputs']
                step.observations = str(response.get('result', ''))
        
        # Log the step immediately if it contains meaningful information
        if (step.input_messages and step.input_messages[-1].content) or step.tool_calls or step.observations:
            self._log_step(step)
        
        # Store the step in memory
        self.memory.add_step(step)
        
        return response

    # <<< Logging Helper Methods >>>
    def _log_user_prompt_node(self, node: UserPromptNode) -> None:
        """Logs information from a UserPromptNode."""
        log_lines = []
        log_lines.append("\n--- User Prompt Node ---")
        log_lines.append(f"Processing Prompt: {str(node.user_prompt)[:100]}...")
        log_lines.append("------------------------")
        self.logger.log_step("Input Node", "\n".join(log_lines))

    def _log_initial_model_request_node(self, node: ModelRequestNode) -> None:
        """Logs information for an initial/intermediate ModelRequestNode."""
        log_lines = []
        log_lines.append("\n--- Model Request Node (Initial/Intermediate) ---")
        log_lines.append("Sending request to LLM with parts:")
        if hasattr(node.request, 'parts'):
            for i, part in enumerate(node.request.parts):
                part_content_str = str(getattr(part, 'content', '(No Content)'))[:150]
                if len(str(getattr(part, 'content', ''))) > 150: part_content_str += "..."
                log_lines.append(f"  Part {i+1}: {type(part).__name__} - Content: {part_content_str}")
        else:
            req_str = str(node.request)[:200] + ("..." if len(str(node.request)) > 200 else "")
            log_lines.append(f"  Request (no parts attr): {req_str}")
        log_lines.append("---------------------------------------------")
        self.logger.log_step("LLM Request Node", "\n".join(log_lines))

    def _log_call_tools_node(self, node: CallToolsNode) -> None:
        """Logs information from a CallToolsNode (LLM response with thought/actions)."""
        log_lines = []
        log_lines.append("\n--- Call Tools Node (LLM Response Received) ---")
        thought = ""
        actions = [] # Formatted action strings
        if hasattr(node.model_response, 'parts'):
            for part in node.model_response.parts:
                if isinstance(part, pydantic_ai_messages.TextPart):
                    thought += part.content + "\n"
                elif isinstance(part, pydantic_ai_messages.ToolCallPart):
                    # Logic to format tool calls
                    tool_name = part.tool_name
                    args_data = part.args
                    parsed_dict_args = None
                    raw_args_fallback = None

                    if isinstance(args_data, str):
                        try:
                            parsed = json.loads(args_data)
                            if isinstance(parsed, dict):
                                parsed_dict_args = parsed
                            else:
                                raw_args_fallback = repr(parsed)
                        except json.JSONDecodeError:
                            raw_args_fallback = repr(args_data)
                    elif isinstance(args_data, dict):
                        parsed_dict_args = args_data
                    else:
                        raw_args_fallback = repr(args_data)
                    
                    actions.append("  Tool Call:")
                    actions.append(f"    Tool Name: {tool_name}")
                    actions.append("    Args:")
                    if parsed_dict_args is not None:
                        if parsed_dict_args:
                            for k, v in parsed_dict_args.items():
                                actions.append(f"      {k}: {v!r}") 
                        else:
                            actions.append("      (No arguments)")
                    else:
                        actions.append(f"      (Non-dict args): {raw_args_fallback}")
        
        log_lines.append("LLM Thought:")
        log_lines.append("-"*20)
        log_lines.append(thought.strip() or "(No explicit thought text)")
        log_lines.append("-"*20)
        
        if actions:
            log_lines.extend(actions)
            
        log_lines.append("---------------------------------------------")
        self.logger.log_step("LLM Response Node", "\n".join(log_lines))

    def _log_end_node(self, node: End) -> None:
        """Logs information from the End node."""
        self.logger.log_step("End","")

    # <<< Keep existing helper >>>
    async def _reconstruct_and_log_action_step(
        self,
        tool_return_request: pydantic_ai_messages.ModelRequest,
        assistant_response: pydantic_ai_messages.ModelResponse
    ) -> Optional[ActionStep]:
        """Reconstructs, stores, and logs a complete Thought-Action-Observation step."""
        try:
            thought = ""
            if hasattr(assistant_response, 'parts'):
                for part in assistant_response.parts:
                    if isinstance(part, pydantic_ai_messages.TextPart):
                        thought += part.content + "\n"
                thought = thought.strip() or "(No explicit thought text)"

            tool_calls = []
            if hasattr(assistant_response, 'parts'):
                for part in assistant_response.parts:
                    if isinstance(part, pydantic_ai_messages.ToolCallPart):
                        tool_name = part.tool_name
                        args_data = part.args
                        parsed_dict_args = None
                        raw_args_fallback = None

                        if isinstance(args_data, str):
                            try:
                                parsed = json.loads(args_data)
                                if isinstance(parsed, dict):
                                    parsed_dict_args = parsed
                                else:
                                    raw_args_fallback = repr(parsed)
                            except json.JSONDecodeError:
                                raw_args_fallback = repr(args_data)
                        elif isinstance(args_data, dict):
                            parsed_dict_args = args_data
                        else:
                            raw_args_fallback = repr(args_data)
                        
                        tool_calls.append({
                            'name': tool_name, 
                            'args': parsed_dict_args if parsed_dict_args is not None else { 'raw_args_fallback': raw_args_fallback },
                            'id': part.tool_call_id
                        })

            tool_outputs = []
            observations_list = []
            if hasattr(tool_return_request, 'parts'):
                for part in tool_return_request.parts:
                    if isinstance(part, pydantic_ai_messages.ToolReturnPart):
                            output_content = str(part.content)
                            output_data = {'name': part.tool_name, 'output': output_content}
                            tool_outputs.append(output_data)
                            status = "OK"
                            observations_list.append(f"Tool {part.tool_name} {status}: {output_content}")
            
            input_messages_for_log = [Message(role=MessageRole.ASSISTANT, content=thought)]

            step = ActionStep(
                input_messages=input_messages_for_log,
                output_messages=[Message(role=MessageRole.ASSISTANT, content=str(assistant_response))],
                tool_calls=tool_calls,
                tool_outputs=tool_outputs,
                observations="\n".join(observations_list) or "No tool outputs.",
                error=None 
            )

            self.step_count += 1
            self.memory.add_step(step)
            self._log_step(step) # Log the reconstructed step (uses its own detailed formatting)
            return step
        
        except Exception as log_err:
            self.logger.error(f"Error during T-A-O step reconstruction/logging: {log_err}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    async def run(self, task: str, **kwargs) -> T:
        """Run the agent with the given task, relying on base agent for final result."""
        # Log agent settings at the start of the run
        # Access model name safely
        model_name = "N/A"
        if hasattr(self.model, 'model_name'): # Check if attribute exists
            model_name = self.model.model_name
        elif hasattr(self.model, 'model'): # Fallback for potential different naming
             model_name = self.model.model
        
        # Format settings for info logging
        settings_info = (
            f"Starting agent run.\n" 
            f"  Task: {task}\n"
            f"  Model: {model_name}\n"
            f"  Planning Interval: {self.planning_interval}\n"
            f"  Max Steps: {self.max_steps}\n"
            f"  Tools: {[tool.name for tool in self.tools]}"
        )
        # Log using logger.info
        self.logger.info(settings_info)
        
        # Reset counters
        self.step_count = 0
        self.planning_step_count = 0
        current_step_logged = False # Flag to avoid double logging within loop

        try:
            # Use the base agent's iterator
            async with super().iter(task, **kwargs) as agent_run:
                # Initial planning
                if self.planning_prompt is not None:
                    planning_step = await self._create_planning_step(task, is_first_step=True)
                    self.memory.add_step(planning_step)
                    # planning_step_count incremented in _create_planning_step
                
                async for node in agent_run:
                    current_step_logged = False # Reset flag for each node

                    # --- Call Logging Helpers Based on Node Type ---
                    if Agent.is_user_prompt_node(node):
                        self._log_user_prompt_node(node)

                    elif Agent.is_model_request_node(node):
                        if hasattr(node.request, 'parts') and \
                           any(isinstance(part, pydantic_ai_messages.ToolReturnPart) for part in node.request.parts):
                            # Handle T-A-O logging via the other helper
                            history = agent_run.ctx.state.message_history
                            if history and isinstance(history[-1], pydantic_ai_messages.ModelResponse):
                                reconstructed_step = await self._reconstruct_and_log_action_step(
                                    node.request, 
                                    history[-1]
                                )
                                if reconstructed_step:
                                    current_step_logged = True
                        else:
                            # Log initial/intermediate request via its helper
                            self._log_initial_model_request_node(node)

                    elif Agent.is_call_tools_node(node):
                         # Log LLM response/actions via its helper
                        self._log_call_tools_node(node)

                    elif Agent.is_end_node(node):
                        # Log final result via its helper
                        self._log_end_node(node)
                    
                    # else: 
                    #    # Handle potential unknown node types if necessary
                    #    pass

                    # --- Replanning Check (Uses updated self.step_count) ---
                    if (self.planning_interval is not None and
                        self.planning_prompt is not None and
                        self.step_count > 0 and
                        self.step_count % self.planning_interval == 0 and
                        not current_step_logged):
                        planning_step = await self._create_planning_step(task, is_first_step=False)
                        self.memory.add_step(planning_step)

                # --- Get Final Result (Now only used for return value) ---
                if agent_run.result:
                    return agent_run.result.data
                else:
                    self.logger.error("Agent run finished via super().iter() but no result was found.")
                    raise RuntimeError("Agent finished without result.")
            
        except Exception as e:
            self.logger.error(f"Error in run: {str(e)}")
            # Consider adding traceback logging
            import traceback
            self.logger.error(traceback.format_exc())
            raise 