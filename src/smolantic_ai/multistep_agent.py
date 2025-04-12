from typing import Any, Dict, List, Optional, TypeVar, Generic, Callable, Union
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings # Import ModelSettings
from pydantic_ai.messages import ModelRequest,SystemPromptPart,UserPromptPart
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
            # Use default from settings if no model is provided
            model_input = f"{settings.multistep_model.provider}:{settings.multistep_model.model_name}"
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
        
        # Initialize base agent - IT handles resolving model_input to self.model
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
        self.tools = tools or []

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
        # Skip logging for non-reasoning steps
        if not step.input_messages or not step.input_messages[-1].content:
            return
            
        # Format the step data in a more readable way
        thought = step.input_messages[-1].content
        
        # Only include non-empty tool calls and outputs
        action_data = {}
        if step.tool_calls:
            action_data["tool_calls"] = step.tool_calls
        if step.tool_outputs:
            action_data["tool_outputs"] = step.tool_outputs
            
        # Format the step data with clear boundaries
        formatted_data = []
        formatted_data.append("\n" + "=" * 80)  # Step boundary
        formatted_data.append(f"Step {self.step_count}")  # Use normal step count
        formatted_data.append("=" * 80)  # Step boundary
        
        if thought:
            formatted_data.append("\nThought:")
            formatted_data.append("-" * 40)  # Thought boundary
            formatted_data.append(thought.strip())
            formatted_data.append("-" * 40)  # Thought boundary
        
        if action_data:
            formatted_data.append("\nAction:")
            formatted_data.append("-" * 40)  # Action boundary
            # Format tool calls in a more readable way
            for tool_call in action_data.get("tool_calls", []):
                formatted_data.append(f"Tool: {tool_call.get('name', 'unknown')}")
                formatted_data.append(f"Arguments: {json.dumps(tool_call.get('args', {}), indent=2)}")
            formatted_data.append("-" * 40)  # Action boundary
        
        # Check if any tool output associated with this step indicates an error
        has_error = any(out.get('is_error', False) for out in step.tool_outputs) if step.tool_outputs else False
        
        if step.observations:
            formatted_data.append("\nObservation:")
            formatted_data.append("-" * 40)  # Observation boundary
            if has_error:
                formatted_data.append(f"ERROR: {step.observations.strip()}") # Prepend ERROR if flagged
            else:
                formatted_data.append(step.observations.strip())
            formatted_data.append("-" * 40)  # Observation boundary
        
        formatted_data.append("=" * 80)  # Step boundary
        
        # Call the step callback if provided
        if self.on_step:
            self.on_step(step)
        
        # Only log if this is not a final_result step
        if not (action_data.get("tool_calls") and 
                any(tool_call.get("name") == "final_result" for tool_call in action_data["tool_calls"])):
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

    async def run(self, task: str, **kwargs) -> T:
        """Run the agent with the given task."""
        # Reset counters
        self.step_count = 0
        self.planning_step_count = 0
        final_result = None
        current_step = None
        
        try:
            async with super().iter(task, **kwargs) as agent_run:
                # Initial planning
                if self.planning_prompt is not None:
                    planning_step = await self._create_planning_step(task, is_first_step=True) # No context needed
                    self.memory.add_step(planning_step)
                    self.planning_step_count += 1
                
                async for node in agent_run:
                    if Agent.is_model_request_node(node):
                        # Process model request
                        if hasattr(node.request, 'parts'):
                            # Create new step if needed
                            if not current_step:
                                current_step = ActionStep(
                                    input_messages=[],
                                    output_messages=[],
                                    tool_calls=[],
                                    tool_outputs=[]
                                )
                            
                            for part in node.request.parts:
                                if part.part_kind == 'tool-return':
                                    # --- Graceful Tool Error Handling ---
                                    tool_name = part.tool_name
                                    tool_output_content = part.content
                                    is_error = False
                                    error_message = ""

                                    # Check for error indication (adapt patterns as needed)
                                    if isinstance(tool_output_content, str) and (
                                        tool_output_content.lower().startswith("error:") or
                                        "exception:" in tool_output_content.lower() or
                                        "failed to run tool" in tool_output_content.lower() # Common patterns
                                    ):
                                        is_error = True
                                        error_message = tool_output_content
                                        self.logger.warning(f"Tool '{tool_name}' execution failed: {error_message}")
                                    
                                    # Add tool output/error to current step
                                    current_step.tool_outputs.append({
                                        'name': tool_name,
                                        'output': tool_output_content,
                                        'is_error': is_error  # Flag indicating if this output is an error
                                    })
                                    
                                    # Initialize observations if it's the first tool return in this step
                                    if current_step.observations is None:
                                        current_step.observations = ""
                                    else:
                                        # Add a newline separator if observations already exist
                                        current_step.observations += "\n"
                                        
                                    # Update observations, aggregating results from multiple tools
                                    observation_prefix = f"{tool_name}: "
                                    if is_error:
                                        observation_prefix += "ERROR: "
                                    current_step.observations += observation_prefix + str(tool_output_content)

                                    # Log step and add to memory when all expected outputs are received
                                    if len(current_step.tool_outputs) == len(current_step.tool_calls):
                                        self._log_step(current_step)
                                        # Add step to memory *after* logging and potential error update
                                        self.memory.add_step(current_step) 
                                        self.step_count += 1  # Increment normal step count

                                        # --- Replanning Check --- MOVE THE BLOCK HERE
                                        if (self.planning_interval is not None and 
                                            self.planning_prompt is not None and 
                                            self.step_count > 0 and 
                                            self.step_count % self.planning_interval == 0):
                                            # self.logger.info(f"Replanning triggered at step {self.step_count}") # Error: AgentLogger has no 'info'
                                            #self.logger.log_action({"action": "replanning_trigger", "step": self.step_count})
                                            planning_step = await self._create_planning_step(task, is_first_step=False)
                                            self.memory.add_step(planning_step)
                                            # self.planning_step_count += 1 # Already incremented in _create_planning_step

                                        # Create new step for next operation, indicating previous outcome
                                        next_step_content = ""
                                        if is_error:
                                            next_step_content = f"Tool '{tool_name}' failed. Error: {error_message}. Continuing process."
                                        else:
                                            # Truncate long outputs for brevity in the next step's input message
                                            output_summary = (str(tool_output_content)[:100] + '...') if len(str(tool_output_content)) > 100 else str(tool_output_content)
                                            next_step_content = f"Tool '{tool_name}' executed. Output: {output_summary}"

                                        current_step = ActionStep(
                                            input_messages=[Message(
                                                role=MessageRole.ASSISTANT,
                                                content=next_step_content
                                            )],
                                            output_messages=[],
                                            tool_calls=[],
                                            tool_outputs=[]
                                        )
                                elif part.part_kind == 'user-prompt':
                                    # Create new step for user input (if not already processing a step)
                                    # Ensure we don't overwrite a step that's waiting for tool returns
                                    if not current_step or not current_step.tool_calls or len(current_step.tool_outputs) == len(current_step.tool_calls):
                                        current_step = ActionStep(
                                            input_messages=[Message(
                                                role=MessageRole.USER,
                                                content=part.content
                                            )],
                                            output_messages=[],
                                            tool_calls=[],
                                            tool_outputs=[]
                                        )
                    elif Agent.is_call_tools_node(node):
                        # Process tool calls
                        if hasattr(node, 'model_response') and hasattr(node.model_response, 'parts'):
                            # Create new step if needed (redundant check, consider removing if current_step always exists here)
                            if not current_step:
                                current_step = ActionStep(
                                    input_messages=[], output_messages=[], tool_calls=[], tool_outputs=[]
                                )
                            
                            # Process all tool calls in this response
                            for part in node.model_response.parts:
                                if hasattr(part, 'tool_name'):
                                    args = json.loads(part.args) if hasattr(part, 'args') and isinstance(part.args, str) else (part.args if hasattr(part, 'args') else {})
                                    tool_name = part.tool_name
                                    
                                    if tool_name == 'final_answer':
                                        # Find the actual final_answer tool function
                                        final_answer_tool = next((t for t in self.tools if t.name == 'final_answer'), None)
                                        if final_answer_tool and callable(final_answer_tool.function):
                                            try:
                                                # Call the function to get the correctly typed result
                                                final_result = final_answer_tool.function(**args)
                                            except Exception as tool_exec_error:
                                                self.logger.error(f"Error executing final_answer tool function: {tool_exec_error}")
                                                # Fallback or raise error, maybe return error dict?
                                                final_result = {"success": False, "error": f"Failed to format final answer: {tool_exec_error}"}
                                        else:
                                            # Tool function not found or not callable, fallback to raw args
                                            self.logger.warning("Could not find or call the 'final_answer' tool function. Returning raw arguments.")
                                            final_result = args
                                    elif tool_name == 'final_result': # Keep handling for potential direct final_result calls
                                        if final_result:
                                            return final_result
                                        else:
                                            # This case might occur if the model calls final_result directly without final_answer
                                            final_result = args # Assume args contain the final result structure
                                    else:
                                        # Append other tool calls to the current step
                                        current_step.tool_calls.append({
                                            'name': tool_name,
                                            'args': args
                                        })
                                        
                                        # Add thought process if available (redundant check, consider removing)
                                        if hasattr(part, 'content') and part.content:
                                            if not current_step.input_messages or current_step.input_messages[-1].content != part.content:
                                                current_step.input_messages.append(Message(
                                                    role=MessageRole.ASSISTANT,
                                                    content=part.content
                                                ))
                    elif Agent.is_end_node(node):
                        # Return the final result if available
                        # This might be set by the final_answer tool call processing above
                        if final_result is not None:
                            return final_result
                        # Or, if PydanticAI produced a final result directly (e.g., simple tasks without tool use)
                        elif hasattr(node, 'result') and node.result is not None:
                           return node.result # Return PydanticAI's direct result
            
            # If loop finishes without returning, check if final_result was set
            if final_result is not None:
                return final_result
            
            # Fallback if no result was ever determined (should ideally not happen in successful runs)
            self.logger.warning("Agent run finished without producing a final result.")
            return {"success": False, "error": "Agent finished without result."}
            
        except Exception as e:
            self.logger.error(f"Error in run: {str(e)}")
            raise 