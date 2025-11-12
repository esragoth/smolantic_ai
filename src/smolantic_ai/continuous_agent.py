from typing import Any, Dict, List, Optional, TypeVar, Type, Callable
from pydantic import BaseModel
from pydantic_ai import Tool
from pydantic_ai.run import AgentRun
from .multistep_agent import MultistepAgent
from .models import Message, MessageRole
from .agent import AgentInfo
import asyncio
from datetime import datetime

ResultT = TypeVar('ResultT', bound=BaseModel)

class ContinuousAgent(MultistepAgent[None, ResultT]):
    """An agent that maintains continuous conversation using agent graph control."""
    
    def __init__(
        self,
        output_type: Type[ResultT],
        tools: List[Tool],
        model: str = "openai:gpt-4.1",
        planning_interval: Optional[int] = 3,
        max_steps: int = 20,
        node_callback: Optional[Callable[[Any, AgentInfo], None]] = None,
        step_callback: Optional[Callable[[Any, AgentInfo], None]] = None,
        verbose: bool = True,
        name: Optional[str] = None,
        use_persistent_session: bool = True,
    ):
        """Initialize the continuous agent.
        
        Args:
            output_type: The type of result to return for each interaction
            tools: The tools available to the agent
            model: The model to use for generating responses
            planning_interval: The number of steps between planning steps
            max_steps: The maximum number of steps per interaction
            node_callback: Callback function for node events
            step_callback: Callback function for step events
            verbose: Whether to enable verbose logging
            name: Custom name for the agent instance
            use_persistent_session: If True, uses agent graph control for true continuity
        """
        super().__init__(
            output_type=output_type,
            tools=tools,
            model=model,
            planning_interval=planning_interval,
            max_steps=max_steps,
            node_callback=node_callback,
            step_callback=step_callback,
            verbose=verbose,
            name=name,
            enable_initial_planning=False
        )
        self.conversation_messages: List[Dict[str, str]] = []
        self.is_conversation_active = False
        self.conversation_start_time = None
        self.use_persistent_session = use_persistent_session
        self._agent_run: Optional[AgentRun] = None
        self._agent_run_context = None
        self._last_result: Optional[ResultT] = None
        
    async def start_conversation(self) -> None:
        """Start a continuous conversation using agent graph control."""
        self.is_conversation_active = True
        self.conversation_start_time = datetime.now()
        self.conversation_messages = []
        self.memory.reset()
        
        if self.use_persistent_session:
            try:
                # Start with an initial prompt to set up the session
                initial_prompt = "Hello, I'm ready to help you. What would you like to do?"
                
                # Start the agent run context
                self._agent_run_context = self.iter(initial_prompt)
                self._agent_run = await self._agent_run_context.__aenter__()
                
                if self.verbose:
                    print("✅ Agent graph session started - ready for continuous conversation")
                    
            except Exception as e:
                if self.verbose:
                    print(f"Failed to start agent graph session: {e}")
                # Clean up
                if self._agent_run_context:
                    try:
                        await self._agent_run_context.__aexit__(None, None, None)
                    except:
                        pass
                self._agent_run = None
                self._agent_run_context = None
        
    async def stop_conversation(self) -> None:
        """Stop the continuous conversation."""
        self.is_conversation_active = False
        
        # Clean up agent run if it exists
        if self._agent_run_context:
            try:
                await self._agent_run_context.__aexit__(None, None, None)
            except Exception as e:
                if self.verbose:
                    print(f"Error closing agent run: {e}")
            finally:
                self._agent_run = None
                self._agent_run_context = None
        
    def _add_to_conversation(self, role: str, content: str) -> None:
        """Add a message to the conversation."""
        self.conversation_messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
    async def process_message(self, user_input: str) -> ResultT:
        """Process a single message using agent graph control.
        
        Args:
            user_input: The user's input message
            
        Returns:
            The agent's response
        """
        if not self.is_conversation_active:
            raise RuntimeError("Conversation is not active. Call start_conversation() first.")
            
        try:
            # Store initial metrics
            initial_step_count = self.step_count
            
            if self.use_persistent_session and self._agent_run:
                result = await self._process_with_graph_control(user_input)
            else:
                # Fallback to message history approach
                result = await self._process_with_message_history(user_input)
            
            # Calculate steps taken in this operation
            steps_taken = self.step_count - initial_step_count
            
            # Store metrics on the result if possible
            if hasattr(result, '__dict__'):
                result._steps_taken = steps_taken
                result._agent_run = self.agent_run if hasattr(self, 'agent_run') else None
            
            # Add both user input and agent response to conversation history
            self._add_to_conversation("user", user_input)
            self._add_to_conversation("assistant", result.message if hasattr(result, 'message') else str(result))
            
            return result
            
        except Exception as e:
            # Add error to conversation
            self._add_to_conversation("error", str(e))
            raise
    
    async def _process_with_graph_control(self, user_input: str) -> ResultT:
        """Process input using manual agent graph control."""
        if not self._agent_run:
            raise RuntimeError("Agent run not available")
        
        try:
            if self.verbose:
                print(f"Processing user input with graph control: {user_input[:50]}...")
            
            # Since we can't easily inject new user input into an existing run,
            # we'll restart the agent run with the conversation context
            # This is a compromise solution that still provides benefits over the original approach
            
            # Build conversation context
            conversation_context = self._build_conversation_context(user_input)
            
            # Close the current run and start a new one with context
            await self._restart_agent_run(conversation_context)
            
            # Process the agent graph manually to detect end nodes
            partial_result = None
            steps_in_this_run = 0
            
            async for node in self._agent_run:
                if self.verbose:
                    print(f"Processing node: {type(node).__name__}")
                
                # Count steps (nodes that represent actual processing)
                if hasattr(node, '__class__') and 'CallTools' in node.__class__.__name__:
                    steps_in_this_run += 1
                
                # Check if this is an end node using the inherited method
                if self.is_end_node(node):
                    # Handle partial result from end node
                    partial_result = await self._handle_partial_result(node)
                    
                    if self.verbose:
                        print("✅ Detected end node - captured partial result, ready for next input...")
                    
                    # Don't break - we want to be ready for the next input
                    # The agent run is now ready to continue
                    break
            
            # Update step count
            self.step_count += steps_in_this_run
            
            if partial_result is not None:
                self._last_result = partial_result
                
                # Store metrics on the result
                if hasattr(partial_result, '__dict__'):
                    partial_result._steps_taken = steps_in_this_run
                    partial_result._agent_run = self._agent_run
                    
                    # Try to get usage immediately and store it
                    try:
                        if self._agent_run and hasattr(self._agent_run, 'usage'):
                            usage = self._agent_run.usage()
                            if usage:
                                partial_result._token_usage = {
                                    'total': usage.total_tokens,
                                    'input': usage.input_tokens,
                                    'output': usage.output_tokens
                                }
                    except Exception:
                        pass
                
                return partial_result
            else:
                # If we didn't get a partial result, create a default response
                default_result = await self._create_default_response()
                
                # Store metrics on the default result
                if hasattr(default_result, '__dict__'):
                    default_result._steps_taken = steps_in_this_run
                    default_result._agent_run = self._agent_run
                    
                    # Try to get usage immediately and store it
                    try:
                        if self._agent_run and hasattr(self._agent_run, 'usage'):
                            usage = self._agent_run.usage()
                            if usage:
                                default_result._token_usage = {
                                    'total': usage.total_tokens,
                                    'input': usage.input_tokens,
                                    'output': usage.output_tokens
                                }
                    except Exception:
                        pass
                
                return default_result
                    
        except Exception as e:
            if self.verbose:
                print(f"Error in graph control, falling back: {e}")
            # Fall back to message history approach
            return await self._process_with_message_history(user_input)
    
    def is_end_node(self, node: Any) -> bool:
        """Detect if a node is an end node."""
        # Use the inherited Agent.is_end_node method
        return super().is_end_node(node)
    
    async def _restart_agent_run(self, conversation_context: str) -> None:
        """Restart the agent run with new context."""
        try:
            # Close the current context
            if self._agent_run_context:
                await self._agent_run_context.__aexit__(None, None, None)
            
            # Start a new context with the conversation context
            self._agent_run_context = self.iter(conversation_context)
            self._agent_run = await self._agent_run_context.__aenter__()
            
        except Exception as e:
            if self.verbose:
                print(f"Error restarting agent run: {e}")
            raise
    
    def _build_conversation_context(self, current_message: str) -> str:
        """Build a conversation context that includes previous messages."""
        if not self.conversation_messages:
            return current_message
            
        # Build conversation history
        context_parts = ["Previous conversation:"]
        for msg in self.conversation_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {msg['content']}")
        
        context_parts.append(f"\nCurrent message from user: {current_message}")
        context_parts.append("\nPlease respond considering the full conversation context above.")
        
        return "\n".join(context_parts)
    
    async def _handle_partial_result(self, end_node: Any) -> ResultT:
        """Handle a partial result from an end node without terminating execution."""
        try:
            # Try to get the result from the agent run first
            if hasattr(self._agent_run, 'output') and self._agent_run.output is not None:
                result_data = self._agent_run.output
                if self.verbose:
                    print(f"Extracted from agent_run.output: {type(result_data)}")
            elif hasattr(end_node, 'result') and end_node.result is not None:
                result_data = end_node.result
                if hasattr(result_data, 'output'):
                    result_data = result_data.output
                if self.verbose:
                    print(f"Extracted from end_node.result: {type(result_data)}")
            else:
                # Create a default result
                result_data = "Conversation continues..."
                if self.verbose:
                    print("Using default result")
            
            if self.verbose:
                print(f"Final result_data type: {type(result_data)}, content: {result_data}")
            
            # Convert to our output type
            if isinstance(result_data, self.output_type):
                return result_data
            elif hasattr(self.output_type, 'model_validate') and isinstance(result_data, dict):
                return self.output_type.model_validate(result_data)
            elif hasattr(self.output_type, 'model_validate'):
                # Try to create a dict with the result
                return self.output_type.model_validate({"message": str(result_data)})
            else:
                # Try to create directly
                try:
                    return self.output_type(message=str(result_data))
                except:
                    return self.output_type()
                    
        except Exception as e:
            if self.verbose:
                print(f"Error handling partial result: {e}")
            
            return await self._create_default_response()
    
    async def _create_default_response(self) -> ResultT:
        """Create a default response when no result is available."""
        try:
            if hasattr(self.output_type, 'model_validate'):
                return self.output_type.model_validate({"message": "Response received - ready for next input"})
            else:
                return self.output_type(message="Response received - ready for next input")
        except:
            return self.output_type()
    
    async def _process_with_message_history(self, user_input: str) -> ResultT:
        """Fallback method using message history."""
        if self.verbose:
            print("Using message history fallback approach")
        
        # Build conversation context
        conversation_context = self._build_conversation_context(user_input)
        
        # Simple run with conversation context
        result = await self.run(conversation_context)
        
        # Extract the output from the AgentRunResult and store metrics
        if hasattr(result, 'output'):
            output = result.output
            
            # Store metrics on the output
            if hasattr(output, '__dict__'):
                # Try to get usage from the result
                if hasattr(result, 'usage') and callable(result.usage):
                    try:
                        usage = result.usage()
                        if usage:
                            output._token_usage = {
                                'total': usage.total_tokens,
                                'input': usage.request_tokens,
                                'output': usage.response_tokens
                            }
                    except:
                        pass
                
                # Store a reference to get usage later
                output._agent_run_result = result
            
            return output
        else:
            return result
        
    async def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.conversation_messages
        
    async def clear_conversation_history(self) -> None:
        """Clear the conversation history but keep the conversation active."""
        self.conversation_messages = []
        self.memory.reset()
        
    @property
    def conversation_uptime(self) -> Optional[float]:
        """Get the conversation's uptime in seconds."""
        if not self.conversation_start_time:
            return None
        return (datetime.now() - self.conversation_start_time).total_seconds()
        
    @property
    def is_active(self) -> bool:
        """Check if the conversation is currently active."""
        return self.is_conversation_active
    
    @property
    def is_using_persistent_session(self) -> bool:
        """Check if the agent is using persistent session with graph control."""
        return self.use_persistent_session and self._agent_run is not None 