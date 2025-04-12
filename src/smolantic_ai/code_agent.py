# TODO: Verify compatibility and integration with the refactored MultistepAgent base class.
from typing import Any, Optional, List, Union, Dict
from pydantic import BaseModel, Field
from pydantic_ai import Tool
from .multistep_agent import MultistepAgent
from .executors import PythonExecutor, LocalPythonExecutor, E2BExecutor, DockerExecutor, CodeExecutionResult
from .config import settings
from .prompts import CODE_AGENT_SYSTEM_PROMPT, CODE_AGENT_PLANNING_INITIAL
from .logging import get_logger

class CodeResult(BaseModel):
    """Result model for code operations."""
    code: str = Field(description="The generated code")
    result: str = Field(description="The execution result as a string")
    explanation: str = Field(description="Explanation of what the code does")

class CodeAgent(MultistepAgent[CodeResult]):
    """Agent specialized for code generation and execution."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        executor_type: str = "local",
        authorized_imports: Optional[List[str]] = None,
        max_print_outputs_length: Optional[int] = None,
        planning_interval: Optional[int] = None,
        logger_name: Optional[str] = None,
        **kwargs
    ):
        # Create the final_answer tool
        def final_answer(code: str, result: str, explanation: str) -> CodeResult:
            return CodeResult(code=code, result=result, explanation=explanation)
        
        tools = [
            Tool(
                name="final_answer",
                function=final_answer,
                description="Return the final answer with code, result, and explanation"
            )
        ]
        
        # Initialize base agent with tools
        super().__init__(
            model=model or settings.code_model.model_string,
            result_type=CodeResult,
            tools=tools,
            system_prompt=CODE_AGENT_SYSTEM_PROMPT,
            planning_prompt=CODE_AGENT_PLANNING_INITIAL,
            planning_interval=planning_interval,
            **kwargs
        )
        
        # Set properties after initialization
        self.executor = self._create_executor(
            executor_type,
            authorized_imports or [],
            max_print_outputs_length
        )
        self.logger = get_logger(logger_name)

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

    def _extract_code_from_response(self, response: Any) -> tuple[str, str, str]:
        """Extract code, result, and explanation from the model's response."""
        self.logger.log_action({"action": "extract_code", "response_type": type(response).__name__})
        
        # Handle AgentRunResult
        if hasattr(response, 'data') and isinstance(response.data, CodeResult):
            return response.data.code, response.data.result, response.data.explanation
        
        if isinstance(response, CodeResult):
            return response.code, response.result, response.explanation
        elif isinstance(response, dict):
            return (
                response.get("code", ""),
                response.get("result", ""),
                response.get("explanation", "")
            )
        elif isinstance(response, str):
            # Try to extract code from the response string
            # Look for code between ```py and ``` markers
            import re
            code_match = re.search(r"```py\n(.*?)```", response, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
                # Extract any text before the code block as explanation
                explanation = response[:code_match.start()].strip()
                return code, "", explanation
            # Look for code after "Code:" marker
            code_match = re.search(r"Code:\s*(.*?)(?:Observation:|$)", response, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
                # Extract any text before "Code:" as explanation
                explanation = response[:code_match.start()].strip()
                return code, "", explanation
        
        raise ValueError(f"Could not extract code from response: {response}")

    async def _process_response(self, response: Any) -> CodeResult:
        """Process the model's response into a CodeResult."""
        try:
            # If response is already a CodeResult wrapped in AgentRunResult, return it
            if hasattr(response, 'data') and isinstance(response.data, CodeResult):
                return response.data
            
            code, result, explanation = self._extract_code_from_response(response)
            
            # Execute the code if we don't have a result yet
            if not result:
                try:
                    self.logger.log_action({"action": "execute_code", "code": code})
                    execution_result = self.executor(code)
                    result = execution_result.execution_logs if execution_result.execution_logs else str(execution_result.output)
                    self.logger.log_result({"execution_result": result})
                except Exception as e:
                    result = f"Error executing code: {str(e)}"
                    self.logger.log_error(e)
            
            # Ensure we have valid values for all fields
            code = code or ""
            result = result or ""
            explanation = explanation or "No explanation provided"
            
            # Create and validate the CodeResult
            try:
                return CodeResult(
                    code=code,
                    result=result,
                    explanation=explanation
                )
            except Exception as e:
                self.logger.log_error(f"Error creating CodeResult: {e}")
                # Return a minimal valid CodeResult
                return CodeResult(
                    code="",
                    result=str(response),
                    explanation="Error processing response"
                )
                
        except Exception as e:
            # If we can't process the response, create a CodeResult with an error
            self.logger.log_error(e)
            return CodeResult(
                code="",
                result=f"Error processing response: {str(e)}",
                explanation=str(response) if response else ""
            )

    async def run(self, task: str, **kwargs) -> CodeResult:
        """Run the agent with the given task."""
        response = await super().run(task, **kwargs)
        return await self._process_response(response) 