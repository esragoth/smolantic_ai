from typing import Any, Dict, List, Optional, Tuple
import ast
import logging
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import importlib
import re
from pydantic_ai import Tool, RunContext
from .models import CodeExecutionResult

logger = logging.getLogger(__name__)

class PythonExecutor:
    """Base class for Python executors."""
    def __init__(
        self,
        authorized_imports: List[str],
        max_print_outputs_length: Optional[int] = None,
    ):
        self.authorized_imports = authorized_imports
        self.max_print_outputs_length = max_print_outputs_length
        self.state = {}

    def __call__(self, code: str, ctx: Optional[RunContext] = None) -> CodeExecutionResult:
        """
        Execute Python code and return the result.
        
        Args:
            code: The Python code to execute
            ctx: Optional RunContext for accessing tools and state
            
        Returns:
            CodeExecutionResult containing the execution results
        """
        raise NotImplementedError

    def _check_imports(self, code: str) -> None:
        """Check if the code contains any unauthorized imports or their subpackages."""
        # Remove potential start/end tags before parsing
        clean_code = code.strip()
        if clean_code.startswith("<start_code>"):
            clean_code = clean_code[len("<start_code>"):]
        if clean_code.endswith("<end_code>"):
            clean_code = clean_code[:-len("<end_code>")]
        clean_code = clean_code.strip() # Remove potential whitespace after tag removal

        tree = ast.parse(clean_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    parts = module_name.split('.')
                    is_authorized = False
                    for i in range(1, len(parts) + 1):
                        prefix = ".".join(parts[:i])
                        if prefix in self.authorized_imports:
                            is_authorized = True
                            break
                    if not is_authorized:
                        raise ImportError(f"Import of {module_name} is not allowed")
            elif isinstance(node, ast.ImportFrom):
                # Handle absolute imports (node.module is not None)
                if node.module:
                    module_name = node.module
                    parts = module_name.split('.')
                    is_authorized = False
                    for i in range(1, len(parts) + 1):
                        prefix = ".".join(parts[:i])
                        if prefix in self.authorized_imports:
                            is_authorized = True
                            break
                    if not is_authorized:
                        raise ImportError(f"Import from {module_name} is not allowed")
                # Note: Relative imports (node.module is None) are implicitly allowed by this logic.
                # If stricter control over relative imports is needed, add checks here based on node.level.

    def _truncate_output(self, output: str) -> str:
        """Truncate output if it exceeds max length."""
        if self.max_print_outputs_length and len(output) > self.max_print_outputs_length:
            return output[:self.max_print_outputs_length] + "... (truncated)"
        return output

class LocalPythonExecutor(PythonExecutor):
    """Executor that runs Python code locally."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = {
            "_print_outputs": "",
            "_final_answer": None,
            "_is_final_answer": False
        }

    def __call__(self, code: str, ctx: Optional[RunContext] = None) -> CodeExecutionResult:
        """Execute Python code locally and return the result."""
        self._check_imports(code)
        
        # Capture stdout and stderr
        stdout = StringIO()
        stderr = StringIO()
        
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                # Define the final_answer function in the local scope
                def final_answer(answer: Any):
                    self.state["_final_answer"] = answer
                    self.state["_is_final_answer"] = True
                
                # Prepare local variables
                local_vars = {
                    "final_answer": final_answer,
                    **self.state
                }
                
                # Add tools from context if available
                if ctx and ctx.tools:
                    local_vars.update({tool.name: tool.function for tool in ctx.tools})
                
                # Execute the code
                exec(code, local_vars)
                
                # Update state with any new variables
                self.state.update({
                    k: v for k, v in local_vars.items()
                    if not k.startswith("_")
                })
                
                # Get output and logs
                output = self.state["_final_answer"]
                execution_logs = self._truncate_output(stdout.getvalue() + stderr.getvalue())
                is_final_answer = self.state["_is_final_answer"]
                
                # Reset final answer state
                self.state["_final_answer"] = None
                self.state["_is_final_answer"] = False
                
                return CodeExecutionResult(
                    success=True,
                    code=code,
                    output=output,
                    execution_logs=execution_logs,
                    is_final_answer=is_final_answer
                )
                
        except Exception as e:
            error_msg = f"Error executing code: {str(e)}"
            logger.error(error_msg)
            return CodeExecutionResult(
                success=False,
                code=code,
                output=None,
                execution_logs=error_msg,
                error=error_msg
            )

class E2BExecutor(PythonExecutor):
    """Executor that runs Python code in an E2B environment."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            from e2b import Session
        except ImportError:
            raise ImportError("E2B executor requires the e2b package. Install it with: pip install e2b")
        
        self.session = Session.create("python3")
        self.state = {}

    def __call__(self, code: str, ctx: Optional[RunContext] = None) -> CodeExecutionResult:
        """Execute Python code in E2B environment and return the result."""
        self._check_imports(code)
        
        try:
            # Prepare the code with state and final_answer function
            final_answer_code = """
def final_answer(answer):
    global _final_answer
    global _is_final_answer
    _final_answer = answer
    _is_final_answer = True
"""
            state_code = "\n".join(f"{k} = {repr(v)}" for k, v in self.state.items())
            
            # Add tools from context if available
            tools_code = ""
            if ctx and ctx.tools:
                tools_code = "\n".join(
                    f"{tool.name} = {repr(tool.function)}"
                    for tool in ctx.tools
                )
            
            full_code = f"{final_answer_code}\n{state_code}\n{tools_code}\n{code}"
            
            # Execute the code
            result = self.session.run_python(full_code)
            
            # Update state with any new variables
            self.state.update(result.variables)
            
            # Get output and logs
            output = result.variables.get("_final_answer")
            execution_logs = self._truncate_output(result.stdout + result.stderr)
            is_final_answer = result.variables.get("_is_final_answer", False)
            
            return CodeExecutionResult(
                success=True,
                code=code,
                output=output,
                execution_logs=execution_logs,
                is_final_answer=is_final_answer
            )
            
        except Exception as e:
            error_msg = f"Error executing code in E2B: {str(e)}"
            logger.error(error_msg)
            return CodeExecutionResult(
                success=False,
                code=code,
                output=None,
                execution_logs=error_msg,
                error=error_msg
            )

    def __del__(self):
        """Clean up the E2B session."""
        if hasattr(self, "session"):
            self.session.close()

class DockerExecutor(PythonExecutor):
    """Executor that runs Python code in a Docker container."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            import docker
        except ImportError:
            raise ImportError("Docker executor requires the docker package. Install it with: pip install docker")
        
        self.client = docker.from_env()
        self.container = None
        self.state = {}

    def __call__(self, code: str, ctx: Optional[RunContext] = None) -> CodeExecutionResult:
        """Execute Python code in Docker container and return the result."""
        self._check_imports(code)
        
        try:
            if not self.container:
                self.container = self.client.containers.run(
                    "python:3.9",
                    command="python",
                    detach=True,
                    tty=True,
                    stdin_open=True
                )
            
            # Prepare the code with state and final_answer function
            final_answer_code = """
def final_answer(answer):
    global _final_answer
    global _is_final_answer
    _final_answer = answer
    _is_final_answer = True
"""
            state_code = "\n".join(f"{k} = {repr(v)}" for k, v in self.state.items())
            
            # Add tools from context if available
            tools_code = ""
            if ctx and ctx.tools:
                tools_code = "\n".join(
                    f"{tool.name} = {repr(tool.function)}"
                    for tool in ctx.tools
                )
            
            full_code = f"{final_answer_code}\n{state_code}\n{tools_code}\n{code}"
            
            # Execute the code
            exec_result = self.container.exec_run(
                "python -c",
                stdin=full_code,
                stdout=True,
                stderr=True
            )
            
            # Parse the output
            output = None
            execution_logs = exec_result.output.decode()
            is_final_answer = False
            
            # Extract variables from the output
            var_pattern = r"(\w+)\s*=\s*(.*?)(?=\n\w+\s*=|$)"
            for match in re.finditer(var_pattern, execution_logs):
                var_name, var_value = match.groups()
                try:
                    value = ast.literal_eval(var_value.strip())
                    self.state[var_name] = value
                    if var_name == "_final_answer":
                        output = value
                    elif var_name == "_is_final_answer":
                        is_final_answer = value
                except (ValueError, SyntaxError):
                    pass
            
            return CodeExecutionResult(
                success=True,
                code=code,
                output=output,
                execution_logs=self._truncate_output(execution_logs),
                is_final_answer=is_final_answer
            )
            
        except Exception as e:
            error_msg = f"Error executing code in Docker: {str(e)}"
            logger.error(error_msg)
            return CodeExecutionResult(
                success=False,
                code=code,
                output=None,
                execution_logs=error_msg,
                error=error_msg
            )

    def __del__(self):
        """Clean up the Docker container."""
        if self.container:
            try:
                self.container.stop()
                self.container.remove()
            except:
                pass 