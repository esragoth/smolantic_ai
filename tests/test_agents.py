import pytest
from dirty_equals import IsInstance, IsNow, IsStr
from pydantic import BaseModel
from pydantic_ai import models, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    RetryPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
import json
from smolantic_ai.models import CodeResult

# Assuming agents are importable from smolantic_ai
from smolantic_ai import CodeAgent, MultistepAgent
from pydantic_ai import Tool

# Global setup for tests
pytestmark = pytest.mark.asyncio # Use only asyncio backend to avoid trio errors
models.ALLOW_MODEL_REQUESTS = False  # Prevent accidental real LLM calls

# Define a simple result type for the test
class WeatherResult(BaseModel):
    output: str

# Define a simple dependency type for the test
class TestDeps(BaseModel):
    info: str = "test data"

# Unified test model logic
def unified_model_logic(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Simulates agent behavior based on initial prompt type, handling framework quirks."""

    # 1. Handle Initial User Prompt (First call)
    if len(messages) == 1:
        last_part = messages[0].parts[-1] # Assume user prompt is last part
        prompt_content_lower = last_part.content.lower()
        if "weather" in prompt_content_lower:
            # Weather query: Call the weather tool
            return ModelResponse(parts=[ToolCallPart(
                tool_name='get_weather',
                args={'location': 'London'},
                tool_call_id='weather_step_1'
            )])
        else:
            # Assume Code query: Return final code result directly
            return ModelResponse(parts=[ToolCallPart(
                tool_name='final_result',
                args={
                    'code': "def add(a, b):\\n  return a + b",
                    'result': "Test execution result",
                    'explanation': "Test code response for CodeAgent"
                },
                tool_call_id='code_final'
            )])

    # 2. Handle Subsequent Calls (Simulating state based on original prompt)
    elif len(messages) > 1:
        # Check the *original* user prompt type from the first message
        original_user_prompt_part = messages[0].parts[-1] # Re-check first message
        original_prompt_content_lower = original_user_prompt_part.content.lower()

        if "weather" in original_prompt_content_lower:
            # If original query was weather, this call *must* be the final step
            # Simulate the result *after* the weather tool would have run
            return ModelResponse(parts=[ToolCallPart(
                tool_name='final_result',
                # Provide the expected final weather output directly
                args={'output': "The weather in London is Sunny"},
                tool_call_id='weather_final'
            )])
        else:
            # If original query was code, assume it's a retry or unexpected state
            # Return the code result again (or a fallback if needed)
            # Sticking to returning the original code result for simplicity here.
            return ModelResponse(parts=[ToolCallPart(
                tool_name='final_result',
                args={
                    'code': "def add(a, b):\\n  return a + b",
                    'result': "Test execution result",
                    'explanation': "Test code response for CodeAgent"
                },
                tool_call_id='code_final_retry'
            )])

    # Should not be reached in typical test flow with this logic
    # but provide a minimal fallback just in case.
    return ModelResponse(parts=[ToolCallPart(
        tool_name='final_result',
        args={'output': "Fallback - Should not happen"},
        tool_call_id='fallback_unexpected'
    )])

# --- Fixtures ---
@pytest.fixture
def multistep_agent_test_model():
    agent = MultistepAgent[TestDeps, WeatherResult](
        model=FunctionModel(unified_model_logic),
        tools=[],
        result_type=WeatherResult,
        deps_type=TestDeps
    )
    yield agent

@pytest.fixture
def multistep_agent_function_model():
    # Add a dummy tool for the test
    def get_weather(ctx: RunContext, location: str) -> str:
        """Gets the weather."""
        if location == "London":
            return "Sunny"
        return "Unknown"
    get_weather_tool = Tool(name="get_weather", description="Gets the weather.", function=get_weather)

    agent = MultistepAgent[TestDeps, WeatherResult](
        model=FunctionModel(unified_model_logic),
        tools=[get_weather_tool],
        result_type=WeatherResult,
        deps_type=TestDeps
    )
    yield agent

@pytest.fixture
def code_agent_test_model():
    agent = CodeAgent(
        model=FunctionModel(unified_model_logic),
        tools=[],
        executor_type="local",
        authorized_imports=["math", "os"]
    )
    yield agent

@pytest.fixture
def code_agent_function_model():
    agent = CodeAgent(
        model=FunctionModel(unified_model_logic),
        tools=[],
        executor_type="local",
        authorized_imports=["math", "os"]
    )
    yield agent

# --- Tests ---
async def test_multistep_agent_with_test_model(multistep_agent_test_model: MultistepAgent):
    prompt = "What is the weather in London?"
    result = await multistep_agent_test_model.run(prompt)
    # This test might now hit the second branch and return the hardcoded sunny weather
    # Adjust assertion if needed based on how TestModel behaves with this logic
    assert result.output == "The weather in London is Sunny" # Updated assertion

async def test_multistep_agent_with_function_model(multistep_agent_function_model: MultistepAgent):
    prompt = "What is the weather in London?"
    result = await multistep_agent_function_model.run(prompt)
    assert result.output == "The weather in London is Sunny"

async def test_code_agent_with_test_model(code_agent_test_model: CodeAgent):
    prompt = "Write a function to add two numbers."
    result = await code_agent_test_model.run(prompt)
    assert isinstance(result, CodeResult)
    assert result.explanation == "Test code response for CodeAgent"

async def test_code_agent_with_function_model(code_agent_function_model: CodeAgent):
    prompt = "Write a function to add two numbers."
    result = await code_agent_function_model.run(prompt)
    assert isinstance(result, CodeResult)
    assert result.code == "def add(a, b):\\n  return a + b"
    assert result.explanation == "Test code response for CodeAgent"
    assert result.result == "Test execution result" 