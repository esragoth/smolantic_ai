import asyncio
import math
from pydantic_ai import Tool
from ..agent import MultistepAgent, ToolCallingAgent, CodeAgent
from ..models import ToolResult, ToolCallingResult, CodeExecutionResult
from typing import Dict, Any

# Define some simple tools
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

def get_weather(city: str) -> str:
    """Get the weather for a city."""
    # This is a mock implementation
    return f"The weather in {city} is sunny and 25°C"

# Create tool objects
tools = [
    Tool(
        name="add_numbers",
        description="Add two numbers together",
        function=add_numbers
    ),
    Tool(
        name="multiply_numbers",
        description="Multiply two numbers together",
        function=multiply_numbers
    ),
    Tool(
        name="get_weather",
        description="Get the weather for a city",
        function=get_weather
    )
]

async def example_multistep_agent():
    """Example using the MultistepAgent with planning and execution."""
    # Define custom tools
    def search_web(query: str) -> str:
        return f"Search results for: {query}"
    
    def calculate(expression: str) -> float:
        # Add math functions to the evaluation context
        globals_dict = {"math": math, "sqrt": math.sqrt}
        return eval(expression, globals_dict)
    
    tools = [
        Tool(
            name="search_web",
            description="Search the web for information",
            function=search_web
        ),
        Tool(
            name="calculate",
            description="Perform mathematical calculations",
            function=calculate
        )
    ]
    
    agent = MultistepAgent(
        model="openai:gpt-4",
        tools=tools,
        planning_interval=3  # Re-plan every 3 steps
    )
    
    # Run asynchronously
    result = await agent.run("What is the square root of 16 plus 5?")
    print(f"\nMultistepAgent response:")
    print(f"Plan: {result.data.planning.plan}")
    print(f"Steps: {result.data.planning.steps}")
    print(f"Final Answer: {result.data.final_answer.answer}")
    print(f"Explanation: {result.data.final_answer.explanation}")

async def example_tool_calling_agent():
    """Example using the ToolCallingAgent with custom tools."""
    # Define custom tools
    def search_web(query: str) -> str:
        return f"Search results for: {query}"
    
    def calculate(expression: str) -> float:
        # Add math functions to the evaluation context
        globals_dict = {
            "math": math,
            "sqrt": math.sqrt,
            "pow": math.pow,  # Use math.pow instead of the ^ operator
            "**": pow  # Support Python's ** operator
        }
        return eval(expression, globals_dict)
    
    tools = [
        Tool(
            name="search_web",
            description="Search the web for information",
            function=search_web
        ),
        Tool(
            name="calculate",
            description="Perform mathematical calculations",
            function=calculate
        )
    ]
    
    agent = ToolCallingAgent(
        model="openai:gpt-4",
        tools=tools
    )
    
    # Run first example
    result = await agent.run("What is the square root of 16 plus 5?")
    print(f"\nToolCallingAgent response:")
    print(f"Result: {result.data.result}")
    print(f"Explanation: {result.data.explanation}")
    
    # Run second example
    result = await agent.run("Search for information about Python and calculate 2^3 using pow")
    print(f"\nToolCallingAgent second response:")
    print(f"Result: {result.data.result}")
    print(f"Explanation: {result.data.explanation}")

async def example_code_agent():
    """Example using the CodeAgent with local executor."""
    # Create agent with local executor
    local_agent = CodeAgent(
        model="openai:gpt-4",
        executor_type="local",
        authorized_imports=["math"]
    )
    
    # Run with local executor
    result = await local_agent.run("Create a function to calculate the factorial of a number.")
    print(f"\nCodeAgent response:")
    print(f"Code:\n{result.data.code}")
    print(f"Result: {result.data.result}")
    print(f"Explanation: {result.data.explanation}")

async def main():
    print("Running MultistepAgent example:")
    await example_multistep_agent()
    
    print("\nRunning ToolCallingAgent examples:")
    await example_tool_calling_agent()
    
    print("\nRunning CodeAgent example:")
    await example_code_agent()

if __name__ == "__main__":
    asyncio.run(main()) 