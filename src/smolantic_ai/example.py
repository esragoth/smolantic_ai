import asyncio
from pydantic_ai import Tool
from .agent import CodeAgent, ToolCallingAgent

async def example_tool_calling_agent():
    """Example using the ToolCallingAgent with custom tools."""
    # Define custom tools
    tools = [
        Tool(
            name="search_web",
            description="Search the web for information",
            function=lambda query: f"Search results for: {query}"
        ),
        Tool(
            name="calculate",
            description="Perform mathematical calculations",
            function=lambda expression: eval(expression)
        )
    ]
    
    agent = ToolCallingAgent(tools=tools)
    
    # Run synchronously
    result = agent.run_sync("What is the square root of 16 plus 5?")
    print(f"\nToolCallingAgent sync response:")
    print(f"Result: {result.data.result}")
    print(f"Explanation: {result.data.explanation}")
    
    # Run asynchronously
    result = await agent.run("Search for information about Python and calculate 2^3")
    print(f"\nToolCallingAgent async response:")
    print(f"Result: {result.data.result}")
    print(f"Explanation: {result.data.explanation}")

async def example_code_agent():
    """Example using the CodeAgent with structured results."""
    agent = CodeAgent()
    
    # Run synchronously
    result = agent.run_sync("Create a function to calculate the factorial of a number.")
    print(f"\nCodeAgent sync response:")
    print(f"Code:\n{result.data.code}")
    print(f"Result: {result.data.result}")
    print(f"Explanation: {result.data.explanation}")
    
    # Run asynchronously
    result = await agent.run("Create a function to find the nth Fibonacci number.")
    print(f"\nCodeAgent async response:")
    print(f"Code:\n{result.data.code}")
    print(f"Result: {result.data.result}")
    print(f"Explanation: {result.data.explanation}")

async def main():
    print("Running ToolCallingAgent examples:")
    await example_tool_calling_agent()
    
    print("\nRunning CodeAgent examples:")
    await example_code_agent()

if __name__ == "__main__":
    asyncio.run(main()) 