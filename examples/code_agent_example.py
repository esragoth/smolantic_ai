import asyncio
import os
from dotenv import load_dotenv
from smolantic_ai import CodeAgent
from smolantic_ai.logging import get_logger

# Configure logging
logger = get_logger()

# Load environment variables from .env file
load_dotenv()

async def main():

    # Initialize the CodeAgent
    # Uses the default model specified in settings or .env
    # Uses the 'local' executor by default
    agent = CodeAgent(
        # You can specify a model like: model="gpt-4o-mini"
        # You can specify executor_type: "local", "e2b", "docker"
        authorized_imports=["math"], # Example: Allow the 'math' library
        planning_interval=3
    )

    # Define the task for the agent
    task = (
        "Write a Python function that takes a list of numbers and returns a new list "
        "containing only the even numbers. Then, write a second function that calculates "
        "the sum of the even numbers from the list. Finally, provide an explanation of how "
        "both functions work."
    )

    print(f"Running CodeAgent with task: '{task}'")
    print("-" * 30)

    try:
        # Run the agent
        result = await agent.run(task)

        # Print the results
        print("\n" + "=" * 30)
        print("Code Agent Result:")
        print(f"Explanation: {result.explanation}")
        print(f"Code:\n```python\n{result.code}\n```")
        print(f"Execution Result: {result.result}")
        print("=" * 30)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 