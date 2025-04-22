import asyncio
import os
from dotenv import load_dotenv
from smolantic_ai import CodeAgent
from smolantic_ai.logging import get_logger
from smolantic_ai.prebuilt_tools import timezone_tool, search_google_tool, read_webpage_tool
from pydantic import BaseModel, Field
# Configure logging
logger = get_logger()

class FinalAnswer(BaseModel):
    answer: str = Field(description="The final answer to the user's question")
    explanation: str = Field(description="Explanation of how the answer was derived")

# Load environment variables from .env file
load_dotenv()

def node_callback(node, agent_info):
    print(f"ID: {agent_info.get('id')}")
    print(f"Name: {agent_info.get('name')}")
    print(f"Current Step: {agent_info.get('current_step_count')}")
    print(node.to_string_summary())

def step_callback(step, agent_info):
    print(f"Step: {step.id}")
    print(f"Agent Info: {agent_info}")

async def main():

    # Initialize the CodeAgent
    # Uses the default model specified in settings or .env
    # Uses the 'local' executor by default
    agent = CodeAgent(
        # You can specify a model like: model="gpt-4o-mini"
        # You can specify executor_type: "local", "e2b", "docker"
        authorized_imports=["math", "requests", "json", "datetime", "os", "re", "matplotlib"],
        planning_interval=3,
        result_type=FinalAnswer,
        tools=[
            timezone_tool,
            search_google_tool,
            read_webpage_tool,
        ],
        node_callback=node_callback,
    )

    # Define the task for the agent
    task = (
        "Draw and save a bar chart of the top 10 most popular programming languages in the world. Get current accurate data from the internet."
    )

    print(f"Running CodeAgent with task: '{task}'")
    print("-" * 30)

    try:
        # Run the agent
        result = await agent.run(task)

        # Print the results
        print("\n" + "=" * 30)
        print("Code Agent Result:")
        print(f"Answer: {result.answer}")
        print(f"Explanation: {result.explanation}")
        print("=" * 30)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 