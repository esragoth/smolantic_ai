import asyncio
import os
from pydantic import BaseModel
from pydantic_ai import Tool
from smolantic_ai.multistep_agent import MultistepAgent
from smolantic_ai.prebuilt_tools import (
    get_weather_tool,
    search_google_tool,
    timezone_tool,
)
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define a simple result model using a Tool for the final answer
class FinalAnswer(BaseModel):
    summary: str

def final_answer(summary: str) -> FinalAnswer:
    """Return the final summary or answer to the user."""
    return FinalAnswer(summary=summary)

# Define the custom final answer tool for this example
final_answer_tool = Tool(
    name="final_answer",
    description="Provide the final summary or answer when the task is complete.",
    function=final_answer,
)

async def main():
    # Define the specific tools the agent needs for this task
    tools = [
        get_weather_tool,
        timezone_tool,
        search_google_tool,
        final_answer_tool, # Use the custom final answer tool
    ]

    # Ensure the necessary API key for the LLM is set in the environment
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it in your .env file or environment.")
        return

    # Instantiate the MultistepAgent
    agent = MultistepAgent[
        FinalAnswer # Specify the expected final result type
    ](
        tools=tools,
        # You can optionally specify the model, planning interval, max steps, etc.
        # model="openai:gpt-4-turbo",
        planning_interval=3, 
        max_steps=15
    )

    # Define the task for the agent
    task = "What is the current time and weather in Tokyo? Also, find the top 3 Google search results for 'latest advancements in renewable energy'. Summarize the findings."

    print(f"--- Running MultiStep Agent ---")
    print(f"Task: {task}")
    print("----------------------------------")
    print("Note: Ensure API keys for weather, search, etc. are set in prebuilt_tools.py")
    print("----------------------------------\n")

    # Run the agent
    try:
        # The agent will use planning and the provided tools to solve the task
        result: FinalAnswer = await agent.run(task)
        print(f"\n--- Agent Finished ---")
        print(f"Final Summary:\n{result.summary}")
        print("----------------------")

    except Exception as e:
        print(f"\n--- Agent Error ---")
        print(f"An error occurred during agent execution: {e}")
        print("-------------------")

if __name__ == "__main__":
    asyncio.run(main()) 