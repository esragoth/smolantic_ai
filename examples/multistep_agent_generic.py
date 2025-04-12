from dotenv import load_dotenv

# Load environment variables from .env file FIRST
dotenv_loaded = load_dotenv()

import os # Import os here for debugging

# Load environment variables from .env file FIRST - Removed, relying on pydantic-settings
# dotenv_loaded = load_dotenv()

import asyncio
import os
from pydantic import BaseModel
from pydantic_ai import Tool
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from smolantic_ai.multistep_agent import MultistepAgent
from smolantic_ai.prebuilt_tools import (
    get_weather_tool,
    search_google_tool,
    timezone_tool,
)
from smolantic_ai.config import settings as config

# --- DEBUG: Print loaded model config ---
print(f"DEBUG: Loaded Model Provider: {config.model_provider}")
print(f"DEBUG: Loaded Model Name: {config.model_name}")
# --- END DEBUG ---

# Re-add the FinalAnswer class definition
class FinalAnswer(BaseModel):
     summary_text: str


async def main():
    # Define the specific tools the agent needs for this task
    tools = [
        get_weather_tool,
        timezone_tool,
        search_google_tool,
    ]

   
    # Instantiate the MultistepAgent with FinalAnswer as result type
    agent = MultistepAgent[FinalAnswer]( # Use FinalAnswer as result type
        tools=tools,
        result_type=FinalAnswer,
        # You can optionally specify the model, planning interval, max steps, etc.
        model=config.model_string, 
        planning_interval=3, 
        max_steps=15,
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
        result: FinalAnswer = await agent.run(task) # Result type is still FinalAnswer
        print(f"\n--- Agent Finished ---")
        print(f"Final Summary:\n{result.summary_text}") # Access the renamed field
        print("----------------------")

    except Exception as e:
        print(f"\n--- Agent Error ---")
        print(f"An error occurred during agent execution: {e}")
        print("-------------------")

if __name__ == "__main__":
    asyncio.run(main()) 