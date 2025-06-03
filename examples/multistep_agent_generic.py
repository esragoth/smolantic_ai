from dotenv import load_dotenv
import os
from typing import List, Dict
# Force reload of .env file
load_dotenv(override=True)

import asyncio
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
from smolantic_ai.config import settings_manager
from smolantic_ai.models import Node, Step, pretty_print_node
from smolantic_ai.agent import AgentInfo

class GoogleSearchResult(BaseModel):
    title: str
    url: str

# Re-add the FinalAnswer class definition
class FinalAnswer(BaseModel):
     tokyo_time: str
     tokyo_weather: str
     google_search: List[GoogleSearchResult]  # List of objects with fields title and url



def node_callback(node: Node, agent_info: AgentInfo):
    print(pretty_print_node(node))

def step_callback(step: Step, agent_info: AgentInfo):
    print(f"Step: {step.step_type}")

async def main():
    # Force reload settings
    settings_manager.reload()
    
    # Create agent with explicit model settings
    agent = MultistepAgent(
        output_type=FinalAnswer,
        tools=[get_weather_tool, search_google_tool, timezone_tool],
        planning_interval=3,
        model="openai:gpt-4.1",
        node_callback=node_callback,
    )

    # Define the task for the agent
    task = (
        "What is the current time and weather in Tokyo? Also, find the top 3 Google search results for "
        "'latest advancements in renewable energy'. Summarize the findings."
    )

    print("\n--- Running MultiStep Agent ---")
    print(f"Task: {task}")
    print("-" * 34)
    print("Note: Ensure API keys for weather, search, etc. are set in prebuilt_tools.py")
    print("-" * 34)
    print()

    # Run the agent
    try:
        # The agent will use planning and the provided tools to solve the task
        result = await agent.run(task)
        print("\nFinal Answer:")
        print("Tokyo Time: ", result.tokyo_time)
        print("Tokyo Weather: ", result.tokyo_weather)
        for search_result in result.google_search:
            print(f"Title: {search_result.title}, URL: {search_result.url}")
        # Print memory contents for verification
        print("\n--- Agent Memory Steps ---")
        for i, step in enumerate(agent.memory.action_steps):
            print(f"Step {i+1} ({type(step).__name__}):\n{step.to_string_summary()}\n")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 