# multistep_agent_delegation_story.py

import asyncio
import os
from typing import List, Dict
from dotenv import load_dotenv

# Assuming your project structure allows these imports
# Adjust paths if necessary, e.g., if running from the root directory:
# from src.smolantic_ai.multistep_agent import MultistepAgent
# from src.smolantic_ai.config import settings_manager
# from src.smolantic_ai.logging import get_logger
# If running from within src/smolantic_ai:
from smolantic_ai.multistep_agent import MultistepAgent
from smolantic_ai.config import settings_manager
from smolantic_ai.logging import get_logger


from pydantic_ai import RunContext, Tool
from pydantic_ai.usage import Usage
# MultistepAgent handles UsageLimits internally via request_limit parameter

# --- Basic Setup ---
load_dotenv() # Load environment variables (e.g., API keys from .env)
settings_manager.reload() # Load configuration (e.g., from config.yaml)
logger = get_logger(__name__)

# --- Define the Delegate Agent (using MultistepAgent) ---
# This agent specializes in generating chapter titles
chapter_title_agent = MultistepAgent[None, List[str]](
    # Use a potentially faster/cheaper model for this focused task
    model=f"{settings_manager.settings.model_provider}:{settings_manager.settings.model_name}", # Example: 'google:gemini-1.5-flash-latest'
    result_type=List[str],
    system_prompt='Generate creative chapter titles based on the provided topic and count. Return only the list of titles.',
    request_limit=5, # Limit requests for this agent
    logger_name="ChapterTitleAgent",
)

# --- Define the Story Writer Agent (New) ---
story_writer_agent = MultistepAgent[None, str](
    model="openai:gpt-4.1",
    result_type=str,
    system_prompt='Write a short, engaging story paragraph (approx. 50-100 words) for the given chapter title. Return only the story paragraph text.',
    request_limit=5,
    logger_name="StoryWriterAgent",
    planning_interval=None,
)

# --- Define the Tool for the Parent Agent ---
# This tool will call the delegate agent
async def chapter_title_factory(ctx: RunContext[None], topic: str, count: int) -> List[str]:
    """Generates a specified number of chapter titles for a given topic using a specialized agent."""
    logger.info(f"--- Delegating to chapter_title_agent for topic '{topic}' ---")
    # Call the delegate agent's run method, which now returns AgentRunResult
    # We need to extract the data from the result object.
    run_result = await chapter_title_agent.run(
        f'Generate {count} chapter titles about: {topic}',
        usage=ctx.usage
    )
    
    # Extract the data
    titles = run_result.data
    
    if isinstance(titles, list):
        logger.info(f"--- Received {len(titles)} titles from chapter_title_agent ---")
        return titles
    else:
        logger.error(f"--- chapter_title_agent returned unexpected type: {type(titles)} ---")
        # Handle error case appropriately, e.g., return an empty list or raise exception
        return ["Error generating titles"]

# --- Define the Tool for Writing Stories (New) ---
async def story_writer_factory(ctx: RunContext[None], chapter_title: str) -> str:
    """Writes a short story paragraph for the given chapter title using a specialized agent."""
    logger.info(f"--- Delegating to story_writer_agent for title '{chapter_title}' ---")
    run_result = await story_writer_agent.run(
        f'Write a story paragraph for the chapter titled: "{chapter_title}"' ,
        usage=ctx.usage
    )
    story = run_result.data
    if isinstance(story, str):
        logger.info(f"--- Received story paragraph from stoty_writer_agent ---")
        return story
    else:
        logger.error(f"--- story_writer_agent returned unexpected type: {type(story)} ---")
        return f"Error generating story for: {chapter_title}"

# --- Define the Parent Agent (using MultistepAgent) ---
# This agent plans the story and uses the tool to get chapter titles
# Convert the async functions to Tool instances
chapter_title_tool = Tool(chapter_title_factory)
story_writer_tool = Tool(story_writer_factory) # Create tool instance for the new function

story_planner_agent = MultistepAgent[None, Dict[str, str]]( # Expect Dict[title, story] as result
    model=f"{settings_manager.settings.model_provider}:{settings_manager.settings.model_name}", # Example: 'openai:gpt-4o-mini'
    result_type=Dict[str, str], # Updated result type
    system_prompt=(
        'You are a master story outline creator.\n'
        '1. First, use the `chapter_title_factory` tool to generate the required number of chapter titles for the given story topic.\n'
        '2. Then, for EACH generated chapter title, use the `story_writer_factory` tool to write a short story paragraph.\n'
        '3. Finally, return ONLY a dictionary where keys are the chapter titles and values are the corresponding story paragraphs.'
    ),
    tools=[chapter_title_tool, story_writer_tool], # Pass BOTH tool instances here
    request_limit=15, # Adjust limit for potentially more calls
    planning_interval=None, # Disable planning for this simple task
    logger_name="StoryPlannerAgent",
)


# --- Main Execution Logic ---
async def main():
    logger.info("Running Multistep Agent Delegation Example...")
    
    # Define the task combining title generation and story writing
    task = 'Create 3 chapter titles for a fantasy novel about a lost dragon egg and write a short story paragraph for each title.'

    try:
        # Run the main planner agent
        # MultistepAgent's run now returns AgentRunResult
        run_result = await story_planner_agent.run(task) # Use the updated task

        # Extract the data (expected to be a dictionary)
        story_chapters = run_result.data
        
        print("\nGenerated Story Outline:")
        if isinstance(story_chapters, dict):
            for title, story in story_chapters.items():
                print("\n--- Chapter --- ")
                print(f"Title: {title}")
                print(f"Story: {story}")
                print("---------------")
        else:
            print("Failed to generate story outline or unexpected output format.")
            print("Received:", story_chapters)

        # Print the final accumulated usage
        print("\nFinal Usage:")
        print(run_result.usage())

        # Note: Accessing detailed usage might require inspecting agent instances
        # or enhancing MultistepAgent if cross-agent usage tracking is needed.
        # logger.info(f"Story Planner Usage: {story_planner_agent.some_usage_property}") # If available
        # logger.info(f"Chapter Title Usage: {chapter_title_agent.some_usage_property}") # If available

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        print(f"An error occurred: {e}")
        print("Please ensure your API keys (e.g., OPENAI_API_KEY) are set in a .env file or environment variables.")
        print("Also check model names in your config and required packages: pip install -r requirements.txt") # Assuming requirements.txt exists

if __name__ == "__main__":
    # Ensure necessary environment variables are set (e.g., OPENAI_API_KEY)
    # and config points to valid models.
    # Check for multiple potential keys
    api_key_found = any(os.getenv(key) for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"])
    if not api_key_found:
         print("Warning: No API key found for OpenAI, Anthropic, or Google in environment variables.")
         print("Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY in your .env file or environment.")

    asyncio.run(main()) 