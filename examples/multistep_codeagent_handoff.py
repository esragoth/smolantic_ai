# multistep_codeagent_handoff.py

import asyncio
import os
from typing import Literal, Union, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Attempt to import rich, handle if not present
try:
    from rich.prompt import Prompt
except ImportError:
    # Fallback simple input function if rich is not installed
    def Prompt_ask_fallback(prompt_text):
        return input(prompt_text + " ")
    Prompt = type('obj', (object,), {'ask': staticmethod(Prompt_ask_fallback)})()


from smolantic_ai.multistep_agent import MultistepAgent
from smolantic_ai.code_agent import CodeAgent, CodeResult
from smolantic_ai.config import settings_manager
from smolantic_ai.logging import get_logger

# --- Basic Setup ---
load_dotenv()
settings_manager.reload()
logger = get_logger(__name__)

# --- Pydantic Models ---
class IdentifiedNeed(BaseModel):
    need_category: Literal["Entertainment", "Productivity", "Learning", "Travel", "Code"] = Field(description="The general category of the user's need.")
    details: Optional[str] = Field(None, description="Optional further details about the need.")

class Recommendation(BaseModel):
    product_name: str = Field(description="The specific product or service recommended.")
    reason: str = Field(description="A brief reason for the recommendation.")

class Failed(BaseModel):
    reason: str = Field(description="Reason why the agent failed.")

# --- Agent 1: Need Identification (MultistepAgent) ---
need_identifier_agent = MultistepAgent[None, IdentifiedNeed](
    model=f"{settings_manager.settings.model_provider}:{settings_manager.settings.model_name}", # Faster model for classification
    result_type=IdentifiedNeed,
    system_prompt=(
        'Identify the user\'s general need category (Entertainment, Productivity, Learning, Travel, Code) '
        'and any key details based on their request. Return the IdentifiedNeed object.'
    ),
    request_limit=5,
    logger_name="NeedIdentifierAgent",
    planning_interval=None, # Disable planning for this simple task
)

# --- Agent 2: Recommendation/Code Generation (CodeAgent) ---
# Use CodeAgent for tasks that might involve generating/running code for recommendations
recommendation_code_agent = CodeAgent(
    model=f"{settings_manager.settings.model_provider}:{settings_manager.settings.model_name}", # More capable model for code/complex reasoning
    # No specific tools needed here, relies on code execution
    executor_type="local", # Or "docker", "e2b" based on your setup
    request_limit=8,
    logger_name="RecommendationCodeAgent",
)

# --- Orchestration Logic ---
async def get_need_category() -> Union[IdentifiedNeed, None]:
    logger.info("\n--- Identifying Need Category ---")
    for attempt in range(3):
        try:
            prompt = Prompt.ask(
                f"[Attempt {attempt+1}/3] What are you looking for help with today? (e.g., 'movie suggestion', 'python script for web scraping', 'task manager app')"
            )
            # MultistepAgent run now returns AgentRunResult
            run_result = await need_identifier_agent.run(prompt)
            
            # Extract data from the result
            result_data = run_result.data

            if isinstance(result_data, IdentifiedNeed):
                logger.info(f"Identified Need: {result_data.need_category}, Details: {result_data.details}")
                return result_data
            else:
                # This case might indicate an internal agent error if result_type wasn't met
                logger.warning(f"Need identifier returned unexpected type: {type(result_data)}. Assuming failure.")
                # You might want to inspect 'result' further if MultistepAgent could return Failure models

        except Exception as e:
            logger.error(f"Error in get_need_category (attempt {attempt+1}): {e}", exc_info=True)
            print(f"An error occurred while identifying the need: {e}")

    logger.warning("Failed to identify need category after multiple attempts.")
    print("Sorry, I couldn't determine your need category after a few tries.")
    return None


async def get_recommendation_or_code(need: IdentifiedNeed) -> Union[CodeResult, None]:
    logger.info(f"\n--- Getting Recommendation/Code for {need.need_category} ---")
    preferences_prompt = Prompt.ask(
        f"Any specific preferences for {need.need_category}? (e.g., budget, genre, features, *specific task for code*)"
    )

    # Construct the prompt for the CodeAgent
    full_prompt = f"Need category: {need.need_category}.\n"
    if need.details:
        full_prompt += f"Initial details: {need.details}.\n"
    full_prompt += f"User preferences/details: {preferences_prompt}"

    # Add an explicit instruction if the category is 'Code' or details mention code/script
    if need.need_category == "Code" or (need.details and ('code' in need.details.lower() or 'script' in need.details.lower())) or ('code' in preferences_prompt.lower() or 'script' in preferences_prompt.lower()):
        full_prompt += "\nTask involves code. Please write and execute Python code to fulfill the request and print the final result."
    else:
        # For other categories, let the model decide if code is needed (e.g., simple calculation for 'Learning')
        full_prompt += "\nAddress the user's request based on the details provided."

    try:
        # CodeAgent's run returns a CodeResult object
        code_result: CodeResult = await recommendation_code_agent.run(full_prompt)

        logger.info(f"Code Agent Result - Code:\n{code_result.code}")
        logger.info(f"Code Agent Result - Execution Result:\n{code_result.result}")
        logger.info(f"Code Agent Result - Explanation:\n{code_result.explanation}")

        # Return the full CodeResult directly (removing the previous heuristic parsing)
        return code_result

    except Exception as e:
        logger.error(f"Error in get_recommendation_or_code: {e}", exc_info=True)
        print(f"An error occurred while getting the recommendation/code: {e}")
        return None

# --- Main Execution ---
async def main():
    logger.info("Running Multistep + CodeAgent Programmatic Hand-off Example...")

    # Step 1: Identify the user's need
    identified_need = await get_need_category()

    # Step 2: If need identified, get a recommendation or code result
    if identified_need:
        result = await get_recommendation_or_code(identified_need)

        print("\n--- Final Result ---")
        # Simplified output focusing on CodeResult
        if isinstance(result, CodeResult):
            print("Code Agent Result:")
            if result.code:
                print(f"\nGenerated Code:\n```python\n{result.code}\n```")
            else:
                print("\nNo code was generated/executed.")

            if result.result:
                 # Limit potentially long execution output for display
                print(f"\nExecution Result:\n{result.result[:1000]}{'...' if len(result.result) > 1000 else ''}")
            else:
                print("\nNo execution output was captured.")

            if result.explanation:
                print(f"\nAgent's Explanation:\n{result.explanation}")
        elif result is None:
            print("Sorry, could not provide a recommendation or code.")
        else:
            # Handle unexpected case (e.g., if future changes reintroduced Recommendation)
            print(f"Received unexpected result type: {type(result)}")
            if hasattr(result, 'product_name'): # Basic check if it looks like a Recommendation
                 print(f"  Product/Service: {getattr(result, 'product_name', 'N/A')}")
                 print(f"  Reason: {getattr(result, 'reason', 'N/A')}")

    else:
        print("\n--- Final Result ---")
        print("Sorry, could not determine your need.")

    # Usage info might be available on agent instances if logged/stored
    # logger.info(f"Need ID Agent Usage: {need_identifier_agent.some_usage_property}")
    # logger.info(f"Rec Code Agent Usage: {recommendation_code_agent.some_usage_property}")


if __name__ == "__main__":
    try:
        # Check for API keys
        api_key_found = any(os.getenv(key) for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"])
        if not api_key_found:
            print("Warning: No API key found for OpenAI, Anthropic, or Google in environment variables.")
            print("Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY in your .env file or environment.")

        # Check for rich (optional but used for prompts)
        try:
            import rich
        except ImportError:
            print("Warning: 'rich' library not found. Prompts will use standard input.")
            print("Install with: pip install rich")

        asyncio.run(main())
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}", exc_info=True)
        print(f"A critical error occurred: {e}")
        print("Check logs and ensure configuration/API keys are correct.") 