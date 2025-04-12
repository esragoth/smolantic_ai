import asyncio
import os
from pydantic import BaseModel
from pydantic_ai import Tool
from smolantic_ai.multistep_agent import MultistepAgent
from smolantic_ai.prebuilt_tools import (
    add, subtract, multiply, divide, convert_currency
)
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define a simple result model using a Tool for the final answer
class FinalAnswer(BaseModel):
    answer: str

def final_answer(answer: str) -> FinalAnswer:
    """Return the final answer to the user."""
    return FinalAnswer(answer=answer)

async def main():
    # Define the tools the agent can use
    tools = [
        Tool(name="add", function=add, description="Add two numbers"),
        Tool(name="subtract", function=subtract, description="Subtract two numbers"),
        Tool(name="multiply", function=multiply, description="Multiply two numbers"),
        Tool(name="divide", function=divide, description="Divide two numbers. Handles division by zero."),
        Tool(name="convert_currency", function=convert_currency, description="Convert an amount from one currency to another (e.g., USD to EUR)"),
        Tool(name="final_answer", function=final_answer, description="Provide the final answer to the user. Use this when the calculation is successfully finished OR when an unrecoverable error occurs during a step.")
    ]

    # Ensure the necessary API key for the LLM is set in the environment
    # Example check for OpenAI, adjust if using a different provider
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it in your .env file or environment.")
        return

    # Instantiate the MultistepAgent
    # It uses the default model specified in smolantic_ai.config if not overridden
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
    task = "I'm planning a trip. My budget is 1500 EUR. My flight costs 600 USD. My accommodation for 5 nights costs 90 GBP per night. First, calculate the total accommodation cost in GBP. Then, convert the flight cost (USD) and the total accommodation cost (GBP) to EUR. Add these two EUR costs together. Finally, subtract the total EUR cost from my budget to see how much I have left for other expenses. What is the remaining amount in EUR?"

    print(f"--- Running MultiStep Agent ---")
    print(f"Task: {task}")
    print("----------------------------------")
    print("Note: Ensure API keys for currency conversion are set in prebuilt_tools.py")
    print("----------------------------------\n")

    # Run the agent
    try:
        # The agent will use planning and the provided tools to solve the task
        # Now we expect the agent to return the correct FinalAnswer type
        result: FinalAnswer = await agent.run(task)
        print(f"\n--- Agent Finished ---")

        # Check if the result is the expected FinalAnswer type
        if isinstance(result, FinalAnswer):
            print(f"Final Answer: {result.answer}")
        # Removed the elif for dict, as run should now return the correct type
        else:
            # If not, print the raw result, which might be an error dict/string
            print(f"Agent returned an unexpected result format: {result}")

        print("----------------------")

    except Exception as e:
        print(f"\n--- Agent Error ---")
        print(f"An error occurred during agent execution: {e}")
        print("-------------------")

if __name__ == "__main__":
    asyncio.run(main()) 