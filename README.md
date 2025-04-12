# Smolantic AI

Smolantic AI is a Python framework leveraging `pydantic-ai` to build specialized AI agents for multi-step task processing, tool calling, and code generation/execution. It provides structured agent types with memory management and planning capabilities.

## Features

*   **Modular Agent Design:** Separate agents for different tasks (`MultistepAgent`, `ToolCallingAgent`, `CodeAgent`).
*   **Structured Planning & Execution:** Agents follow defined steps for planning, execution, and error handling.
*   **Tool Integration:** Easily integrate and use custom or pre-built tools with `ToolCallingAgent`.
*   **Code Generation & Execution:** Generate and safely execute Python code using `CodeAgent` with configurable executors (local, Docker, E2B).
*   **Configuration:** Flexible configuration via environment variables or `.env` files using `pydantic-settings`.
*   **Extensible Models:** Uses Pydantic models for clear data structures (Messages, Actions, Memory).

## Installation

Currently, the package is best used by cloning the repository and installing in editable mode for development.

1.  Clone the repository:
    ```bash
    git clone https://github.com/esragoth/smolantic_ai.git
    cd smolantic_ai
    ```
2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Install the package in editable mode (optional, for development):
    ```bash
    pip install -e .
    ```
5.  **Environment Variables:** Set up your API keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`) in a `.env` file in the root directory or as environment variables.

## Usage

Here are basic examples of how to use the agents:

**ToolCallingAgent:**

```python
import asyncio
from pydantic_ai import Tool
from smolantic_ai import ToolCallingAgent
from smolantic_ai.models import Message, MessageRole

# Define a simple tool
def get_weather(city: str) -> str:
    """Gets the weather for a city."""
    # Replace with actual API call
    return f"The weather in {city} is sunny."

weather_tool = Tool(
    name="get_weather",
    description="Get the current weather for a specific city",
    function=get_weather,
)

async def run_tool_agent():
    agent = ToolCallingAgent(tools=[weather_tool])
    messages = [Message(role=MessageRole.USER, content="What's the weather like in London?")]
    result = await agent.run(messages)
    print(f"Tool Agent Input: {messages[-1].content}")
    print(f"Tool Agent Thought: {result.output_messages[0].content if result.output_messages else 'N/A'}")
    print(f"Tool Agent Action: {result.memory.action_steps[-1].to_string_summary() if result.memory.action_steps else 'N/A'}")
    print(f"Tool Agent Final Answer: {result.final_answer}")

asyncio.run(run_tool_agent())
```

**CodeAgent:**

```python
import asyncio
from smolantic_ai import CodeAgent
from smolantic_ai.models import Message, MessageRole

async def run_code_agent():
    # Uses local executor by default
    agent = CodeAgent(authorized_imports=["math"]) # Allow math module
    messages = [Message(role=MessageRole.USER, content="Write a Python function to calculate the area of a circle given its radius.")]
    result = await agent.run(messages) # Result contains code, execution output, etc.
    print(f"Code Agent Input: {messages[-1].content}")
    print(f"Code Agent Thought: {result.output_messages[0].content if result.output_messages else 'N/A'}")
    print(f"Code Agent Action: {result.memory.action_steps[-1].to_string_summary() if result.memory.action_steps else 'N/A'}") # May not always have explicit action step if simple
    print(f"Code Agent Generated Code:\n{result.code}")
    print(f"Code Agent Final Answer: {result.final_answer}")


asyncio.run(run_code_agent())
```

**MultistepAgent:**

(See `examples/multistep_agent_generic.py` and `examples/multistep_agent_numbers.py` for more detailed usage)

```python
import asyncio
from smolantic_ai import MultistepAgent
from smolantic_ai.models import Message, MessageRole

async def run_multistep_agent():
    agent = MultistepAgent() # Uses default tools if none provided
    messages = [Message(role=MessageRole.USER, content="Research the capital of France and then find its population.")]
    result = await agent.run(messages) # Result contains planning, task steps, final answer, etc.
    print(f"Multistep Input: {messages[-1].content}")
    print("Multistep Plan:")
    if result.memory and result.memory.action_steps:
         # Find the first planning step
         planning_step = next((step for step in result.memory.action_steps if step.step_type == 'planning'), None)
         if planning_step:
             print(planning_step.to_string_summary())
         else:
             print("No planning step found in memory.")
    else:
        print("No memory or action steps recorded.")

    print(f"Multistep Final Answer: {result.final_answer}")

asyncio.run(run_multistep_agent())

```

## Development

To set up the development environment:

1.  Clone the repository (`git clone ...`)
2.  Create and activate a virtual environment (`python -m venv venv`, `source venv/bin/activate`)
3.  Install development dependencies:
    ```bash
    pip install -r requirements.txt
    pip install -e ".[dev]" # Installs test dependencies
    ```

## Testing

Run tests using:
```bash
pytest
```

## License

[Your chosen license] - *Please specify which license you want to use (e.g., MIT, Apache 2.0)* 