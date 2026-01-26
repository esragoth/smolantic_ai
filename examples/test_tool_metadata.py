"""
Example script to test tool metadata capture in node and step callbacks.

This script demonstrates how tools that return ToolReturn objects with metadata
have that metadata captured and passed to both node_callback and step_callback.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import asyncio
import time
from typing import Dict, Any
from pydantic import BaseModel
from pydantic_ai import Tool
from pydantic_ai.messages import ToolReturn
from smolantic_ai.multistep_agent import MultistepAgent
from smolantic_ai.config import settings_manager
from smolantic_ai.models import Node, Step
from smolantic_ai.agent import AgentInfo


class SimpleResult(BaseModel):
    """Simple result model for testing."""
    answer: str
    explanation: str = ""


# Track metadata captured in callbacks
captured_metadata = {
    'node_callback': [],
    'step_callback': []
}


def node_callback(node: Node, agent_info: AgentInfo):
    """Callback for node events - captures tool metadata."""
    # Check if this is a ModelRequestNode with tool returns
    if hasattr(node, 'tool_return') and node.tool_return:
        for tool_return in node.tool_return:
            if 'metadata' in tool_return:
                metadata_info = {
                    'tool_name': tool_return.get('tool_name', 'unknown'),
                    'metadata': tool_return['metadata'],
                    'node_type': node.node_type.value
                }
                captured_metadata['node_callback'].append(metadata_info)
                print(f"\n[Node Callback] Captured metadata from tool '{tool_return.get('tool_name')}':")
                print(f"  Metadata: {tool_return['metadata']}")


def step_callback(step: Step, agent_info: AgentInfo):
    """Callback for step events - captures tool metadata."""
    # Check if this is an ActionStep with tool outputs
    if hasattr(step, 'tool_outputs') and step.tool_outputs:
        for tool_output in step.tool_outputs:
            if 'metadata' in tool_output:
                metadata_info = {
                    'tool_name': tool_output.get('name', 'unknown'),
                    'metadata': tool_output['metadata'],
                    'step_number': getattr(step, 'step_number', 0)
                }
                captured_metadata['step_callback'].append(metadata_info)
                print(f"\n[Step Callback] Captured metadata from tool '{tool_output.get('name')}':")
                print(f"  Metadata: {tool_output['metadata']}")


# Create a tool that returns ToolReturn with metadata
@Tool
def get_data_with_metadata(query: str) -> ToolReturn:
    """
    Get data and return it with metadata.
    
    This tool demonstrates ToolReturn usage with metadata that will be
    captured in both node and step callbacks.
    """
    # Simulate some data retrieval
    data = f"Data for query: {query}"
    
    # Create metadata with useful information
    metadata = {
        'query': query,
        'timestamp': time.time(),
        'source': 'example_tool',
        'execution_time_ms': 42.5,
        'cache_hit': False,
        'version': '1.0'
    }
    
    # Return ToolReturn with return_value, content, and metadata
    return ToolReturn(
        return_value=data,
        content=f"Successfully retrieved: {data}",
        metadata=metadata
    )


# Create another tool that returns ToolReturn with different metadata
@Tool
def process_data_with_metadata(data: str, operation: str) -> ToolReturn:
    """
    Process data and return it with metadata.
    
    Another example tool that returns metadata.
    """
    # Simulate processing
    processed = f"Processed {data} using {operation}"
    
    metadata = {
        'operation': operation,
        'input_length': len(data),
        'output_length': len(processed),
        'processing_time_ms': 15.3,
        'status': 'success'
    }
    
    return ToolReturn(
        return_value=processed,
        content=f"Processing complete: {processed}",
        metadata=metadata
    )


# Create a simple tool without metadata for comparison
@Tool
def simple_tool(message: str) -> str:
    """A simple tool that doesn't return metadata."""
    return f"Simple response: {message}"


async def main():
    """Main function to test tool metadata capture."""
    # Force reload settings
    settings_manager.reload()
    
    # Reset captured metadata
    captured_metadata['node_callback'] = []
    captured_metadata['step_callback'] = []
    
    print("=" * 80)
    print("Testing Tool Metadata Capture in Node and Step Callbacks")
    print("=" * 80)
    print()
    
    # Create agent with tools that return metadata
    agent = MultistepAgent(
        output_type=SimpleResult,
        tools=[get_data_with_metadata, process_data_with_metadata, simple_tool],
        planning_interval=None,  # Disable planning for simpler test
        model="openai:gpt-4.1",
        node_callback=node_callback,
        step_callback=step_callback,
        verbose=True
    )
    
    # Define a task that will use the tools
    task = (
        "First, get data for 'test query' using get_data_with_metadata. "
        "Then process that data using 'transform' operation with process_data_with_metadata. "
        "Finally, use simple_tool with message 'hello'. "
        "Provide a summary of what was done."
    )
    
    print(f"Task: {task}")
    print("-" * 80)
    print()
    
    try:
        # Run the agent
        result = await agent.run(task)
        
        print("\n" + "=" * 80)
        print("Final Result:")
        print("=" * 80)
        print(f"Answer: {result.answer}")
        if result.explanation:
            print(f"Explanation: {result.explanation}")
        
        print("\n" + "=" * 80)
        print("Metadata Capture Summary:")
        print("=" * 80)
        
        print(f"\nNode Callback captured {len(captured_metadata['node_callback'])} metadata entries:")
        for i, meta_info in enumerate(captured_metadata['node_callback'], 1):
            print(f"  {i}. Tool: {meta_info['tool_name']}")
            print(f"     Metadata: {meta_info['metadata']}")
        
        print(f"\nStep Callback captured {len(captured_metadata['step_callback'])} metadata entries:")
        for i, meta_info in enumerate(captured_metadata['step_callback'], 1):
            print(f"  {i}. Tool: {meta_info['tool_name']} (Step {meta_info['step_number']})")
            print(f"     Metadata: {meta_info['metadata']}")
        
        # Verification
        print("\n" + "=" * 80)
        print("Verification:")
        print("=" * 80)
        
        # Check that metadata was captured
        tools_with_metadata = ['get_data_with_metadata', 'process_data_with_metadata']
        node_metadata_count = len(captured_metadata['node_callback'])
        step_metadata_count = len(captured_metadata['step_callback'])
        
        print(f"Expected: Tools with metadata should be called: {tools_with_metadata}")
        print(f"Node callback metadata entries: {node_metadata_count}")
        print(f"Step callback metadata entries: {step_metadata_count}")
        
        if node_metadata_count > 0 and step_metadata_count > 0:
            print("\n✓ SUCCESS: Metadata was captured in both callbacks!")
            
            # Verify specific metadata fields
            all_metadata = captured_metadata['node_callback'] + captured_metadata['step_callback']
            for meta_info in all_metadata:
                metadata = meta_info['metadata']
                if isinstance(metadata, dict):
                    print(f"\n  Tool '{meta_info['tool_name']}' metadata fields:")
                    for key, value in metadata.items():
                        print(f"    - {key}: {value}")
        else:
            print("\n✗ WARNING: Metadata was not captured. Check tool execution.")
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
