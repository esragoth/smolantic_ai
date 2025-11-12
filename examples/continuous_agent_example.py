from dotenv import load_dotenv
import os
import asyncio
import time
from typing import List, Optional
from pydantic import BaseModel
from smolantic_ai.continuous_agent import ContinuousAgent
from smolantic_ai.prebuilt_tools import (
    get_weather_tool,
    search_google_tool,
    timezone_tool,
    read_webpage_tool
)
from smolantic_ai.config import settings_manager

# Force reload of .env file
load_dotenv(override=True)

class AgentResponse(BaseModel):
    """Response model for the continuous agent."""
    message: str
    weather: Optional[str] = None
    time: Optional[str] = None
    search_results: Optional[List[dict]] = None

async def main():
    # Force reload settings
    settings_manager.reload()
    
    # Create continuous agent with graph control
    agent = ContinuousAgent(
        output_type=AgentResponse,
        tools=[
            get_weather_tool,
            search_google_tool,
            timezone_tool,
        ],
        model="openai:gpt-4o-mini",
        verbose=True,  # Disable verbose logging for clean chat
        use_persistent_session=True,  # Enable graph control
        name="GraphControlledContinuousAgent"
    )
    
    print("🚀 Interactive Continuous Agent")
    print("=" * 50)
    
    # Start the conversation
    await agent.start_conversation()
    
    session_status = "✅ Graph Control" if agent.is_using_persistent_session else "⚠️  Fallback Mode"
    print(f"{session_status} | Ready to chat!")
    print("\n💡 Commands: 'history', 'clear', 'stats', 'exit'")
    print("=" * 50)
    
    try:
        while agent.is_active:
            # Get user input
            try:
                user_input = input("\n👤 ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Goodbye!")
                break
            
            # Handle empty input
            if not user_input:
                continue
                
            # Handle special commands
            if user_input.lower() in ['exit', 'quit']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'history':
                await show_conversation_history(agent)
                continue
            elif user_input.lower() == 'clear':
                await agent.clear_conversation_history()
                print("🧹 History cleared!")
                continue
            elif user_input.lower() == 'stats':
                await show_conversation_stats(agent)
                continue
            
            # Process user input with timing
            try:
                # Show thinking indicator
                print("🤖 ", end="", flush=True)
                
                # Record start time
                start_time = time.time()
                
                response = await agent.process_message(user_input)
                
                # Calculate processing time
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Get steps taken from the response if available
                steps_taken = getattr(response, '_steps_taken', 0)
                
                # Get token usage - prioritize directly stored usage
                token_usage = None
                
                # First check for directly stored token usage
                stored_token_usage = getattr(response, '_token_usage', None)
                if stored_token_usage:
                    token_usage = stored_token_usage
                
                # Fallback: try stored agent_run
                if not token_usage:
                    stored_agent_run = getattr(response, '_agent_run', None)
                    if stored_agent_run:
                        try:
                            usage = stored_agent_run.usage()
                            if usage:
                                token_usage = {
                                    'total': usage.total_tokens,
                                    'input': usage.input_tokens,
                                    'output': usage.output_tokens
                                }
                        except Exception:
                            pass
                
                # Check for agent run result from fallback method
                if not token_usage:
                    agent_run_result = getattr(response, '_agent_run_result', None)
                    if agent_run_result and hasattr(agent_run_result, 'usage'):
                        try:
                            usage = agent_run_result.usage()
                            if usage:
                                token_usage = {
                                    'total': usage.total_tokens,
                                    'input': usage.input_tokens,
                                    'output': usage.output_tokens
                                }
                        except Exception:
                            pass
                
                # Final fallback: try current agent_run
                if not token_usage and hasattr(agent, 'agent_run') and agent.agent_run:
                    try:
                        usage = agent.agent_run.usage()
                        if usage:
                            token_usage = {
                                'total': usage.total_tokens,
                                'input': usage.request_tokens,
                                'output': usage.response_tokens
                            }
                    except Exception:
                        pass
                
                # Extract the actual message from the response
                if hasattr(response, 'message'):
                    message = response.message
                elif hasattr(response, 'output') and hasattr(response.output, 'message'):
                    message = response.output.message
                    response = response.output  # Use the output for other fields
                else:
                    message = str(response)
                
                # Clear the thinking indicator and show response
                print(f"\r🤖 {message}")
                
                # Show additional information in a clean format
                extras = []
                if hasattr(response, 'weather') and response.weather:
                    extras.append(f"🌤️  {response.weather}")
                if hasattr(response, 'time') and response.time:
                    extras.append(f"🕐 {response.time}")
                if hasattr(response, 'search_results') and response.search_results:
                    extras.append(f"🔍 {len(response.search_results)} search results:")
                    for i, result in enumerate(response.search_results[:3], 1):
                        title = result.get('title', 'No title')[:60]
                        if len(result.get('title', '')) > 60:
                            title += "..."
                        extras.append(f"   {i}. {title}")
                        if result.get('url'):
                            extras.append(f"      🔗 {result['url']}")
                
                # Print extras with proper spacing
                if extras:
                    print()
                    for extra in extras:
                        print(extra)
                
                # Show metrics
                print()
                metrics = []
                metrics.append(f"⏱️  {processing_time:.2f}s")
                if steps_taken > 0:
                    metrics.append(f"👣 {steps_taken} steps")
                if token_usage and token_usage['total'] > 0:
                    metrics.append(f"🪙 {token_usage['total']} tokens ({token_usage['input']}→{token_usage['output']})")
                
                if metrics:
                    print(f"📊 {' | '.join(metrics)}")
                
            except Exception as e:
                print(f"\r❌ Error: {str(e)}")
    
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    
    finally:
        # Stop the conversation
        await agent.stop_conversation()
        print("\n" + "=" * 50)
        print("🛑 Chat ended")
        
        # Show final stats
        await show_final_stats(agent)

async def show_conversation_history(agent: ContinuousAgent):
    """Display the conversation history in a chat-like format."""
    history = await agent.get_conversation_history()
    
    if not history:
        print("📭 No messages yet.")
        return
    
    print(f"\n💬 Chat History ({len(history)} messages)")
    print("-" * 40)
    
    for msg in history:
        if msg["role"] == "user":
            content = msg["content"]
            if len(content) > 80:
                content = content[:77] + "..."
            print(f"👤 {content}")
        elif msg["role"] == "assistant":
            content = msg["content"]
            if len(content) > 80:
                content = content[:77] + "..."
            print(f"🤖 {content.output.message}")
        elif msg["role"] == "error":
            print(f"❌ {msg['content'][:80]}...")
    
    print("-" * 40)

async def show_conversation_stats(agent: ContinuousAgent):
    """Show conversation statistics in a compact format."""
    history = await agent.get_conversation_history()
    uptime = agent.conversation_uptime or 0
    
    user_messages = len([msg for msg in history if msg["role"] == "user"])
    agent_messages = len([msg for msg in history if msg["role"] == "assistant"])
    
    # Get total steps
    total_steps = agent.step_count
    
    # Get total token usage from the most recent agent run
    total_tokens = 0
    if hasattr(agent, 'agent_run') and agent.agent_run:
        try:
            usage = agent.agent_run.usage()
            if usage:
                total_tokens = usage.total_tokens
        except Exception:
            pass
    
    stats_parts = [f"{uptime:.0f}s uptime", f"{user_messages} user", f"{agent_messages} agent"]
    if total_steps > 0:
        stats_parts.append(f"{total_steps} steps")
    if total_tokens > 0:
        stats_parts.append(f"{total_tokens} tokens")
    stats_parts.append('Graph' if agent.is_using_persistent_session else 'Fallback')
    
    print(f"\n📊 Stats: {' | '.join(stats_parts)}")

async def show_final_stats(agent: ContinuousAgent):
    """Show final conversation statistics."""
    history = await agent.get_conversation_history()
    uptime = agent.conversation_uptime or 0
    
    if history:
        user_count = len([m for m in history if m['role'] == 'user'])
        avg_time = uptime / user_count if user_count > 0 else 0
        
        stats_parts = [f"{uptime:.0f}s total", f"{len(history)} messages", f"{avg_time:.1f}s avg"]
        
        # Add total steps
        if agent.step_count > 0:
            stats_parts.append(f"{agent.step_count} total steps")
        
        print(f"📊 {' | '.join(stats_parts)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}") 