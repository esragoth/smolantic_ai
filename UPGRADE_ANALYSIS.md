# Pydantic-AI Upgrade Analysis: 0.1.6 → 1.14.1

## Current Status
- **Current Version**: 0.1.6
- **Target Version**: 1.14.1
- **Version Jump**: Major version change (0.x → 1.x) - **BREAKING CHANGES EXPECTED**

## Initial Findings (Verified with Current Version 0.1.6)

### Agent Constructor Compatibility ✅
The current codebase already uses `output_type` parameter which matches the new API:
- ✅ `BaseAgent.__init__` uses `output_type: Type[ResultT]` 
- ✅ `MultistepAgent.__init__` uses `output_type: Type[ResultT]`
- ✅ `ContinuousAgent.__init__` uses `output_type: Type[ResultT]`
- ⚠️ `CodeAgent.__init__` uses `result_type` parameter but converts it to `output_type` when calling super()

**Current Agent API (0.1.6) signature:**
```python
Agent(model, *, output_type, deps_type, tools, system_prompt, ...)
```

**New Agent API (1.14.1) signature:**
```python
Agent(model, *, output_type, deps_type, tools, system_prompt, instructions, ...)
```

**Good News:** The `output_type` parameter name is consistent! However, there may be new parameters like `instructions` that could be useful.

## Critical Areas to Review

### 1. Agent Initialization
**Current Usage:**
- `Agent[DepsT, ResultT]` constructor with parameters: `model`, `output_type`, `deps_type`, `tools`, `system_prompt`
- `BaseAgent` extends `Agent` and passes these parameters

**Potential Impact:**
- Constructor signature may have changed
- Parameter names or types may differ
- Model string format (`"provider:model"`) may have changed

**Files Affected:**
- `src/smolantic_ai/agent.py` (lines 34-82)
- `src/smolantic_ai/code_agent.py` (lines 41-78)
- `src/smolantic_ai/multistep_agent.py` (lines 83-131)
- `src/smolantic_ai/continuous_agent.py` (lines 16-54)

### 2. AgentRunResult vs AgentRun
**Current Usage:**
- `AgentRunResult` imported from `pydantic_ai.agent`
- Used in `_process_run_result` methods
- Accessing `agent_run.result`, `agent_run.result.data`
- `AgentRun` used in continuous agent for graph control

**Potential Impact:**
- Class name or structure may have changed
- Result access patterns (`result`, `data`, `output`) may differ
- Usage statistics access (`agent_run.usage()`) may have changed

**Files Affected:**
- `src/smolantic_ai/agent.py` (lines 5, 134, 231, 608, 614, 660)
- `src/smolantic_ai/code_agent.py` (lines 5, 137-183)
- `src/smolantic_ai/multistep_agent.py` (lines 5, 164-221)
- `src/smolantic_ai/continuous_agent.py` (lines 4, 59, 76, 214, 238, 297, 301)

### 3. Message Types and Structures
**Current Usage:**
- `messages.ModelRequest`, `messages.ModelResponse`
- `messages.UserPromptPart`, `messages.TextPart`, `messages.ToolCallPart`, `messages.ToolReturnPart`
- `messages.ModelRequest` with `parts` attribute
- `messages.ModelResponse` with `parts` attribute

**Potential Impact:**
- Message structure may have changed
- Part types may have been renamed or restructured
- Access patterns may differ

**Files Affected:**
- `src/smolantic_ai/agent.py` (lines 4, 189, 192, 284-286, 316-317, 328, 404-406, 477)
- `src/smolantic_ai/code_agent.py` (line 4)
- `src/smolantic_ai/models.py` (lines 4-5)
- `src/smolantic_ai/multistep_agent.py` (line 28)

### 4. Graph Control and Iteration
**Current Usage:**
- `self.iter(task)` returns async context manager
- `async with self.iter(task) as agent_run:`
- Iterating over nodes: `async for node in agent_run:`
- Node types: `UserPromptNode`, `ModelRequestNode`, `CallToolsNode`, `End`
- Node checking: `Agent.is_user_prompt_node()`, `Agent.is_call_tools_node()`, `Agent.is_model_request_node()`, `Agent.is_end_node()`

**Potential Impact:**
- Graph iteration API may have changed
- Node types may have been renamed or restructured
- Node checking methods may have changed signatures

**Files Affected:**
- `src/smolantic_ai/agent.py` (lines 175-228)
- `src/smolantic_ai/continuous_agent.py` (lines 76, 181, 190, 256, 269)

### 5. Model Settings and Request Parameters
**Current Usage:**
- `ModelSettings` from `pydantic_ai.models`
- `ModelRequestParameters` from `pydantic_ai.models`
- `self.model.request()` method
- `ModelResponse` from `pydantic_ai.models`

**Potential Impact:**
- Model request API may have changed
- Settings structure may differ
- Request parameters may have been restructured

**Files Affected:**
- `src/smolantic_ai/agent.py` (lines 6, 478-487)
- `src/smolantic_ai/code_agent.py` (line 6)

### 6. Tool Definitions
**Current Usage:**
- `Tool` from `pydantic_ai`
- `Tool(name=..., description=..., function=...)`
- `RunContext` parameter in tool functions

**Potential Impact:**
- Tool definition API may have changed
- `RunContext` may have been renamed or restructured
- Tool execution may work differently

**Files Affected:**
- All agent files use `Tool`
- `src/smolantic_ai/prebuilt_tools.py`
- `src/smolantic_ai/executors.py` (line 10)

### 7. Usage Statistics
**Current Usage:**
- `agent_run.usage()` returns usage object
- Accessing `usage.request_tokens`, `usage.response_tokens`, `usage.total_tokens`
- `from pydantic_ai.usage import Usage` (in examples)

**Potential Impact:**
- Usage API may have changed
- Property names may differ
- Access pattern may be different

**Files Affected:**
- `src/smolantic_ai/agent.py` (lines 608, 660)
- `src/smolantic_ai/continuous_agent.py` (lines 214, 238, 254, 372)
- `examples/multistep_agent_delegation_story.py` (line 20)

### 8. Model String Format
**Current Usage:**
- Format: `"provider:model"` (e.g., `"openai:gpt-4o"`, `"google:gemini-1.5-flash-latest"`)
- Used in `config.py` as `f"{provider}:{model_name}"`

**Potential Impact:**
- Format may have changed
- Provider names may differ
- Model name format may have changed

**Files Affected:**
- `src/smolantic_ai/config.py` (line 32)
- All agent initialization code

## Recommended Upgrade Strategy

### Phase 1: Preparation
1. **Create a backup branch** of your current working code
2. **Review release notes** for versions between 0.1.6 and 1.14.1
3. **Set up a test environment** with the new version

### Phase 2: Incremental Testing
1. **Update requirements.txt** to `pydantic-ai>=1.14.1`
2. **Install new version** in a virtual environment
3. **Run existing tests** to identify immediate breakages
4. **Fix import errors** first (easiest to identify)

### Phase 3: API Compatibility
1. **Check Agent initialization** - verify constructor parameters
2. **Check AgentRunResult/AgentRun** - verify result access patterns
3. **Check message structures** - verify part types and access
4. **Check graph iteration** - verify node types and iteration
5. **Check tool definitions** - verify Tool API
6. **Check usage statistics** - verify usage API

### Phase 4: Testing
1. **Run all unit tests**
2. **Run all example scripts**
3. **Test each agent type** (BaseAgent, CodeAgent, MultistepAgent, ContinuousAgent)
4. **Test edge cases** and error handling

## Key Files to Test First

1. **`src/smolantic_ai/agent.py`** - Core base agent class
2. **`src/smolantic_ai/code_agent.py`** - Code execution agent
3. **`src/smolantic_ai/multistep_agent.py`** - Multi-step agent
4. **`src/smolantic_ai/continuous_agent.py`** - Continuous conversation agent
5. **`tests/test_agents.py`** - Test suite

## Potential Breaking Changes Summary

Based on the major version jump, expect:
- ✅ Constructor parameter changes
- ✅ Class/import name changes
- ✅ API method signature changes
- ✅ Result structure changes
- ✅ Message/part type changes
- ✅ Graph iteration API changes
- ✅ Usage statistics API changes

## Next Steps

1. **Review GitHub releases page** for detailed changelog: https://github.com/pydantic/pydantic-ai/releases
2. **Check pydantic-ai documentation** for migration guides
3. **Test incrementally** - don't update everything at once
4. **Keep notes** on what breaks and how to fix it

## Testing Command

After updating, run:
```bash
source venv/bin/activate
pip install --upgrade pydantic-ai
python -m pytest tests/test_agents.py -v
```

Then test each example:
```bash
python examples/continuous_agent_example.py
python examples/code_agent_example.py
python examples/multistep_agent_generic.py
```

## Specific Code Patterns to Check

### Pattern 1: AgentRunResult.result.data
**Location:** `src/smolantic_ai/agent.py:139`, `src/smolantic_ai/code_agent.py:140-143`
```python
if hasattr(agent_run, 'result') and agent_run.result is not None:
    if hasattr(agent_run.result, 'data'):
        actual_result_data = agent_run.result.data
```
**Check:** Verify if `result.data` still exists or if it's now `result.output` or just `result`

### Pattern 2: agent_run.usage()
**Location:** Multiple files
```python
usage_stats = agent_run.usage()
usage_stats.request_tokens
usage_stats.response_tokens
usage_stats.total_tokens
```
**Check:** Verify the usage API hasn't changed

### Pattern 3: Graph Iteration
**Location:** `src/smolantic_ai/agent.py:175-228`
```python
async with self.iter(task, **kwargs) as agent_run:
    async for node in agent_run:
        if Agent.is_user_prompt_node(node):
        elif Agent.is_call_tools_node(node):
        elif Agent.is_model_request_node(node):
        elif Agent.is_end_node(node):
```
**Check:** Verify node types and checking methods still work

### Pattern 4: Model Request
**Location:** `src/smolantic_ai/agent.py:477-487`
```python
message = messages.ModelRequest(parts=[messages.UserPromptPart(content=rendered_user_prompt)])
planning_settings = ModelSettings(stop_sequences=["<end_plan>"])
request_params = ModelRequestParameters(function_tools=[], allow_text_output=True, output_tools=[])
response = await self.model.request(messages=[message], model_settings=planning_settings, model_request_parameters=request_params)
```
**Check:** Verify `model.request()` API and parameter names

### Pattern 5: Message Parts Access
**Location:** Multiple files
```python
if hasattr(assistant_response, 'parts'):
    for part in assistant_response.parts:
        if isinstance(part, messages.TextPart):
        elif isinstance(part, messages.ToolCallPart):
        elif isinstance(part, messages.ToolReturnPart):
```
**Check:** Verify part types and access patterns

## Resources

- **GitHub Releases:** https://github.com/pydantic/pydantic-ai/releases
- **Documentation:** https://ai.pydantic.dev (check for migration guides)
- **Current Version Info:** Run `pip show pydantic-ai` to see installed version details

