"""Utility functions for the smolantic_ai package."""

import re
from typing import List, Tuple, Optional


def parse_code_blobs(text: str) -> str:
    """Extract Python code from a Markdown codeblock.
    
    This function extracts code from various formats that might be produced
    by the LLM, including:
    - ```python\ncode\n```
    - ```py\ncode\n```
    - Code in 'Thought:', 'Code:' sequences
    
    Args:
        text: The text containing the code blobs
        
    Returns:
        The extracted code as a string
    """
    # First try to extract code blocks with explicit Python markers
    code_block_pattern = r"```(?:python|py)\n(.*?)```"
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Try generic code blocks if no Python-specific ones found
    generic_block_pattern = r"```\n?(.*?)```"
    matches = re.findall(generic_block_pattern, text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Look for 'Code:' sequence if no code blocks found
    code_sequence_pattern = r"Code:\s*\n(.*?)(?:\n\s*(?:Observation|<end_code>)|$)"
    matches = re.findall(code_sequence_pattern, text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # If still nothing found, return the whole text stripped
    # This is a fallback for simple code responses
    return text.strip()


def fix_final_answer_code(code: str) -> str:
    """Ensure code properly uses the final_answer function.
    
    This function checks if the code contains a final_answer call,
    and if not, handles potential return statements to wrap them
    with final_answer.
    
    Args:
        code: The code to fix
        
    Returns:
        The fixed code string
    """
    if "final_answer(" in code:
        return code
    
    # Look for return statements at the end of the code
    lines = code.splitlines()
    new_lines = []
    
    # Process each line, handling return statements
    for i, line in enumerate(lines):
        if line.strip().startswith("return ") and (i == len(lines) - 1 or all(l.strip() == "" for l in lines[i+1:])):
            # Replace return with final_answer
            value = line.strip()[7:].strip()
            new_lines.append(f"final_answer({value})")
        else:
            new_lines.append(line)
    
    # If the last non-empty line isn't a final_answer call or return,
    # check if it's a variable or expression that should be returned
    if not any("final_answer(" in line for line in new_lines) and new_lines:
        last_line = new_lines[-1].strip()
        if last_line and not last_line.endswith(":") and not last_line.endswith(";") and "=" not in last_line:
            # It's likely an expression to return
            new_lines[-1] = f"final_answer({last_line})"
    
    return "\n".join(new_lines)


def extract_thought_action_observation(text: str) -> Tuple[str, str, str]:
    """Extract thought, action (code), and observation from a step.
    
    Args:
        text: The text containing the thought-code-observation sequence
        
    Returns:
        A tuple of (thought, code, observation)
    """
    thought_pattern = r"Thought:(.*?)(?:Code:|$)"
    code_pattern = r"Code:(.*?)(?:Observation:|<end_code>|$)"
    observation_pattern = r"Observation:(.*?)(?:Thought:|$)"
    
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    code_match = re.search(code_pattern, text, re.DOTALL)
    observation_match = re.search(observation_pattern, text, re.DOTALL)
    
    thought = thought_match.group(1).strip() if thought_match else ""
    code = code_match.group(1).strip() if code_match else ""
    observation = observation_match.group(1).strip() if observation_match else ""
    
    # If code has markdown formatting, extract it
    if code and ("```" in code):
        code = parse_code_blobs(code)
    
    return thought, code, observation 