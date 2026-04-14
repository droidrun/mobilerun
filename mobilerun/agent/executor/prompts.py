"""
Prompts for the ExecutorAgent.
"""


def parse_executor_response(response: str) -> dict:
    """
    Parse the Executor LLM response.

    Extracts:
    - thought: Content between "### Thought" and "### Action"
    - action: Content between "### Action" and "### Description"
    - description: Content after "### Description"

    Args:
        response: Raw LLM response string

    Returns:
        Dictionary with 'thought', 'action', 'description' keys
    """
    thought = (
        response.split("### Thought")[-1]
        .split("### Action")[0]
        .replace("\n", " ")
        .replace("  ", " ")
        .replace("###", "")
        .strip()
    )
    action_raw = (
        response.split("### Action")[-1]
        .split("### Description")[0]
        .replace("\n", " ")
        .replace("  ", " ")
        .replace("###", "")
        .strip()
    )
    start_idx = action_raw.find("{")
    end_idx = action_raw.rfind("}")
    if start_idx != -1 and end_idx != -1:
        action = action_raw[start_idx : end_idx + 1]
    else:
        action = action_raw
    description = (
        response.split("### Description")[-1]
        .replace("\n", " ")
        .replace("  ", " ")
        .replace("###", "")
        .strip()
    )

    return {"thought": thought, "action": action, "description": description}
