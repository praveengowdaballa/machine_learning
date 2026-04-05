# Import Agent framework
from google.adk.agents.llm_agent import Agent

# Import LLM model wrapper (Ollama / LiteLLM)
from google.adk.models.lite_llm import LiteLlm

# Import your compiled onboarding logic
# NOTE: Import from .core (important for compiled packages)
from ocp_gcp_onboarder.core import main


# ============================================================
# TOOL FUNCTION (this is what LLM will call)
# ============================================================
def run_onboarding_agent(project: str = None) -> dict:
    """
    Tool: Executes Jira onboarding workflow.

    Args:
        project (optional): Filter by project name

    Returns:
        dict: Structured response for LLM to understand
    """
    try:
        # ----------------------------------------------------
        # Call your core logic
        # IMPORTANT: Keep arguments optional for LLM flexibility
        # ----------------------------------------------------
        result = main(project) if project else main()

        # ----------------------------------------------------
        # Ensure structured output (CRITICAL for LLM)
        # If your main() already returns dict → keep it
        # If not → wrap it safely
        # ----------------------------------------------------
        if isinstance(result, dict):
            data = result
        else:
            data = {
                "raw_output": str(result)
            }

        # ----------------------------------------------------
        # Debug log (helps during development)
        # You can remove later
        # ----------------------------------------------------
        print("DEBUG: Tool Output ->", data)

        # ----------------------------------------------------
        # Final structured response
        # ----------------------------------------------------
        return {
            "status": "success",
            "data": data
        }

    except Exception as e:
        # ----------------------------------------------------
        # Proper error handling for LLM visibility
        # ----------------------------------------------------
        return {
            "status": "error",
            "message": str(e)
        }


# ============================================================
# ROOT AGENT DEFINITION
# ============================================================
root_agent = Agent(
    # --------------------------------------------------------
    # LLM Model (Ollama local model)
    # --------------------------------------------------------
    model=LiteLlm(model='ollama_chat/llama3.1:latest'),

    # --------------------------------------------------------
    # Agent Identity
    # --------------------------------------------------------
    name='cloud_intake_agent',
    description="Processes GCP onboarding requests from Jira",

    # --------------------------------------------------------
    # Instruction (VERY IMPORTANT — controls behavior)
    # --------------------------------------------------------
    instruction="""
You are a Cloud Intake Agent responsible for handling Jira onboarding requests.

RULES:
- ALWAYS use the 'run_onboarding_agent' tool when user asks about:
  - onboarding
  - Jira tickets
  - project setup
  - cloud intake

- The tool returns structured data in JSON format.

- DO NOT assume results.
- DO NOT say "no tickets" unless tool confirms it.

- Always summarize the returned data clearly for the user.

Example:
If tickets exist → list them.
If none → clearly say no onboarding tickets found.
""",

    # --------------------------------------------------------
    # Register tools for the agent
    # --------------------------------------------------------
    tools=[run_onboarding_agent],
)
