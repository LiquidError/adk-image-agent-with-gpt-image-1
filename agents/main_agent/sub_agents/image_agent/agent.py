import logging
from google.adk.agents import Agent


from ...tools.image_generation import (
    openai_image_generation_tool,
    prompt_enhance_tool,
    image_edit_tool,
    image_mask_edit_tool,
    mask_generation_tool,
    combine_images_tool,
    clear_image_state_tool,
    save_artifact_to_state_tool,
)
from .prompt import IMAGE_AGENT_INSTR

logger = logging.getLogger(__name__)

# Define the Image Generation Agent
image_agent = Agent(
    name="image_agent",
    model="gemini-2.0-flash-001", 
    description="Handles user requests related to image generation and image prompt enhancement.",
    instruction=IMAGE_AGENT_INSTR,
    tools=[
        prompt_enhance_tool,
        openai_image_generation_tool,
        image_edit_tool,
        image_mask_edit_tool,
        mask_generation_tool,
        combine_images_tool,
        clear_image_state_tool,
        save_artifact_to_state_tool,
    
    ],
    # after_tool_callback=
    # before_tool_callback=
    # before_agent_callback=
    # after_agent_callback=
    
    # Example of enabling debug logging for this specific agent if needed
    # log_level=logging.DEBUG 
)

logger.info(f"image_agent '{image_agent.name}' initialized with tools.") 
