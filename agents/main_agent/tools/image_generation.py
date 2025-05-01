from datetime import datetime
import logging
import os
import pathlib
from typing import Dict, Any, List
import base64
import uuid
from google.genai import types 
from google import genai
from openai import OpenAI
from PIL import Image
from io import BytesIO
from . import settings
from google.adk.tools import ToolContext, FunctionTool, BaseTool
from google.adk.agents.callback_context import CallbackContext
from . import image_prompt_examples
def _decode_b64_str(s: str) -> bytes:
    """Decode a base64 string, stripping data URL prefixes and adding padding."""
    if isinstance(s, str) and s.startswith("data:"):
        parts = s.split(",", 1)
        if len(parts) == 2:
            s = parts[1]
    s = s.strip()
    padding = len(s) % 4
    if padding:
        s += "=" * (4 - padding)
    return base64.b64decode(s)

logger = logging.getLogger(__name__)

# --- Image Generation Tool Implementation

def log_prompt_to_file(formatted_prompt: str, raw_llm_response: Any, final_enhanced_prompt: str) -> None:
    """Log the formatted prompt sent to the LLM, the raw response, and the final enhanced prompt string."""
    if not settings.SAVE_PROMPT_TO_FILE:
        return
    try:
        # Create the logs/prompts directory if it doesn't exist
        PROMPT_LOG_DIR = pathlib.Path("logs/prompts")
        PROMPT_LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Use date in filename for daily rotation
        today = datetime.now().strftime("%Y%m%d")
        prompt_log_file = PROMPT_LOG_DIR / f"prompts_image_gen_{today}.log"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(prompt_log_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n{'='*50}\n")
            f.write(f"TIMESTAMP: {timestamp}\n")
            f.write(f"{'='*50}\n\n")
            f.write("--- Formatted Prompt Sent to LLM ---\n")
            f.write(formatted_prompt) # Log the fully formatted prompt
            f.write("\n\n--- Raw LLM Response Object ---\n")
            # Attempt to convert the raw response to a string safely
            try:
                response_str = str(raw_llm_response)
            except Exception as str_err:
                response_str = f"<Error converting response object to string: {str_err}>"
            f.write(response_str)
            f.write("\n\n--- Final Enhanced Prompt String Used ---\n")
            f.write(final_enhanced_prompt) # Log the final extracted prompt
            f.write(f"\n\n{'='*50}\n")

    except Exception as e:
        logger.error(f"Failed to log prompt to file: {e}")
        import traceback
        logger.error(f"Prompt logging error details: {traceback.format_exc()}")

# This is the callback version, keep it for uploads
async def _save_uploaded_image_to_state_callback(callback_context: CallbackContext):
    """Extracts image data from incoming message, base64 encodes it, and saves to session state."""
    logger.info("--- Entering _save_uploaded_image_to_state_callback ---")
    state = callback_context.state
    user_content = callback_context.user_content

    if not user_content or not user_content.parts:
        logger.info("Callback: No content or parts found in user_content.")
        return

    image_parts_list = [] # List to store all image parts
    first_image_b64 = None # To maintain compatibility with single-edit tools


    if "uploaded_image_b64" in state:
        del state["uploaded_image_b64"]
    if "uploaded_mask_b64" in state:
        del state["uploaded_mask_b64"]
    if "uploaded_image_parts" in state:
        del state["uploaded_image_parts"]
    logger.debug("Callback: Cleared existing image/mask state keys.")

    # Iterate through parts of the user_content
    for i, part in enumerate(user_content.parts):
        if hasattr(part, 'inline_data') and getattr(part.inline_data, 'mime_type', '').startswith('image/'):
            mime_type = part.inline_data.mime_type
            logger.debug(f"Callback: Found image part {i} with mime_type: {mime_type}")
            part_data_container = part.inline_data
            current_data_bytes = None

            # Try extracting raw bytes first
            raw_bytes = getattr(part_data_container, 'data', None)
            if raw_bytes and isinstance(raw_bytes, (bytes, bytearray)):
                current_data_bytes = raw_bytes
                logger.debug(f"Callback: Extracted raw bytes ({len(current_data_bytes)}) from inline_data for part {i}.")
            else:
                # Fallback to base64 string
                b64_data = getattr(part_data_container, 'b64_json', None)
                if b64_data and isinstance(b64_data, str):
                    logger.debug(f"Callback: Found base64 string in inline_data for part {i}. Decoding...")
                    try:
                        current_data_bytes = _decode_b64_str(b64_data)
                        logger.debug(f"Callback: Decoded base64 part {i}. Length: {len(current_data_bytes)}")
                    except Exception as e:
                        logger.error(f"Callback: Failed to decode base64 from part {i}: {e}")
                        continue # Skip this part if decoding fails
                else:
                    logger.warning(f"Callback: Image part {i} found but no suitable data/b64_json.")
                    continue

            # If data was successfully obtained, process and add to list
            if current_data_bytes:
                try:
                    # Re-encode to ensure clean base64 for state
                    current_b64_str = base64.b64encode(current_data_bytes).decode('utf-8')
                    image_part_info = {"b64": current_b64_str, "mime_type": mime_type}
                    image_parts_list.append(image_part_info)
                    logger.debug(f"Callback: Added image part {i} info (mime: {mime_type}, b64 len: {len(current_b64_str)}) to list.")

                    # Save the first image separately for compatibility
                    if first_image_b64 is None:
                        first_image_b64 = current_b64_str
                        logger.debug(f"Callback: Storing part {i} as the first image for single-edit compatibility.")

                except Exception as e:
                    logger.error(f"Callback: Error processing/encoding image part {i}: {e}")
                    continue

    # Save the collected image parts list and the first image to state
    if image_parts_list:
        state["uploaded_image_parts"] = image_parts_list
        logger.info(f"Callback: Saved list of {len(image_parts_list)} image parts to state['uploaded_image_parts'].")
    else:
        logger.info("Callback: No valid image parts found to save to the list.")

    if first_image_b64:
        state["uploaded_image_b64"] = first_image_b64
        logger.info(f"Callback: Saved first image base64 string (len: {len(first_image_b64)}) to state['uploaded_image_b64'].")
    else:
        # If no images were found at all, ensure the key is removed
        if "uploaded_image_b64" in state:
            del state["uploaded_image_b64"]
        logger.info("Callback: No first image found, ensuring state['uploaded_image_b64'] is clear.")

    logger.info("--- Exiting _save_uploaded_image_to_state_callback ---")

save_uploaded_image_to_state_tool = FunctionTool(func=_save_uploaded_image_to_state_callback) 

# Tool function to save image as artifact and conditionally save locally
async def _image_save_func(image_bytes: bytes, file_extension: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Saves image bytes as an ADK artifact and conditionally saves a copy locally."""
    logger.debug("Entering _image_save_func (artifact save version)...")

    # Determine mime type and ensure extension format
    if file_extension and not file_extension.startswith('.'):
        file_extension = '.' + file_extension
    mime_type = f"image/{file_extension.lstrip('.')}" if file_extension else "image/png"
    # Make filename unique for the artifact itself and local copy
    filename = f"generated_image_{uuid.uuid4()}{file_extension if file_extension else '.png'}" 

    # --- Conditional Local Save Logic  --- 
    if settings.SAVE_IMAGES_LOCALLY:
        save_dir = settings.LOCAL_IMAGE_SAVE_PATH
        try:
            os.makedirs(save_dir, exist_ok=True)
            # Use the unique filename generated above for local copy consistency
            local_path = os.path.join(save_dir, filename) 
            
            logger.info(f"Attempting to save image locally to: {local_path} (SAVE_IMAGES_LOCALLY is True)")
            with open(local_path, "wb") as f:
                f.write(image_bytes)
            logger.info(f"Successfully saved image locally to: {local_path}")
        except Exception as e:
            # Log error but don't stop the main process
            logger.warning(f"Local file save failed (path: {save_dir}): {e}", exc_info=False)
    else:
        logger.debug("Local image saving skipped (SAVE_IMAGES_LOCALLY is False).")
    # --- End Conditional Local Save Logic ---
    
    # --- Save as ADK Artifact --- 
    artifact_version = None # Initialize artifact_version variable
    local_path_if_saved = local_path if settings.SAVE_IMAGES_LOCALLY and 'local_path' in locals() else None
    try:
        logger.info(f"Saving {len(image_bytes)} bytes as ADK artifact (name: {filename}, mime: {mime_type})")
        # Create a Part object for the artifact data
        artifact_part = types.Part(inline_data=types.Blob(data=image_bytes, mime_type=mime_type))
        
        # Call save_artifact with filename and artifact keywords 
        logger.debug(f"--->>> PRE-SAVE ARTIFACT CALL (sync) for {filename}")
        artifact_version = tool_context.save_artifact( 
            filename=filename, 
            artifact=artifact_part 
        )
        # Log using filename and returned version
        logger.info(f"Successfully saved artifact {filename} (version {artifact_version})")
    except Exception as e:
        logger.error(f"Failed to save ADK artifact: {e}", exc_info=True)
        # If artifact saving fails, return specific info if local save happened
        error_msg = f"Failed to save image as artifact: {e}"
        if local_path_if_saved:
             # Return confirmation of local save + artifact error, avoiding generic "error" key
             logger.warning(f"Artifact save failed, but local copy exists at {local_path_if_saved}")
             return {
                 "confirmation": f"Image saved locally to {local_path_if_saved}. Failed to save as artifact: {e}", 
                 "local_path": local_path_if_saved,
                 "artifact_error": str(e) # Specific key for artifact error
             }
        else:
             # Only return generic error if local save didn't happen/succeed
             return {"error": error_msg}
    # --- End Artifact Save ---

    # --- Return confirmation and artifact info (NO raw data) ---
    confirmation_msg = f"Image artifact {filename} (version {artifact_version}) saved successfully."
    if local_path_if_saved:
        confirmation_msg += f" Local copy saved to {local_path_if_saved}."
        
    logger.info("Returning confirmation and artifact info to agent.")
    return {
        "filename": filename, 
        "artifact_version": artifact_version, 
        "confirmation": confirmation_msg
    }
    # --- End Return ---

image_save_tool = FunctionTool(func=_image_save_func)

# Helper function for prompt enhancement 
async def _enhance_prompt_for_image_gen(desc: str) -> str | None:
    """Enhances the user-provided description into a detailed prompt using Gemini."""
    logger.debug(f"Enhancing prompt for description: '{desc}' with Gemini Flash")
    try:
       
        # Initialize client (using API key from settings)
        client = genai.Client(api_key=settings.GOOGLE_API_KEY)

        # Construct the prompt string directly
        prompt_text = (
            "You are a creative image generation assistant. Enhance the following user request "
            f"into a detailed and vivid image generation prompt. User request: '{desc}'\n"
            "Examples:\n"
            f"- {image_prompt_examples.ENHANCE_PROMPT_CHARACTER[0]}\n"
            f"- {image_prompt_examples.ENHANCE_PROMPT_YOUTUBE_THUMBNAIL[0]}\n"
            "Return ONLY the enhanced prompt string."
        )

        logger.debug("Sending prompt to genai.generate_content...")
        # Call generate_content
        response = client.models.generate_content(
            model="gemini-2.0-flash-001", # Use the specified model
            contents=prompt_text, # Pass the prompt string directly
            config=types.GenerateContentConfig(
                temperature=0.3, # Use the specified temperature
            )
        )
        logger.debug(f"Received response from genai.generate_content: {response}")

        # Extract the text from the response
        detailed_prompt = response.text
        if not detailed_prompt:
             logger.error(f"Gemini prompt enhancement failed: No text in response: {response}")
             return None

        logger.info(f"Enhanced prompt for {desc}: '{detailed_prompt}'")
        # Log the constructed prompt and the raw response object, along with the final extracted text
        log_prompt_to_file(prompt_text, response, detailed_prompt.strip())

        return detailed_prompt.strip() # Return the string
    except Exception as e:
        logger.error(f"Gemini prompt enhancement failed: {e}", exc_info=True)
        return None # Indicate failure


# Wrap prompt enhancement as a tool
prompt_enhance_tool = FunctionTool(func=_enhance_prompt_for_image_gen)

# --- New OpenAI GPT Image Generation Tool ---
async def _generate_image_with_openai(desc: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Generates a new image using GPT-Image-1 model based on a text description.
    Use this for creating completely new images from text descriptions.
    
    Args:
        desc: A detailed text description of the image to generate.
        tool_context: The ADK tool context for accessing state and services.
        
    Returns:
        A dictionary containing information about the generated image, including
        the artifact name and version where the image is stored.
    """

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    try:
        response = client.images.generate(
            model=settings.OPENAI_IMAGE_MODEL,
            prompt=desc,
            size=settings.OPENAI_IMAGE_SIZE,
            quality=settings.OPENAI_IMAGE_QUALITY,
            moderation=settings.OPENAI_IMAGE_MODERATION
        )
        b64 = response.data[0].b64_json
        data = base64.b64decode(b64)

        # --- Save the artifact ---
        artifact_result = await _image_save_func(data, ".png", tool_context)
        logger.info(f"Generate artifact save result keys: {artifact_result.keys()}")

        # --- Manually construct return dictionary ---
        if "error" in artifact_result or "artifact_error" in artifact_result:
             error_key = "error" if "error" in artifact_result else "artifact_error"
             confirm = artifact_result.get("confirmation", "Image generated but artifact save failed.") + f" Error: {artifact_result.get(error_key)}"
             return {"confirmation": confirm, "artifact_error": artifact_result.get(error_key)}
        else:
            # Return the original dict from _image_save_func if successful
            # This ensures filename/version are present for UI trigger
            return artifact_result
    
    except Exception as e:
        logger.error(f"OpenAI image generation failed: {e}", exc_info=True)
        return {"error": str(e)}

# Create the FunctionTool for OpenAI image generation
openai_image_generation_tool = FunctionTool(func=_generate_image_with_openai)

# --- OpenAI Image Edit Tool ---
async def _edit_image(prompt: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Edit an existing image (loaded from state) based on a prompt."""
    logger.debug("--- Entering _edit_image (state loading version) ---")

    # Attempt to load base64 string from state
    image_parts = tool_context.state.get("uploaded_image_parts")
    image_b64_str = None
    if image_parts and isinstance(image_parts, list) and len(image_parts) > 0 and isinstance(image_parts[0], dict):
        image_b64_str = image_parts[0].get("b64") # Use .get() for safer access
    else:
        logger.error("Cannot edit image: 'uploaded_image_parts' structure in state is unexpected or empty.")
        return {"error": "Cannot edit image: Image data structure in state is unexpected or missing."}

    raw = None

    if image_b64_str and isinstance(image_b64_str, str):
        logger.debug(f"Found image base64 string (len: {len(image_b64_str)}) in state. Decoding...")
        try:
            raw = _decode_b64_str(image_b64_str) # Decode the string from state
            logger.debug(f"Successfully decoded state image data ({len(raw)} bytes).)")
        except Exception as e:
            logger.error(f"Failed to decode base64 string from state: {e}", exc_info=True)
            return {"error": f"Failed to decode image data stored in session state: {e}"}
    else:
        logger.error("Image base64 string not found or invalid in state['uploaded_image_b64'].")
        return {"error": "Image data not found in session state. Please ensure the image was uploaded correctly."}

    # Ensure raw bytes were obtained
    if not raw:
         # This case should be caught by the error handling above, but for safety:
         logger.error("Image data (raw) is None after attempting to load and decode from state.")
         return {"error": "Could not obtain image data from state."}

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    try:
        # --- Image Processing (using decoded state data) ---
        logger.debug(f"Input image (raw) first 100 bytes: {raw[:100]}")
        img = Image.open(BytesIO(raw)).convert("RGBA")
        # No explicit mask needed for full image edit

        # Prepare image file for OpenAI API
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        img_file_tuple = ("image.png", img_byte_arr, "image/png")

        logger.debug(f"Sending edit request to OpenAI with prompt: '{prompt}'")
        response = client.images.edit(
            model=settings.OPENAI_IMAGE_MODEL,
            image=img_file_tuple,
            prompt=prompt,
            n=1,
            size=settings.OPENAI_IMAGE_SIZE,
            quality=settings.OPENAI_IMAGE_QUALITY,
        )

        b64 = response.data[0].b64_json
        data = base64.b64decode(b64)

        # --- Save the artifact ---
        artifact_result = await _image_save_func(data, ".png", tool_context)
        logger.info(f"Edit artifact save result keys: {artifact_result.keys()}")

        # --- Manually construct return dictionary ---
        if "error" in artifact_result or "artifact_error" in artifact_result:
             error_key = "error" if "error" in artifact_result else "artifact_error"
             confirm = artifact_result.get("confirmation", "Image generated but artifact save failed.") + f" Error: {artifact_result.get(error_key)}"
             return {"confirmation": confirm, "artifact_error": artifact_result.get(error_key)}
        else:
            # Return the original dict from _image_save_func if successful
            # This ensures filename/version are present for UI trigger
            return artifact_result

    except ImportError:
        logger.error("Pillow library not installed. Please install it: pip install Pillow")
        return {"error": "Image processing library (Pillow) not found."}
    except Exception as e:
        logger.error(f"OpenAI image edit failed: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred during image editing: {e}"}

image_edit_tool = FunctionTool(func=_edit_image)

# --- OpenAI Image Masked Edit Tool ---
async def _edit_image_with_mask(prompt: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Edit an existing image in masked regions (loaded from state) based on a prompt."""
    logger.debug("--- Entering _edit_image_with_mask (state loading version) ---")

    # Load base64 strings from state
    image_parts = tool_context.state.get("uploaded_image_parts")
    image_b64_str = None
    if image_parts and isinstance(image_parts, list) and len(image_parts) > 0 and isinstance(image_parts[0], dict):
        image_b64_str = image_parts[0].get("b64") # Use .get() for safer access
    else:
        logger.error("Cannot edit image with mask: 'uploaded_image_parts' structure in state is unexpected or empty.")
        return {"error": "Cannot edit image with mask: Image data structure in state is unexpected or missing."}

    mask_b64_str = tool_context.state.get("uploaded_mask_b64")
    raw_image = None
    raw_mask = None

    # Decode image
    if image_b64_str and isinstance(image_b64_str, str):
        logger.debug(f"Found image b64 string (len {len(image_b64_str)}) in state. Decoding...")
        try:
            raw_image = _decode_b64_str(image_b64_str)
            logger.debug(f"Decoded image data ({len(raw_image)} bytes)." )
        except Exception as e:
             logger.error(f"Failed to decode image base64 from state: {e}")
             # Potentially return error early
    else:
        logger.error("Original image base64 string not found or invalid in state.")
        return {"error": "Original image data not found in session state."}

    # Decode mask
    if mask_b64_str and isinstance(mask_b64_str, str):
        logger.debug(f"Found mask b64 string (len {len(mask_b64_str)}) in state. Decoding...")
        try:
            raw_mask = _decode_b64_str(mask_b64_str)
            logger.debug(f"Decoded mask data ({len(raw_mask)} bytes)." )
        except Exception as e:
             logger.error(f"Failed to decode mask base64 from state: {e}")
             # Potentially return error early
    else:
        logger.error("Mask image base64 string not found or invalid in state.")
        return {"error": "Mask image data not found in session state. Please upload both image and mask."}

    # Validation
    if not raw_image:
        return {"error": "Could not obtain valid original image data from state."}
    if not raw_mask:
        return {"error": "Could not obtain valid mask image data from state."}

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    try:
        # --- Image Processing (using decoded state data) ---
        logger.debug(f"Processing original image. Length: {len(raw_image)}, First 100 bytes: {raw_image[:100]}")
        logger.debug(f"Processing mask image. Length: {len(raw_mask)}, First 100 bytes: {raw_mask[:100]}")

        try:
            img = Image.open(BytesIO(raw_image))
            mask = Image.open(BytesIO(raw_mask)).convert("RGBA")
            if img.size != mask.size:
                 logger.error(f"Image size {img.size} and mask size {mask.size} do not match.")
                 return {"error": "Image and mask dimensions must be the same."}
        except ImportError:
            logger.error("Pillow library not installed. Please install it: pip install Pillow")
            return {"error": "Image processing library (Pillow) not found."}
        except Image.UnidentifiedImageError as img_err:
            logger.error(f"Pillow could not identify image or mask file from state: {img_err}", exc_info=True)
            return {"error": f"Cannot identify image or mask file loaded from state. Data may be corrupted: {img_err}"}

        # Prepare files for OpenAI API
        img_byte_arr = BytesIO(raw_image)
        img_file_tuple = ("image.png", img_byte_arr, "image/png")
        mask_byte_arr = BytesIO(raw_mask)
        mask_file_tuple = ("mask.png", mask_byte_arr, "image/png")

        logger.debug(f"Sending masked edit request to OpenAI with prompt: '{prompt}'")
        response = client.images.edit(
            model=settings.OPENAI_IMAGE_MODEL,
            image=img_file_tuple,
            mask=mask_file_tuple,
            prompt=prompt,
            n=1,
            size=settings.OPENAI_IMAGE_SIZE,
            quality=settings.OPENAI_IMAGE_QUALITY,
        )

        b64 = response.data[0].b64_json
        data = base64.b64decode(b64)

        # --- Save the artifact ---
        artifact_result = await _image_save_func(data, ".png", tool_context)
        logger.info(f"Masked Edit artifact save result keys: {artifact_result.keys()}")

        # --- Manually construct return dictionary ---
        if "error" in artifact_result or "artifact_error" in artifact_result:
             error_key = "error" if "error" in artifact_result else "artifact_error"
             confirm = artifact_result.get("confirmation", "Masked image edit done but artifact save failed.") + f" Error: {artifact_result.get(error_key)}"
             return {"confirmation": confirm, "artifact_error": artifact_result.get(error_key)}
        else:
             # Return the original dict from _image_save_func if successful
             return artifact_result

    except Exception as e:
        logger.error(f"OpenAI masked image edit failed: {e}", exc_info=True)
        error_msg = f"An unexpected error occurred during masked image editing: {e}"
        if 'mask' in str(e).lower() and 'alpha' in str(e).lower():
             error_msg += " Ensure the mask is a valid RGBA PNG image."
        return {"error": error_msg}

image_mask_edit_tool = FunctionTool(func=_edit_image_with_mask)

# --- Mask Generation Tool (Using OpenAI Edit Endpoint) ---
async def _generate_mask(object_description: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Generates a segmentation mask using the OpenAI edit endpoint and saves it to state."""
    logger.debug(f"--- Entering _generate_mask (OpenAI version) for object: '{object_description}' ---")

    # 1. Load original image b64 from state
    image_parts = tool_context.state.get("uploaded_image_parts")
    # Correctly access the 'b64' key from the dictionary in the list
    image_b64_str = None
    if image_parts and isinstance(image_parts, list) and len(image_parts) > 0 and isinstance(image_parts[0], dict):
        image_b64_str = image_parts[0].get("b64") # Use .get() for safer access
    else:
        logger.error("Cannot generate mask: 'uploaded_image_parts' structure in state is unexpected or empty.")
        return {"error": "Cannot generate mask: Image data structure in state is unexpected or missing."}

    if not image_b64_str or not isinstance(image_b64_str, str):
        logger.error("Cannot generate mask: Original image base64 string not found or invalid in state.")
        return {"error": "Original image data not found in session state. Cannot generate mask."}

    try:
        # Keep raw bytes for API call
        raw_image_bytes = _decode_b64_str(image_b64_str)
        original_image = Image.open(BytesIO(raw_image_bytes))
        img_width, img_height = original_image.size
        logger.debug(f"Loaded original image from state ({img_width}x{img_height}).")
    except ImportError:
        logger.error("Pillow library not installed. Cannot process image for mask generation.")
        return {"error": "Image processing library (Pillow) not found."}
    except Exception as e:
        logger.error(f"Failed to load or decode original image from state for mask generation: {e}", exc_info=True)
        return {"error": f"Failed to process original image from state: {e}"}

    # 2. Prepare the mask generation prompt
    mask_prompt = (
        f"Generate a mask delimiting the '{object_description}' in the picture. "
        f"Use white (#FFFFFF) where the '{object_description}' is located and black (#000000) for the background. "
        f"Return an image in the same size as the input image."
    )
    logger.debug(f"Using mask generation prompt: '{mask_prompt}'")

    # 3. Prepare image file for OpenAI API
    try:
        img_byte_arr = BytesIO(raw_image_bytes)
        img_file_tuple = ("image.png", img_byte_arr, "image/png")
    except Exception as e:
        logger.error(f"Failed to prepare image bytes for API: {e}", exc_info=True)
        return {"error": f"Failed to prepare image data for mask generation: {e}"}

    # 4. Call OpenAI images.edit endpoint to generate the mask
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    try:
        logger.debug("Sending mask generation request to OpenAI...")
        response = client.images.edit(
            model=settings.OPENAI_IMAGE_MODEL, # Use the same model as edits
            image=img_file_tuple,
            prompt=mask_prompt,
            n=1,
            size=f"{img_width}x{img_height}",
            quality=settings.OPENAI_IMAGE_QUALITY,
        )

        # 5. Process the response (expecting b64_json)
        mask_b64_generated = response.data[0].b64_json
        if not mask_b64_generated:
            logger.error("OpenAI mask generation response did not contain b64_json data.")
            try:
                logger.error(f"Full OpenAI response object: {response}")
            except Exception:
                 logger.error("Could not log full OpenAI response object.")
            return {"error": "Failed to retrieve mask image data (b64_json) from OpenAI."}

        logger.debug(f"Received generated mask b64_json from OpenAI (len: {len(mask_b64_generated)}). Converting to RGBA...")

        # --- Convert generated mask to RGBA PNG before saving --- 
        try:
            mask_data_bytes = base64.b64decode(mask_b64_generated)
            mask_image = Image.open(BytesIO(mask_data_bytes))
            rgba_mask_image = mask_image.convert("RGBA")
            
            # Save the RGBA mask to bytes
            rgba_mask_byte_arr = BytesIO()
            rgba_mask_image.save(rgba_mask_byte_arr, format="PNG")
            rgba_mask_byte_arr.seek(0)
            final_mask_bytes = rgba_mask_byte_arr.getvalue()
            
            # Re-encode the final RGBA mask bytes to base64
            final_mask_b64 = base64.b64encode(final_mask_bytes).decode('utf-8')
            logger.debug("Successfully converted generated mask to RGBA PNG format.")
            
        except Exception as convert_err:
            logger.error(f"Failed to convert generated mask to RGBA PNG: {convert_err}", exc_info=True)
            return {"error": f"Failed to process the generated mask image: {convert_err}"}
        # --- End Conversion ---

        # --- Update State BEFORE Saving Artifact (Mask Specific) ---
        # This is the state update that casues the UI to not get the artifact.
        try:
            tool_context.state["uploaded_mask_b64"] = final_mask_b64 # Update the specific mask key
            logger.debug("Mask state update completed BEFORE artifact save.")
        except Exception as state_err:
            logger.error(f"Failed to update state before mask generation artifact save: {state_err}", exc_info=True)
            # Log error, but proceed to save artifact anyway

        # --- Save the generated mask as an artifact ---
        logger.debug("Saving generated mask as an artifact...")
        artifact_result = await _image_save_func(final_mask_bytes, ".png", tool_context)
        logger.info(f"Generate Mask artifact save result keys: {artifact_result.keys()}")

        # --- Manually construct return dictionary ---
        if "error" in artifact_result or "artifact_error" in artifact_result:
             error_key = "error" if "error" in artifact_result else "artifact_error"
             confirm = artifact_result.get("confirmation", "Mask generated but artifact save failed.") + f" Error: {artifact_result.get(error_key)}"
             return {"confirmation": confirm, "artifact_error": artifact_result.get(error_key)}
        else:
            return {
                "confirmation": artifact_result.get("confirmation", "Mask generated and saved."),
                "artifact_filename": artifact_result.get("filename"),
                "artifact_version": artifact_result.get("artifact_version")
            }

    except Exception as e:
        logger.error(f"OpenAI mask generation API call failed: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred during mask generation: {e}"}

mask_generation_tool = FunctionTool(func=_generate_mask)

# --- Tool to Save Artifact Data to State ---

def _save_image_artifact_to_state(filename: str, version: int, tool_context: ToolContext) -> Dict[str, str]:
    """Loads an existing image artifact and saves its data to the session state 
       ('uploaded_image_b64' and 'uploaded_image_parts') for subsequent editing. 
       Call this *after* generating or editing an image if further modifications are needed.
       Args:
           filename: The filename of the artifact to load.
           version: The version number of the artifact to load.
           tool_context: The ADK tool context.
       Returns:
           A dictionary with a confirmation message or an error.
    """
    logger.info(f"--- Entering _save_image_artifact_to_state for {filename} v{version} ---")
    state = tool_context.state

    try:
        # 1. Load the artifact
        logger.debug(f"Attempting to load artifact: {filename} v{version}")
        artifact_part = tool_context.load_artifact(filename, version=version)

        if not artifact_part:
            logger.error(f"Artifact {filename} v{version} not found.")
            return {"error": f"Artifact {filename} v{version} not found."}
        
        if not hasattr(artifact_part, 'inline_data') or not hasattr(artifact_part.inline_data, 'data'):
            logger.error(f"Artifact {filename} v{version} does not contain expected data structure.")
            return {"error": f"Artifact {filename} v{version} has invalid data structure."}

        image_bytes = artifact_part.inline_data.data
        mime_type = artifact_part.inline_data.mime_type
        logger.debug(f"Loaded artifact data: {len(image_bytes)} bytes, mime: {mime_type}")

        # 2. Encode and Update State
        image_b64_str = base64.b64encode(image_bytes).decode('utf-8')
        image_part_info = {"b64": image_b64_str, "mime_type": mime_type}
        image_parts_list = [image_part_info]

        state["uploaded_image_parts"] = image_parts_list
        state["uploaded_image_b64"] = image_b64_str
        # Optional: Clear mask state when saving a new base image?
        # state["uploaded_mask_b64"] = None

        confirmation_msg = f"Successfully loaded artifact {filename} v{version} and updated session state for editing."
        logger.info(confirmation_msg)
        return {"confirmation": confirmation_msg}

    except ValueError as e:
        logger.error(f"Error loading artifact {filename} v{version}: {e}", exc_info=True)
        return {"error": f"Error accessing artifact service: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error saving artifact {filename} v{version} to state: {e}", exc_info=True)
        return {"error": f"Unexpected error processing artifact: {e}"}

save_artifact_to_state_tool = FunctionTool(func=_save_image_artifact_to_state)

# --- Image Combination Tool ---
async def _combine_images(prompt: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Combines multiple images loaded from state based on a prompt using OpenAI."""
    logger.debug(f"--- Entering _combine_images tool for prompt: '{prompt}' ---")

    # 1. Load image parts list from state
    image_parts_list = tool_context.state.get("uploaded_image_parts")
    if not image_parts_list or not isinstance(image_parts_list, list) or len(image_parts_list) < 2:
        error_msg = "Cannot combine images: Requires at least two images previously uploaded in the session."
        logger.error(error_msg + f" Found {len(image_parts_list) if image_parts_list else 0} parts in state.")
        return {"error": error_msg}

    logger.debug(f"Found {len(image_parts_list)} image parts in state for combination.")

    # 2. Prepare image file tuples for OpenAI API
    image_tuples_for_api = []
    try:
        for i, part_info in enumerate(image_parts_list):
            b64_string = part_info.get("b64")
            mime_type = part_info.get("mime_type", "image/png") # Default to png if missing
            if not b64_string:
                logger.warning(f"Skipping image part {i} due to missing 'b64' data.")
                continue
            
            image_bytes = base64.b64decode(b64_string)
            # Determine file extension - simplistic, might need improvement
            extension = "." + mime_type.split('/')[-1] if '/' in mime_type else ".png"
            filename = f"input_image_{i}{extension}"
            
            img_byte_arr = BytesIO(image_bytes)
            img_file_tuple = (filename, img_byte_arr, mime_type)
            image_tuples_for_api.append(img_file_tuple)
            logger.debug(f"Prepared image tuple: {filename}, mime: {mime_type}")
            
    except Exception as e:
        logger.error(f"Failed to prepare image data for API: {e}", exc_info=True)
        return {"error": f"Failed to prepare image data for combination: {e}"}

    if len(image_tuples_for_api) < 2:
        error_msg = "Cannot combine images: Need at least two valid images prepared for the API."
        logger.error(error_msg)
        return {"error": error_msg}

    # 3. Call OpenAI images.edit endpoint
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    try:
        logger.debug(f"Sending combination request to OpenAI with {len(image_tuples_for_api)} images and prompt: '{prompt}'")
        response = client.images.edit(
            model=settings.OPENAI_IMAGE_MODEL,
            image=image_tuples_for_api, # Pass the list of prepared image tuples
            prompt=prompt,
            n=1,
            size=settings.OPENAI_IMAGE_SIZE,
            quality=settings.OPENAI_IMAGE_QUALITY,
        )

        # 4. Process the response
        combined_b64 = response.data[0].b64_json
        if not combined_b64:
            logger.error("OpenAI image combination response did not contain b64_json data.")
            return {"error": "Failed to retrieve combined image data (b64_json) from OpenAI."}

        logger.debug(f"Received combined image b64_json from OpenAI (len: {len(combined_b64)}). Decoding...")
        combined_data_bytes = base64.b64decode(combined_b64)
        logger.debug(f"Successfully decoded combined image data ({len(combined_data_bytes)} bytes).")

        # 5. Update State: Replace the list with the single combined image - If this is enabled, the UI will not get the artifact.
        #try:
        #    state = tool_context.state # Get state
        #    state["uploaded_image_parts"] = [{"b64": combined_b64, "mime_type": "image/png"}] # Replace parts list
        #    state["uploaded_image_b64"] = combined_b64 # Save the new combined b64
        #    # Optionally clear mask state if it doesn't apply to combined image
        #    state["uploaded_mask_b64"] = None
        #    logger.debug("Updated state: Replaced parts list and saved combined image b64.")
        #except Exception as state_err:
        #    logger.error(f"Failed to update state after image combination: {state_err}", exc_info=True)
        #    # Proceed anyway, main task succeeded

        # 6. Save combined image artifact
        logger.debug("OpenAI image combination successful, calling save function.")
       
        # Save the combined image to the artifact service
        artifact_result = await _image_save_func(combined_data_bytes, "_combined.png", tool_context)
        # --- Manually construct return dictionary ---
        if "error" in artifact_result or "artifact_error" in artifact_result:
             error_key = "error" if "error" in artifact_result else "artifact_error"
             confirm = artifact_result.get("confirmation", "Mask generated but artifact save failed.") + f" Error: {artifact_result.get(error_key)}"
             return {"confirmation": confirm, "artifact_error": artifact_result.get(error_key)}
        else:
            return {
                "confirmation": artifact_result.get("confirmation", "Mask generated and saved."),
                "artifact_filename": artifact_result.get("filename"),
                "artifact_version": artifact_result.get("artifact_version")
            }

    except Exception as e:
        logger.error(f"OpenAI image combination API call failed: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred during image combination: {e}"}

combine_images_tool = FunctionTool(func=_combine_images)

async def _clear_image_state(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Clears image-related data (uploaded images, masks) from the current session state.
    Use this when the user wants to start over with a new image or process.
    """
    logger.debug("--- Entering _clear_image_state tool ---")
    state = tool_context.state
    keys_to_clear = [
        "uploaded_image_b64",
        "uploaded_mask_b64",
        "uploaded_image_parts",
        # Add any other image-specific state keys you might introduce later
    ]
    cleared_keys_log = []

    for key in keys_to_clear:
        # Set to None instead of pop/del
        if state.get(key) is not None:
            state[key] = None 
            logger.debug(f"Cleared key '{key}' in session state by setting to None.")
            cleared_keys_log.append(key)
        else:
            logger.debug(f"Key '{key}' not found or already None in session state, skipping.")

    confirmation_msg = f"Image session state cleared. Cleared keys: {cleared_keys_log}" if cleared_keys_log else "Image session state was already clear (or no relevant keys found)."
    logger.info(confirmation_msg)
    return {"confirmation": confirmation_msg}

# Create the FunctionTool instance for the agent
clear_image_state_tool = FunctionTool(func=_clear_image_state)

