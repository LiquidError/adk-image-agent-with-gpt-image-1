import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Logging Settings ---
# Base logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO").upper()
# Set to True to see detailed logs from ADK and other libraries
DETAILED_FRAMEWORK_LOGGING = os.getenv("DETAILED_FRAMEWORK_LOGGING", "False").lower() == "true"

# --- OpenAI Settings ---
OPENAI_IMAGE_SIZE: str = "1024x1024" # Options: 1024x1024, 1536x1024, 1024x1536
OPENAI_IMAGE_QUALITY: str = "low" # Options: high, medium, low, auto
OPENAI_IMAGE_MODEL: str = "gpt-image-1" # dall-e-2, dall-e-3
OPENAI_IMAGE_MODERATION: str = "low" # Options: low, or auto
# NEW Image Saving Settings
SAVE_IMAGES_LOCALLY: bool = True # Set to False to disable local saving
LOCAL_IMAGE_SAVE_PATH: str = "local_image_results" # Directory relative to project root

# Save prompt to file
SAVE_PROMPT_TO_FILE: bool = True

# Load API keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Optional: Add checks to ensure the keys were loaded
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found in environment variables.")
    # raise ValueError("GOOGLE_API_KEY not found in environment variables.")

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found in environment variables.")
    # raise ValueError("OPENAI_API_KEY not found in environment variables.")

# --- End API Key Checks ---

# --- Logging Configuration Function ---
import logging
import sys # Import sys to access stdout

def setup_logging():
    """Configures logging based on settings."""
    # Get the root logger
    root_logger = logging.getLogger()
    # Set the base level from settings
    root_logger.setLevel(LOGGING_LEVEL)

    # Remove existing handlers (if any) to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Define a stylish format
    log_format = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Create a handler (e.g., StreamHandler to output to console)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Add the handler to the root logger
    root_logger.addHandler(handler)

    # If detailed logging is OFF, set noisy libraries to WARNING
    if not DETAILED_FRAMEWORK_LOGGING:
        noisy_loggers = [
            "google.adk",
            "openai",
            "google.generativeai",
            "urllib3",
            "httpx",
            # Add other noisy library root names here if needed
        ]
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
            logging.info(f"Set logger '{logger_name}' to WARNING level.") # Confirm setting
        
        # Suppress specific verbose warnings unless detailed logging is on
        logging.getLogger("google_genai.types").setLevel(logging.ERROR)
        logging.info("Set logger 'google_genai.types' to ERROR level.")

    logging.info(f"Logging configured. Base level: {LOGGING_LEVEL}, Detailed framework logs: {DETAILED_FRAMEWORK_LOGGING}")

# --- End Logging Configuration ---