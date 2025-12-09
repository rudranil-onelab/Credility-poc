"""
Configuration settings for the Document Validation API.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# AWS Configuration
# IMPORTANT: Region must match the S3 bucket region (lendingwise-poc is in us-east-1)
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
if AWS_REGION != "us-east-1":
    print(f"[WARNING] AWS_REGION is set to {AWS_REGION}, but S3 bucket lendingwise-poc is in us-east-1. Overriding to us-east-1.")
    AWS_REGION = "us-east-1"
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# AWS Bedrock Configuration (Claude 3.5 Sonnet v2 for Vision support)
# Note: Claude 3.5 Haiku does NOT support vision/images - only text
# Claude 3.5 Sonnet v2 supports both text and vision
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")  # Use us-east-1 for best model availability
#BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")

# Alternative models with vision support:
# - anthropic.claude-3-5-sonnet-20241022-v2:0 (Claude 3.5 Sonnet v2 - recommended)
# - anthropic.claude-3-sonnet-20240229-v1:0 (Claude 3 Sonnet)
# - anthropic.claude-3-haiku-20240307-v1:0 (Claude 3 Haiku - has vision, cheaper)
# Text-only models (NO vision):
# - anthropic.claude-3-5-haiku-20241022-v1:0 (Claude 3.5 Haiku - text only!)

# S3 Configuration - POC bucket for Textract
S3_BUCKET = os.getenv("S3_BUCKET", "lendingwise-poc")

# Textract Configuration
TEXTRACT_MAX_WAIT_SECONDS = int(os.getenv("TEXTRACT_MAX_WAIT_SECONDS", "600"))
TEXTRACT_FEATURE_TYPES = ["TABLES", "FORMS"]

# Output Configuration
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Document Type Labels
ROUTE_LABELS = [
    # Identity Documents
    "aadhaar", "pan_card", "voter_id", "indian_passport", "indian_driving_license",
    "driving_license", "passport", "state_id", "identity",
    # Other Categories
    "bank_statement", "property", "entity", "loan",
    "invoice", "contract", "certificate",
    # Fallback
    "unknown"
]

# OCR Configuration
OCR_MODE = os.getenv("OCR_MODE", "ocr+llm")
DOC_CATEGORY = os.getenv("DOC_CATEGORY", "")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Demo Mode Configuration
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower().strip() in ("true", "1", "yes", "on")


def is_demo_mode() -> bool:
    """Check if demo mode is enabled."""
    return DEMO_MODE


# Database Configuration (for custom agents)
DB_HOST = os.getenv("DB_HOST", "")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "aiagentdb")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Agents@1252")
DB_NAME = os.getenv("DB_NAME", "stage_newskinny")
