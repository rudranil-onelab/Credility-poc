"""
Custom Agent API - Dynamic API Creator for Document Validation.

This module provides endpoints for:
- Creating custom validation agents with user-defined prompts
- Validating documents against custom rules
- Analytics and usage tracking
"""

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form, status
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import sys
import time
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from S3_Sqs.custom_agent_serviceV2 import CustomAgentService
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv


router = APIRouter(prefix="/api", tags=["Custom Agents"])

def get_database_connection():
    """Get database connection."""
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST", ""),
            port=int(os.getenv("DB_PORT", "3306")),
            user=os.getenv("DB_USER", "aiagentdb"),
            password=os.getenv("DB_PASSWORD", "Agents@1252"),
            database=os.getenv("DB_NAME", "stage_newskinny")
        )
        if connection.is_connected():
            return connection
    except Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection error: {str(e)}"
        )


class CreateAgentResponsev2(BaseModel):
    """Response after creating a new agent."""
    success: bool
    agent_id: int
    agent_name: str
    endpoint: str
    OCR: bool = Field(..., description="OCR mode: true = OCR+LLM, false = LLM only")
    tamper: bool = Field(..., description="Tampering detection enabled")
    message: str
    user_description: Optional[str] = Field(None, description="Original user description/prompt")
    
@router.post("/agents/v2/create", response_model=CreateAgentResponsev2, status_code=status.HTTP_201_CREATED)
async def create_agent(
    request: Request,
    agent_name: str = Form(..., description="Unique agent identifier (lowercase, underscores)"),
    display_name: str = Form(..., description="Human-readable agent name"),
    description: str = Form(..., description="Validation rules in natural language (REQUIRED - centralized description for all images)"),
    creator_id: Optional[str] = Form(None, description="Creator user ID"),
    OCR: Optional[bool] = Form(None, description="OCR mode: true = OCR+LLM, false = LLM only"),
    tamper: Optional[bool] = Form(None, description="Enable tampering detection"),

):
    """Create a new custom validation agent.

How it works:
1. Uses ONLY the provided textual description to define validation behavior.
2. Stores the finalized agent configuration in the database.
3. No reference images, image analysis, or image-based knowledge extraction is performed.
4. The agent relies strictly on OCR output and/or LLM reasoning based on configuration.

Form Data:
- agent_name: (REQUIRED)
  Unique identifier for the agent.
  • lowercase letters
  • underscores allowed
  • must be unique

- display_name: (OPTIONAL)
  Human-readable agent name.

- description: (REQUIRED)
  Centralized natural-language validation rules.
  This is the ONLY source of logic for:
  • field validation
  • format checks
  • pass/fail rules
  • conditional requirements

- OCR: (REQUIRED)
  true  → OCR + LLM (Textract + LLM)
  false → LLM only (Vision / text-based reasoning)

- tamper: (OPTIONAL, default: false)
  Enables tampering detection such as:
  • layout inconsistencies
  • missing structural elements
  • unusual spacing or alignment
  • inconsistent fonts or formatting
  • unexpected color or pattern anomalies

- creator_id: (OPTIONAL)
  Identifier of the user creating the agent.

Example request:
agent_name: pan_validator
display_name: PAN Card Validator
description: Pass if PAN number is exactly 10 characters, name is readable, DOB is present, and document is not tampered.
OCR: true
tamper: true

Response includes:
- agent_name
- display_name
- user_description
- final_description (stored validation logic)
- OCR
- tamper
- creator_id
"""
    import re
    
    connection = None
    temp_file_paths = []
    uploaded_s3_urls = []
    
    # Convert OCR bool to mode string for internal processing
    mode = "ocr+llm" if OCR else "llm"
    
    try:
        # Validate agent_name format
        if not re.match(r'^[a-z0-9_]+$', agent_name):
            raise HTTPException(
                status_code=400,
                detail="agent_name must contain only lowercase letters, numbers, and underscores"
            )
        
        if len(agent_name) < 3:
            raise HTTPException(status_code=400, detail="agent_name must be at least 3 characters")
        
        if len(agent_name) > 100:
            raise HTTPException(status_code=400, detail="agent_name must be at most 100 characters")
        

        
        user_description = description  # Store original description
        final_description = description
        connection = get_database_connection()
        service = CustomAgentService(connection)
        result = service.create_agent(
            agent_name=agent_name,
            display_name=display_name,
            prompt=final_description,
            mode=mode,
            tamper_check=tamper,
            creator_id=creator_id,
        )
        
        message = f"Agent '{agent_name}' created successfully. Use the endpoint to validate documents."
        return CreateAgentResponsev2(
            success=True,
            agent_id=result["agent_id"],
            agent_name=result["agent_name"],
            endpoint=result["endpoint"],
            OCR=OCR,
            tamper=tamper,
            message=message,
            user_description=user_description,
        )
        
    except ValueError as e:
        # Don't delete S3 images on error - they may be needed for debugging
        # or the user may want to retry with the same images
        print(f"[API] Agent creation failed (ValueError): {str(e)}")
        if uploaded_s3_urls:
            print(f"[API] Reference images preserved in S3: {uploaded_s3_urls}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # Don't delete S3 images on HTTP errors
        print(f"[API] Agent creation failed (HTTPException)")
        if uploaded_s3_urls:
            print(f"[API] Reference images preserved in S3: {uploaded_s3_urls}")
        raise
    except Exception as e:
        # Don't delete S3 images on unexpected errors
        print(f"[API] Agent creation failed (Exception): {str(e)}")
        if uploaded_s3_urls:
            print(f"[API] Reference images preserved in S3: {uploaded_s3_urls}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")
    finally:
        # Cleanup temp files
        for temp_path in temp_file_paths:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        if connection and connection.is_connected():
            connection.close()
