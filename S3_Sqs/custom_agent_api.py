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

from S3_Sqs.custom_agent_service import CustomAgentService
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

load_dotenv()

# Create router
router = APIRouter(prefix="/api", tags=["Custom Agents"])


# ==================== Pydantic Models ====================

class CreateAgentRequest(BaseModel):
    """Request model for creating a new custom agent."""
    agent_name: str = Field(
        ...,
        description="Unique identifier for the agent (lowercase, numbers, underscores only)",
        example="senior_citizen_check",
        min_length=3,
        max_length=100
    )
    display_name: str = Field(
        ...,
        description="Human-readable name for the agent",
        example="Senior Citizen Document Validator"
    )
    description: str = Field(
        ...,
        description="Natural language validation rules (validation prompt)",
        example="Pass the document only if: 1) User age is above 54 years, 2) Document is not expired, 3) ID number has exactly 10 digits"
    )
    OCR: bool = Field(
        default=True,
        description="OCR mode: true = OCR+LLM (Textract + GPT), false = LLM only (Vision API)",
        example=True
    )
    tamper: bool = Field(
        default=False,
        description="Enable tampering detection: true = run tampering check, false = skip tampering check",
        example=False
    )
    creator_id: Optional[str] = Field(
        None,
        description="User ID of the creator",
        example="user_123"
    )


class CreateAgentResponse(BaseModel):
    """Response after creating a new agent."""
    success: bool
    agent_id: int
    agent_name: str
    endpoint: str
    OCR: bool = Field(..., description="OCR mode: true = OCR+LLM, false = LLM only")
    tamper: bool = Field(..., description="Tampering detection enabled")
    message: str
    user_description: Optional[str] = Field(None, description="Original user description/prompt")
    extracted_knowledge: Optional[str] = Field(None, description="Knowledge extracted from reference image (if provided)")
    final_description: Optional[str] = Field(None, description="Final enhanced description stored for the agent")


class UpdateAgentRequest(BaseModel):
    """Request model for updating an agent."""
    display_name: Optional[str] = None
    description: Optional[str] = Field(None, description="Validation rules/prompt")
    OCR: Optional[bool] = Field(None, description="OCR mode: true = OCR+LLM, false = LLM only")
    tamper: Optional[bool] = Field(None, description="Enable tampering detection")
    is_active: Optional[bool] = None


class AgentResponse(BaseModel):
    """Response model for agent details."""
    id: int
    agent_name: str
    display_name: Optional[str]
    description: str = Field(..., description="Validation rules/prompt")
    endpoint: str
    OCR: bool = Field(..., description="OCR mode: true = OCR+LLM, false = LLM only")
    tamper: bool = Field(..., description="Tampering detection enabled")
    creator_id: Optional[str]
    is_active: bool
    total_hits: int
    created_at: str
    updated_at: str


class ValidationResponse(BaseModel):
    """Response from document validation."""
    success: bool
    status: str  # pass, fail, error
    score: int
    reason: Any  # Can be List[str] or Dict with pass_conditions, fail_conditions, etc.
    file_name: str
    doc_extracted_json: Dict[str, Any]
    document_type: Optional[str]
    processing_time_ms: int
    agent_name: str
    tampering_score: Optional[int] = Field(None, description="Tampering risk score (0-100)")
    tampering_status: Optional[str] = Field(None, description="Tampering detection status: 'pass' or 'fail'")
    tampering_details: Optional[Dict[str, Any]] = Field(None, description="Tampering detection details")


# ==================== Database Connection ====================

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


def get_client_ip(request: Request) -> str:
    """Extract client IP from request, handling proxies."""
    # Check X-Forwarded-For header (for proxies/load balancers)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    # Check X-Real-IP header (nginx)
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Direct connection
    return request.client.host if request.client else "unknown"


# ==================== Agent Management Endpoints ====================

# Import knowledge extraction function
try:
    from Nodes.nodes.generic_extraction import extract_knowledge_from_reference_image
except ImportError:
    # Fallback if import path differs
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from Nodes.nodes.generic_extraction import extract_knowledge_from_reference_image
    except ImportError:
        extract_knowledge_from_reference_image = None


@router.post("/agents/create", response_model=CreateAgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    request: Request,
    agent_name: str = Form(..., description="Unique agent identifier (lowercase, underscores)"),
    display_name: str = Form(..., description="Human-readable agent name"),
    description: str = Form(..., description="Validation rules in natural language"),
    OCR: bool = Form(default=True, description="OCR mode: true = OCR+LLM (Textract + GPT), false = LLM only"),
    tamper: bool = Form(default=False, description="Enable tampering detection (default: false)"),
    creator_id: Optional[str] = Form(None, description="Creator user ID"),
    reference_image: Optional[UploadFile] = File(None, description="Optional: Reference/training image to learn from")
):
    """
    Create a new custom validation agent with OPTIONAL reference image training.
    
    **How it works:**
    1. If reference_image is provided:
       - Analyzes the image to extract knowledge (document type, fields, formats)
       - Merges extracted knowledge into your description
       - Stores the ENHANCED description (not the image)
       - Returns both user_description and extracted_knowledge in response
    2. If no reference_image:
       - Works with just the user description
       - Returns user_description in response (extracted_knowledge will be null)
    
    **Form Data:**
    - agent_name: Unique identifier (lowercase, underscores allowed)
    - display_name: Human-readable name  
    - description: Your validation rules in natural language
    - OCR: true = OCR+LLM (Textract + GPT), false = LLM only (Vision API)
    - tamper: Enable tampering detection (default: false)
    - creator_id: (Optional) Your user ID
    - reference_image: (Optional) Example image of a VALID document
    
    **Example without reference image:**
    ```
    agent_name: pan_validator
    description: Pass if PAN is 10 chars, name readable, DOB present
    OCR: true
    tamper: false
    ```
    
    **Example with reference image:**
    ```
    agent_name: pan_validator
    description: Pass if all fields are present and valid
    OCR: true
    tamper: true
    reference_image: [upload valid_pan_card.png]
    ```
    The system will analyze the image and learn:
    - PAN format: AAAAA9999A
    - Required fields: PAN, Name, Father's Name, DOB
    - Date format: DD/MM/YYYY
    
    This knowledge is automatically added to your description!
    
    **Response includes:**
    - user_description: Your original description
    - extracted_knowledge: Knowledge learned from reference image (if provided)
    - final_description: The complete description stored for the agent
    """
    import re
    
    connection = None
    temp_file_path = None
    
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
        extracted_knowledge = None
        knowledge_extracted = False
        document_type_learned = None
        
        # If reference image provided, extract knowledge
        if reference_image and extract_knowledge_from_reference_image:
            # Validate file type
            allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp", "application/pdf"]
            if reference_image.content_type not in allowed_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type '{reference_image.content_type}'. Allowed: JPEG, PNG, WebP, PDF"
                )
            
            # Save to temp file
            suffix = Path(reference_image.filename).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await reference_image.read()
                tmp.write(content)
                temp_file_path = tmp.name
            
            print(f"[Training] Extracting knowledge from reference image: {reference_image.filename}")
            
            # Extract knowledge from reference image
            knowledge_result = extract_knowledge_from_reference_image(
                image_path_or_url=temp_file_path,
                user_prompt=description
            )
            
            if knowledge_result.get("success"):
                final_description = knowledge_result["enhanced_prompt"]
                extracted_knowledge = knowledge_result["extracted_knowledge"]
                knowledge_extracted = True
                document_type_learned = knowledge_result.get("document_type", "Unknown")
                print(f"[Training] Successfully learned from reference: {document_type_learned}")
                print(f"[Training] Enhanced description length: {len(final_description)} chars")
            else:
                print(f"[Training] Warning: Could not extract knowledge: {knowledge_result.get('error')}")
                # Continue with original description
        elif reference_image and not extract_knowledge_from_reference_image:
            print("[Training] Warning: Knowledge extraction not available, using original description")
        
        # Create agent with (possibly enhanced) description
        connection = get_database_connection()
        service = CustomAgentService(connection)
        
        result = service.create_agent(
            agent_name=agent_name,
            display_name=display_name,
            prompt=final_description,
            mode=mode,
            tamper_check=tamper,
            creator_id=creator_id
        )
        
        # Build response message
        if knowledge_extracted:
            message = f"Agent '{agent_name}' created with knowledge learned from reference image ({document_type_learned}). Enhanced description stored."
        else:
            message = f"Agent '{agent_name}' created successfully. Use the endpoint to validate documents."
        
        return CreateAgentResponse(
            success=True,
            agent_id=result["agent_id"],
            agent_name=result["agent_name"],
            endpoint=result["endpoint"],
            OCR=OCR,
            tamper=tamper,
            message=message,
            user_description=user_description,
            extracted_knowledge=extracted_knowledge,
            final_description=final_description
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")
    finally:
        # Cleanup temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        if connection and connection.is_connected():
            connection.close()


@router.get("/agents/{agent_name}")
async def get_agent(agent_name: str):
    """Get details of a specific agent."""
    connection = None
    try:
        connection = get_database_connection()
        service = CustomAgentService(connection)
        
        agent = service.get_agent(agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        return {
            "success": True,
            "agent": agent
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if connection and connection.is_connected():
            connection.close()


@router.put("/agents/{agent_name}")
async def update_agent(agent_name: str, request: UpdateAgentRequest):
    """
    Update an existing agent.
    
    **Request Body:**
    - display_name: Human-readable name (optional)
    - description: Validation rules/prompt (optional)
    - OCR: true = OCR+LLM, false = LLM only (optional)
    - tamper: Enable tampering detection (optional)
    - is_active: Enable/disable agent (optional)
    """
    connection = None
    try:
        connection = get_database_connection()
        service = CustomAgentService(connection)
        
        # Check if agent exists
        agent = service.get_agent(agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        # Convert OCR bool to mode string if provided
        mode = None
        if request.OCR is not None:
            mode = "ocr+llm" if request.OCR else "llm"
        
        updated = service.update_agent(
            agent_name=agent_name,
            display_name=request.display_name,
            prompt=request.description,
            mode=mode,
            tamper_check=request.tamper,
            is_active=request.is_active
        )
        
        if not updated:
            raise HTTPException(status_code=400, detail="No changes provided")
        
        # Get updated agent
        agent = service.get_agent(agent_name)
        
        return {
            "success": True,
            "message": f"Agent '{agent_name}' updated successfully",
            "agent": agent
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if connection and connection.is_connected():
            connection.close()


@router.delete("/agents/{agent_name}")
async def delete_agent(agent_name: str):
    """Soft delete (deactivate) an agent."""
    connection = None
    try:
        connection = get_database_connection()
        service = CustomAgentService(connection)
        
        deleted = service.delete_agent(agent_name)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        return {
            "success": True,
            "message": f"Agent '{agent_name}' has been deactivated"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if connection and connection.is_connected():
            connection.close()


@router.get("/agents")
async def list_agents(
    creator_id: Optional[str] = None,
    is_active: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0
):
    """List all agents with optional filtering."""
    connection = None
    try:
        connection = get_database_connection()
        service = CustomAgentService(connection)
        
        agents = service.list_agents(
            creator_id=creator_id,
            is_active=is_active,
            limit=limit,
            offset=offset
        )
        
        return {
            "success": True,
            "total": len(agents),
            "agents": agents
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if connection and connection.is_connected():
            connection.close()


# ==================== Validation Endpoint ====================

@router.post("/agent/{agent_name}/validate", response_model=ValidationResponse)
async def validate_document(
    agent_name: str,
    request: Request,
    file: UploadFile = File(..., description="Document file to validate (PDF, JPG, PNG)"),
    user_id: Optional[str] = Form(None, description="User ID who is using this API")
):
    """
    Validate a document using a custom agent's rules.
    
    This endpoint processes the document through:
    1. OCR (text extraction)
    2. Classification (document type detection)
    3. Extraction (structured field extraction)
    4. Custom Validation (user's rules)
    
    **Parameters:**
    - **agent_name**: The agent to use for validation
    - **file**: Document file (PDF, JPG, JPEG, PNG)
    - **user_id**: Optional user identifier for tracking
    
    **Returns:**
    - **status**: "pass", "fail", or "error"
    - **score**: 0-100 validation score
    - **reason**: List of explanations
    - **doc_extracted_json**: All extracted fields
    """
    start_time = time.time()
    connection = None
    temp_file = None
    
    try:
        connection = get_database_connection()
        service = CustomAgentService(connection)
        
        # Get agent configuration
        agent = service.get_agent(agent_name)
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{agent_name}' not found or is inactive"
            )
        
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        allowed_extensions = ['.pdf', '.jpg', '.jpeg', '.png']
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Save uploaded file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        
        # Get tamper_check setting (default to False if not set)
        tamper_check = agent.get('tamper_check', False)
        
        print(f"[API] Processing file: {file.filename}")
        print(f"[API] Agent: {agent_name}")
        print(f"[API] Mode: {agent['mode']}")
        print(f"[API] Tamper Check: {tamper_check}")
        
        # Run custom validation pipeline
        from Nodes.nodes.custom_validation_node import run_custom_validation_pipeline
        
        result = run_custom_validation_pipeline(
            file_path=temp_file.name,
            user_prompt=agent['prompt'],
            mode=agent['mode'],
            tamper_check=tamper_check
        )
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Build result JSON for database
        db_result = {
            "status": result.get("status", "error"),
            "score": result.get("score", 0),
            "reason": result.get("reason", []),
            "file_name": file.filename,
            "doc_extracted_json": result.get("doc_extracted_json", {}),
            "document_type": result.get("document_type"),
            "checks": result.get("checks", [])
        }
        
        # Record result in database
        client_ip = get_client_ip(request)
        service.record_result(
            agent_id=agent['id'],
            agent_name=agent_name,
            api_endpoint=f"/api/agent/{agent_name}/validate",
            user_id=user_id,
            result=db_result,
            request_ip=client_ip,
            processing_time_ms=processing_time_ms
        )
        
        return ValidationResponse(
            success=True,
            status=result.get("status", "error"),
            score=result.get("score", 0),
            reason=result.get("reason", []),
            file_name=file.filename,
            doc_extracted_json=result.get("doc_extracted_json", {}),
            document_type=result.get("document_type"),
            processing_time_ms=processing_time_ms,
            agent_name=agent_name,
            tampering_score=result.get("tampering_score"),
            tampering_status=result.get("tampering_status"),
            tampering_details=result.get("tampering_details")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Try to record error
        if connection and 'agent' in locals() and agent:
            try:
                error_result = {
                    "status": "error",
                    "score": 0,
                    "reason": [str(e)],
                    "file_name": file.filename if file else "unknown",
                    "doc_extracted_json": {}
                }
                service.record_result(
                    agent_id=agent['id'],
                    agent_name=agent_name,
                    api_endpoint=f"/api/agent/{agent_name}/validate",
                    user_id=user_id,
                    result=error_result,
                    request_ip=get_client_ip(request),
                    processing_time_ms=processing_time_ms
                )
            except:
                pass
        
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")
        
    finally:
        # Cleanup temp file
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except:
                pass
        if connection and connection.is_connected():
            connection.close()


# ==================== Analytics Endpoints ====================

@router.get("/creator/{creator_id}/agents")
async def get_agents_by_creator(
    creator_id: str,
    is_active: Optional[bool] = None
):
    """
    Get all agents created by a specific user.
    
    **Parameters:**
    - **creator_id**: User ID of the creator
    - **is_active**: Optional filter by active status
    """
    connection = None
    try:
        connection = get_database_connection()
        service = CustomAgentService(connection)
        
        agents = service.get_agents_by_creator(creator_id, is_active)
        
        return {
            "success": True,
            "creator_id": creator_id,
            "total_agents": len(agents),
            "agents": agents
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if connection and connection.is_connected():
            connection.close()


@router.get("/agent/{agent_name}/users")
async def get_agent_users(
    agent_name: str,
    limit: int = 100,
    offset: int = 0
):
    """
    Get all users who have used a specific agent.
    
    Returns user statistics including:
    - Total requests per user
    - Pass/fail counts
    - First and last usage timestamps
    """
    connection = None
    try:
        connection = get_database_connection()
        service = CustomAgentService(connection)
        
        # Check if agent exists
        agent = service.get_agent(agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        result = service.get_agent_users(agent_name, limit, offset)
        
        return {
            "success": True,
            "agent_name": agent_name,
            "total_unique_users": result["total_unique_users"],
            "users": result["users"],
            "pagination": {
                "limit": limit,
                "offset": offset,
                "has_more": result["has_more"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if connection and connection.is_connected():
            connection.close()


@router.get("/agent/{agent_name}/stats")
async def get_agent_stats(agent_name: str):
    """
    Get detailed usage statistics for an agent.
    
    Returns:
    - Total hits, pass/fail/error counts
    - Success rate
    - Average processing time
    - Unique users count
    - Today/week/month breakdowns
    """
    connection = None
    try:
        connection = get_database_connection()
        service = CustomAgentService(connection)
        
        # Check if agent exists
        agent = service.get_agent(agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        stats = service.get_agent_stats(agent_name)
        
        return {
            "success": True,
            "agent_name": agent_name,
            "stats": stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if connection and connection.is_connected():
            connection.close()


@router.get("/creator/{creator_id}/stats")
async def get_creator_stats(creator_id: str):
    """
    Get aggregated stats for all agents created by a user.
    
    Returns:
    - Summary: total agents, active/inactive counts, total hits
    - Per-agent breakdown
    - Recent activity (today/week/month)
    """
    connection = None
    try:
        connection = get_database_connection()
        service = CustomAgentService(connection)
        
        stats = service.get_creator_stats(creator_id)
        
        if not stats:
            raise HTTPException(
                status_code=404,
                detail=f"No agents found for creator '{creator_id}'"
            )
        
        return {
            "success": True,
            "creator_id": creator_id,
            **stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if connection and connection.is_connected():
            connection.close()


@router.get("/agent/{agent_name}/count")
async def get_agent_hit_count(agent_name: str):
    """
    Simple endpoint to get just the hit count for an agent.
    """
    connection = None
    try:
        connection = get_database_connection()
        service = CustomAgentService(connection)
        
        count = service.get_agent_hit_count(agent_name)
        
        if count is None:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        return {
            "agent_name": agent_name,
            "total_hits": count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if connection and connection.is_connected():
            connection.close()

