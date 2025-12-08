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

# S3 Configuration for reference images
S3_BUCKET = os.getenv("S3_BUCKET", "lendingwise-poc")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
REFERENCE_IMAGES_PREFIX = "custom_agents/reference_images"
MAX_REFERENCE_IMAGES = 5


def get_s3_client():
    """Create S3 client with credentials from environment."""
    import boto3
    kwargs = {"region_name": AWS_REGION}
    
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    if aws_access_key_id and aws_secret_access_key:
        kwargs["aws_access_key_id"] = aws_access_key_id
        kwargs["aws_secret_access_key"] = aws_secret_access_key
    
    return boto3.client("s3", **kwargs)


def upload_reference_image_to_s3(
    file_content: bytes,
    filename: str,
    agent_name: str,
    image_index: int
) -> str:
    """
    Upload a reference image to S3 and return the S3 URL.
    
    Args:
        file_content: Raw bytes of the image
        filename: Original filename
        agent_name: Name of the agent (used in S3 path)
        image_index: Index of the image (1-5)
    
    Returns:
        S3 URL of the uploaded image
    """
    import datetime
    import re
    
    s3_client = get_s3_client()
    
    # Sanitize filename
    safe_filename = re.sub(r"[^A-Za-z0-9._()-]", "_", filename)
    
    # Get file extension
    ext = os.path.splitext(safe_filename)[1].lower() or ".png"
    
    # Generate unique S3 key with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    s3_key = f"{REFERENCE_IMAGES_PREFIX}/{agent_name}/ref_image_{image_index}_{timestamp}{ext}"
    
    # Determine content type
    content_types = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    content_type = content_types.get(ext, "application/octet-stream")
    
    # Upload to S3
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=file_content,
        ContentType=content_type
    )
    
    s3_url = f"s3://{S3_BUCKET}/{s3_key}"
    print(f"[S3] Uploaded reference image: {s3_url}")
    
    return s3_url


def delete_reference_images_from_s3(agent_name: str) -> int:
    """
    Delete all reference images for an agent from S3.
    
    Args:
        agent_name: Name of the agent
    
    Returns:
        Number of images deleted
    """
    s3_client = get_s3_client()
    prefix = f"{REFERENCE_IMAGES_PREFIX}/{agent_name}/"
    
    try:
        # List objects with prefix
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        
        if 'Contents' not in response:
            return 0
        
        # Delete each object
        deleted_count = 0
        for obj in response['Contents']:
            s3_client.delete_object(Bucket=S3_BUCKET, Key=obj['Key'])
            print(f"[S3] Deleted: s3://{S3_BUCKET}/{obj['Key']}")
            deleted_count += 1
        
        return deleted_count
    except Exception as e:
        print(f"[S3] Error deleting reference images: {e}")
        return 0


# Prefix for storing input documents that are validated
INPUT_FILES_PREFIX = "custom_agents/input_files"


def upload_input_file_to_s3(
    file_content: bytes,
    filename: str,
    agent_name: str
) -> str:
    """
    Upload an input document to S3 for logging/audit purposes.
    
    Args:
        file_content: Raw bytes of the file
        filename: Original filename
        agent_name: Name of the agent processing this file
    
    Returns:
        S3 URL of the uploaded file
    """
    import datetime
    import re
    import uuid
    
    s3_client = get_s3_client()
    
    # Sanitize filename
    safe_filename = re.sub(r"[^A-Za-z0-9._()-]", "_", filename)
    
    # Get file extension
    ext = os.path.splitext(safe_filename)[1].lower() or ".bin"
    
    # Generate unique S3 key with timestamp and UUID for uniqueness
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    s3_key = f"{INPUT_FILES_PREFIX}/{agent_name}/{timestamp}_{unique_id}{ext}"
    
    # Determine content type
    content_types = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    content_type = content_types.get(ext, "application/octet-stream")
    
    # Upload to S3
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=file_content,
        ContentType=content_type
    )
    
    s3_url = f"s3://{S3_BUCKET}/{s3_key}"
    return s3_url


def generate_presigned_url_from_s3_uri(s3_uri: str, expiration: int = 3600) -> str:
    """
    Generate a presigned HTTPS URL from an S3 URI.
    
    Args:
        s3_uri: S3 URI in format s3://bucket/key
        expiration: URL expiration time in seconds (default 1 hour)
    
    Returns:
        Presigned HTTPS URL or original URI if conversion fails
    """
    if not s3_uri or not s3_uri.startswith("s3://"):
        return s3_uri
    
    try:
        # Parse s3://bucket/key format
        s3_path = s3_uri[5:]  # Remove 's3://'
        parts = s3_path.split("/", 1)
        if len(parts) != 2:
            return s3_uri
        
        bucket = parts[0]
        key = parts[1]
        
        s3_client = get_s3_client()
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=expiration
        )
        return presigned_url
    except Exception as e:
        print(f"[S3] Error generating presigned URL: {e}")
        return s3_uri


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


class ContradictionResolution(BaseModel):
    """Details about a contradiction found and resolved during multi-image training."""
    field_or_rule: str = Field(..., description="The field or rule that had conflicting information")
    image_1_value: Optional[str] = Field(None, description="Value from first image")
    image_2_value: Optional[str] = Field(None, description="Value from second image")
    resolution: str = Field(..., description="How the contradiction was resolved")
    confidence: str = Field(default="medium", description="Confidence in resolution: high/medium/low")


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
    extracted_knowledge: Optional[str] = Field(None, description="Knowledge extracted from reference images (consolidated if multiple)")
    final_description: Optional[str] = Field(None, description="Final enhanced description stored for the agent")
    reference_images: Optional[List[str]] = Field(None, description="S3 URLs of stored reference images (up to 5)")
    images_processed: Optional[int] = Field(None, description="Number of reference images successfully processed")
    unique_fields_found: Optional[List[str]] = Field(None, description="List of unique fields identified across all reference images")
    contradictions_resolved: Optional[List[ContradictionResolution]] = Field(None, description="Contradictions found and resolved during multi-image training")


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
    reference_images: Optional[List[str]] = Field(None, description="S3 URLs of stored reference images")


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
    # OCR extraction quality fields
    ocr_extraction_status: Optional[str] = Field(None, description="OCR extraction quality status: 'pass' or 'fail'. Pass if confidence >= 70%, fail otherwise.")
    ocr_extraction_confidence: Optional[float] = Field(None, description="Average OCR confidence score from AWS Textract (0-100)")
    ocr_extraction_reason: Optional[str] = Field(None, description="Human-readable reason explaining the OCR extraction quality")
    # Reference images
    reference_images: Optional[List[str]] = Field(None, description="S3 URLs of reference images used by this agent")


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

# Import knowledge extraction functions
try:
    from Nodes.nodes.generic_extraction import extract_knowledge_from_reference_image, merge_knowledge_from_multiple_images
except ImportError:
    # Fallback if import path differs
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from Nodes.nodes.generic_extraction import extract_knowledge_from_reference_image, merge_knowledge_from_multiple_images
    except ImportError:
        extract_knowledge_from_reference_image = None
        merge_knowledge_from_multiple_images = None


@router.post("/agents/create", response_model=CreateAgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    request: Request,
    agent_name: str = Form(..., description="Unique agent identifier (lowercase, underscores)"),
    display_name: str = Form(..., description="Human-readable agent name"),
    description: str = Form(..., description="Validation rules in natural language"),
    OCR: bool = Form(default=True, description="OCR mode: true = OCR+LLM (Textract + GPT), false = LLM only"),
    tamper: bool = Form(default=False, description="Enable tampering detection (default: false)"),
    creator_id: Optional[str] = Form(None, description="Creator user ID"),
    reference_images: List[UploadFile] = File(default=[], description="Optional: Up to 5 reference/training images to learn from and store")
):
    """
    Create a new custom validation agent with OPTIONAL reference images (up to 5).
    
    **How it works:**
    1. If reference_images are provided (up to 5):
       - Stores ALL images in S3 at: s3://{bucket}/custom_agents/reference_images/{agent_name}/
       - Analyzes the FIRST image to extract knowledge (document type, fields, formats)
       - Merges extracted knowledge into your description
       - Stores the ENHANCED description + image S3 URLs in database
       - Returns reference_images array with all S3 URLs
    2. If no reference_images:
       - Works with just the user description
       - Returns empty reference_images array
    
    **Form Data:**
    - agent_name: Unique identifier (lowercase, underscores allowed)
    - display_name: Human-readable name  
    - description: Your validation rules in natural language
    - OCR: true = OCR+LLM (Textract + GPT), false = LLM only (Vision API)
    - tamper: Enable tampering detection (default: false)
    - creator_id: (Optional) Your user ID
    - reference_images: (Optional) Up to 5 example images of VALID documents
    
    **Example without reference images:**
    ```
    agent_name: pan_validator
    description: Pass if PAN is 10 chars, name readable, DOB present
    OCR: true
    tamper: false
    ```
    
    **Example with reference images:**
    ```
    agent_name: pan_validator
    description: Pass if all fields are present and valid
    OCR: true
    tamper: true
    reference_images: [upload valid_pan_card.png, upload sample_pan.jpg, ...]
    ```
    The system will:
    - Store all images in S3
    - Analyze the first image and learn:
      - PAN format: AAAAA9999A
      - Required fields: PAN, Name, Father's Name, DOB
      - Date format: DD/MM/YYYY
    
    **Response includes:**
    - user_description: Your original description
    - extracted_knowledge: Knowledge learned from first reference image (if provided)
    - final_description: The complete description stored for the agent
    - reference_images: Array of S3 URLs where images are stored
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
        
        # Validate number of reference images
        if len(reference_images) > MAX_REFERENCE_IMAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {MAX_REFERENCE_IMAGES} reference images allowed. You provided {len(reference_images)}."
            )
        
        user_description = description  # Store original description
        final_description = description
        extracted_knowledge = None
        knowledge_extracted = False
        document_type_learned = None
        unique_fields_found = []
        contradictions_resolved = []
        images_processed_count = 0
        
        # Process reference images if provided
        if reference_images and len(reference_images) > 0:
            allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp", "application/pdf"]
            
            # First pass: validate and upload all images to S3, save to temp files for knowledge extraction
            for idx, ref_image in enumerate(reference_images, start=1):
                # Skip empty files
                if not ref_image.filename:
                    continue
                
                # Validate file type
                if ref_image.content_type not in allowed_types:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid file type '{ref_image.content_type}' for image {idx}. Allowed: JPEG, PNG, WebP, PDF"
                    )
                
                # Read file content
                content = await ref_image.read()
                
                # Upload to S3
                s3_url = upload_reference_image_to_s3(
                    file_content=content,
                    filename=ref_image.filename,
                    agent_name=agent_name,
                    image_index=idx
                )
                uploaded_s3_urls.append(s3_url)
                
                # Save ALL images to temp files for knowledge extraction
                suffix = Path(ref_image.filename).suffix or ".png"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(content)
                    temp_file_paths.append(tmp.name)
            
            print(f"[S3] Uploaded {len(uploaded_s3_urls)} reference images for agent '{agent_name}'")
            
            # Second pass: Extract and merge knowledge from ALL images
            if temp_file_paths and (extract_knowledge_from_reference_image or merge_knowledge_from_multiple_images):
                if len(temp_file_paths) == 1:
                    # Single image - use simple extraction
                    print(f"[Training] Extracting knowledge from single reference image")
                    
                    if extract_knowledge_from_reference_image:
                        knowledge_result = extract_knowledge_from_reference_image(
                            image_path_or_url=temp_file_paths[0],
                            user_prompt=description
                        )
                        
                        if knowledge_result.get("success"):
                            final_description = knowledge_result["enhanced_prompt"]
                            extracted_knowledge = knowledge_result["extracted_knowledge"]
                            knowledge_extracted = True
                            document_type_learned = knowledge_result.get("document_type", "Unknown")
                            unique_fields_found = [f.get("name", "") for f in knowledge_result.get("fields", [])]
                            images_processed_count = 1
                            print(f"[Training] Successfully learned from reference: {document_type_learned}")
                        else:
                            print(f"[Training] Warning: Could not extract knowledge: {knowledge_result.get('error')}")
                else:
                    # Multiple images - use merge function to consolidate knowledge
                    print(f"[Training] Extracting and merging knowledge from {len(temp_file_paths)} reference images")
                    
                    if merge_knowledge_from_multiple_images:
                        merge_result = merge_knowledge_from_multiple_images(
                            image_paths=temp_file_paths,
                            user_prompt=description
                        )
                        
                        if merge_result.get("success"):
                            final_description = merge_result["enhanced_prompt"]
                            extracted_knowledge = merge_result["extracted_knowledge"]
                            knowledge_extracted = True
                            document_type_learned = merge_result.get("document_type", "Unknown")
                            unique_fields_found = merge_result.get("unique_fields", [])
                            
                            # Count successfully processed images
                            per_image = merge_result.get("per_image_knowledge", [])
                            images_processed_count = sum(1 for img in per_image if img.get("success"))
                            
                            # Extract contradiction details for response
                            raw_contradictions = merge_result.get("contradictions_found", [])
                            for c in raw_contradictions:
                                contradictions_resolved.append(ContradictionResolution(
                                    field_or_rule=c.get("field_or_rule", "Unknown"),
                                    image_1_value=c.get("image_1_value"),
                                    image_2_value=c.get("image_2_value"),
                                    resolution=c.get("resolution", ""),
                                    confidence=c.get("confidence", "medium")
                                ))
                            
                            print(f"[Training] Successfully merged knowledge from {images_processed_count} images")
                            print(f"[Training] Document type: {document_type_learned}")
                            print(f"[Training] Unique fields found: {len(unique_fields_found)}")
                            print(f"[Training] Contradictions resolved: {len(contradictions_resolved)}")
                        else:
                            print(f"[Training] Warning: Could not merge knowledge: {merge_result.get('error')}")
                    else:
                        # Fallback to single image extraction if merge not available
                        print("[Training] Merge function not available, falling back to first image only")
                        if extract_knowledge_from_reference_image:
                            knowledge_result = extract_knowledge_from_reference_image(
                                image_path_or_url=temp_file_paths[0],
                                user_prompt=description
                            )
                            if knowledge_result.get("success"):
                                final_description = knowledge_result["enhanced_prompt"]
                                extracted_knowledge = knowledge_result["extracted_knowledge"]
                                knowledge_extracted = True
                                document_type_learned = knowledge_result.get("document_type", "Unknown")
                                images_processed_count = 1
        
        # Create agent with (possibly enhanced) description
        connection = get_database_connection()
        service = CustomAgentService(connection)
        
        result = service.create_agent(
            agent_name=agent_name,
            display_name=display_name,
            prompt=final_description,
            mode=mode,
            tamper_check=tamper,
            creator_id=creator_id,
            reference_images=uploaded_s3_urls if uploaded_s3_urls else None
        )
        
        # Build response message
        if knowledge_extracted and uploaded_s3_urls:
            if images_processed_count > 1:
                contradiction_note = f" {len(contradictions_resolved)} contradiction(s) resolved." if contradictions_resolved else ""
                message = f"Agent '{agent_name}' created with {len(uploaded_s3_urls)} reference image(s) stored in S3. Knowledge consolidated from {images_processed_count} images ({document_type_learned}).{contradiction_note}"
            else:
                message = f"Agent '{agent_name}' created with {len(uploaded_s3_urls)} reference image(s) stored in S3. Knowledge learned from reference image ({document_type_learned})."
        elif uploaded_s3_urls:
            message = f"Agent '{agent_name}' created with {len(uploaded_s3_urls)} reference image(s) stored in S3."
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
            final_description=final_description,
            reference_images=uploaded_s3_urls if uploaded_s3_urls else None,
            images_processed=images_processed_count if images_processed_count > 0 else None,
            unique_fields_found=unique_fields_found if unique_fields_found else None,
            contradictions_resolved=contradictions_resolved if contradictions_resolved else None
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
async def update_agent(
    agent_name: str,
    display_name: Optional[str] = Form(None, description="Human-readable agent name"),
    description: Optional[str] = Form(None, description="Validation rules in natural language (prompt)"),
    OCR: Optional[bool] = Form(None, description="OCR mode: true = OCR+LLM, false = LLM only"),
    tamper: Optional[bool] = Form(None, description="Enable tampering detection"),
    is_active: Optional[bool] = Form(None, description="Enable/disable agent"),
    creator_id: Optional[str] = Form(None, description="Update creator ID"),
    reference_images: List[UploadFile] = File(default=[], description="New reference images (replaces existing, up to 5)"),
    append_images: bool = Form(default=False, description="If true, append new images to existing. If false, replace all.")
):
    """
    Update an existing agent's configuration.
    
    **Form Data (all optional):**
    - **display_name**: Human-readable name
    - **description**: Validation rules/prompt (will re-extract knowledge if reference_images provided)
    - **OCR**: true = OCR+LLM (Textract + GPT), false = LLM only (Vision API)
    - **tamper**: Enable/disable tampering detection
    - **is_active**: Enable/disable agent
    - **creator_id**: Update creator ID
    - **reference_images**: New reference images (up to 5 files)
    - **append_images**: If true, add to existing images. If false (default), replace all images.
    
    **How reference_images update works:**
    1. If `reference_images` provided and `append_images=false`:
       - Old images are kept in S3 (not deleted)
       - New images are uploaded to S3
       - Knowledge is re-extracted from new images
       - Database is updated with new image URLs
    2. If `reference_images` provided and `append_images=true`:
       - Existing images are preserved
       - New images are added (total max 5)
       - Knowledge is re-extracted from ALL images (existing + new)
    3. If no `reference_images` provided:
       - Only other fields are updated
       - Existing images remain unchanged
    
    **Example - Update just OCR mode:**
    ```
    OCR: false
    ```
    
    **Example - Update prompt and add new reference images:**
    ```
    description: New validation rules here
    reference_images: [file1.png, file2.jpg]
    append_images: false
    ```
    """
    connection = None
    temp_file_paths = []
    uploaded_s3_urls = []
    
    try:
        connection = get_database_connection()
        service = CustomAgentService(connection)
        
        # Check if agent exists
        agent = service.get_agent(agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        # Track what's being updated
        updates_made = []
        
        # Convert OCR bool to mode string if provided
        mode = None
        if OCR is not None:
            mode = "ocr+llm" if OCR else "llm"
            updates_made.append(f"OCR mode → {'OCR+LLM' if OCR else 'LLM only'}")
        
        # Process reference images if provided
        new_reference_images = None
        extracted_knowledge = None
        final_description = description  # Use provided description or None
        knowledge_extracted = False
        
        if reference_images and len(reference_images) > 0 and reference_images[0].filename:
            # Filter out empty uploads
            valid_images = [img for img in reference_images if img.filename]
            
            if valid_images:
                # Get existing images if appending
                existing_images = []
                if append_images and agent.get("reference_images"):
                    existing_images = agent["reference_images"]
                
                # Check total count
                total_images = len(existing_images) + len(valid_images)
                if total_images > MAX_REFERENCE_IMAGES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Maximum {MAX_REFERENCE_IMAGES} reference images allowed. You have {len(existing_images)} existing + {len(valid_images)} new = {total_images} total."
                    )
                
                allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp", "application/pdf"]
                
                # Upload new images
                for idx, ref_image in enumerate(valid_images, start=len(existing_images) + 1):
                    if ref_image.content_type not in allowed_types:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid file type '{ref_image.content_type}' for image {idx}. Allowed: JPEG, PNG, WebP, PDF"
                        )
                    
                    content = await ref_image.read()
                    
                    # Upload to S3
                    s3_url = upload_reference_image_to_s3(
                        file_content=content,
                        filename=ref_image.filename,
                        agent_name=agent_name,
                        image_index=idx
                    )
                    uploaded_s3_urls.append(s3_url)
                    
                    # Save to temp file for knowledge extraction
                    suffix = Path(ref_image.filename).suffix or ".png"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(content)
                        temp_file_paths.append(tmp.name)
                
                print(f"[Update] Uploaded {len(uploaded_s3_urls)} new reference images for agent '{agent_name}'")
                
                # Combine with existing if appending
                if append_images:
                    new_reference_images = existing_images + uploaded_s3_urls
                    updates_made.append(f"Added {len(uploaded_s3_urls)} reference images (total: {len(new_reference_images)})")
                else:
                    new_reference_images = uploaded_s3_urls
                    updates_made.append(f"Replaced reference images with {len(uploaded_s3_urls)} new images")
                
                # Re-extract knowledge from images
                # Use the description to update, or fall back to existing prompt
                base_description = description if description else agent.get("prompt", "")
                
                if temp_file_paths and (extract_knowledge_from_reference_image or merge_knowledge_from_multiple_images):
                    if len(temp_file_paths) == 1 and not append_images:
                        # Single new image
                        if extract_knowledge_from_reference_image:
                            print(f"[Update] Extracting knowledge from new reference image")
                            knowledge_result = extract_knowledge_from_reference_image(
                                image_path_or_url=temp_file_paths[0],
                                user_prompt=base_description
                            )
                            if knowledge_result.get("success"):
                                final_description = knowledge_result["enhanced_prompt"]
                                extracted_knowledge = knowledge_result["extracted_knowledge"]
                                knowledge_extracted = True
                                print(f"[Update] Knowledge extracted: {knowledge_result.get('document_type', 'Unknown')}")
                    else:
                        # Multiple images - merge knowledge
                        if merge_knowledge_from_multiple_images:
                            print(f"[Update] Merging knowledge from {len(temp_file_paths)} reference images")
                            merge_result = merge_knowledge_from_multiple_images(
                                image_paths=temp_file_paths,
                                user_prompt=base_description
                            )
                            if merge_result.get("success"):
                                final_description = merge_result["enhanced_prompt"]
                                extracted_knowledge = merge_result["extracted_knowledge"]
                                knowledge_extracted = True
                                print(f"[Update] Knowledge merged from {len(temp_file_paths)} images")
        
        # Update the agent
        updated = service.update_agent(
            agent_name=agent_name,
            display_name=display_name,
            prompt=final_description,
            mode=mode,
            tamper_check=tamper,
            is_active=is_active,
            reference_images=new_reference_images
        )
        
        # Track other updates
        if display_name:
            updates_made.append(f"display_name → '{display_name}'")
        if final_description and final_description != agent.get("prompt"):
            if knowledge_extracted:
                updates_made.append("prompt updated with new knowledge from reference images")
            else:
                updates_made.append("prompt updated")
        if tamper is not None:
            updates_made.append(f"tamper_check → {tamper}")
        if is_active is not None:
            updates_made.append(f"is_active → {is_active}")
        
        if not updated and not updates_made:
            raise HTTPException(status_code=400, detail="No changes provided")
        
        # Get updated agent
        updated_agent = service.get_agent(agent_name)
        
        # Convert reference images to presigned URLs for response
        if updated_agent and updated_agent.get("reference_images"):
            updated_agent["reference_images"] = [
                generate_presigned_url_from_s3_uri(img) for img in updated_agent["reference_images"]
            ]
        
        return {
            "success": True,
            "message": f"Agent '{agent_name}' updated successfully",
            "updates": updates_made,
            "knowledge_extracted": knowledge_extracted,
            "agent": updated_agent
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
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
        
        # Upload input file to S3 for logging/audit
        input_file_s3_url = None
        try:
            input_file_s3_url = upload_input_file_to_s3(
                file_content=content,
                filename=file.filename,
                agent_name=agent_name
            )
            print(f"[API] Input file uploaded to S3: {input_file_s3_url}")
        except Exception as upload_err:
            print(f"[API] Warning: Could not upload input file to S3: {upload_err}")
        
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
        
        # Build comprehensive result JSON for database (includes all details for logs)
        db_result = {
            "status": result.get("status", "error"),
            "score": result.get("score", 0),
            "reason": result.get("reason", []),
            "file_name": file.filename,
            "file_input": input_file_s3_url,  # S3 URL of input document
            "doc_extracted_json": result.get("doc_extracted_json", {}),
            "document_type": result.get("document_type"),
            "checks": result.get("checks", []),
            # Tampering details
            "tampering_score": result.get("tampering_score"),
            "tampering_status": result.get("tampering_status"),
            "tampering_details": result.get("tampering_details"),
            # OCR extraction quality
            "ocr_extraction_status": result.get("ocr_extraction_status"),
            "ocr_extraction_confidence": result.get("ocr_extraction_confidence"),
            "ocr_extraction_reason": result.get("ocr_extraction_reason")
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
            tampering_details=result.get("tampering_details"),
            # OCR extraction quality fields
            ocr_extraction_status=result.get("ocr_extraction_status"),
            ocr_extraction_confidence=result.get("ocr_extraction_confidence"),
            ocr_extraction_reason=result.get("ocr_extraction_reason"),
            # Reference images from agent
            reference_images=agent.get("reference_images")
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
            },
            "reference_images": agent.get("reference_images")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if connection and connection.is_connected():
            connection.close()


@router.get("/agent/{agent_name}/stats")
async def get_agent_stats(
    agent_name: str,
    logs_limit: int = 100,
    logs_offset: int = 0
):
    """
    Get detailed usage statistics and validation logs for an agent.
    
    Returns:
    - **stats**: Total hits, pass/fail/error counts, success rate, avg processing time
    - **logs**: Complete validation history with:
      - user_id, status, score, reason
      - file_input (presigned S3 URL of input document)
      - doc_extracted_json, document_type
      - tampering_score, tampering_status, tampering_details
      - ocr_extraction_status, ocr_extraction_confidence, ocr_extraction_reason
      - processing_time_ms, created_at
    - **agent_info**: prompt, reference_images, mode, tamper_check
    
    **Query Parameters:**
    - logs_limit: Max logs to return (default 100)
    - logs_offset: Offset for pagination (default 0)
    """
    connection = None
    try:
        connection = get_database_connection()
        service = CustomAgentService(connection)
        
        # Check if agent exists
        agent = service.get_agent(agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        # Get stats
        stats = service.get_agent_stats(agent_name)
        
        # Get logs
        logs_result = service.get_agent_logs(agent_name, limit=logs_limit, offset=logs_offset)
        
        # Convert S3 URIs to presigned URLs for logs
        logs_with_urls = []
        for log in logs_result.get("logs", []):
            log_copy = log.copy()
            
            # Convert file_input S3 URI to presigned URL
            if log_copy.get("file_input"):
                log_copy["file_input"] = generate_presigned_url_from_s3_uri(log_copy["file_input"])
            
            logs_with_urls.append(log_copy)
        
        # Convert reference_images S3 URIs to presigned URLs
        reference_images_urls = None
        if agent.get("reference_images"):
            reference_images_urls = [
                generate_presigned_url_from_s3_uri(img) for img in agent["reference_images"]
            ]
        
        return {
            "success": True,
            "agent_name": agent_name,
            "stats": stats,
            "logs": logs_with_urls,
            "logs_pagination": logs_result.get("pagination"),
            "total_logs": logs_result.get("total_logs", 0),
            "agent_info": {
                "display_name": agent.get("display_name"),
                "prompt": agent.get("prompt"),
                "mode": agent.get("mode"),
                "tamper_check": agent.get("tamper_check"),
                "reference_images": reference_images_urls,
                "created_at": agent.get("created_at"),
                "is_active": agent.get("is_active")
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
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

