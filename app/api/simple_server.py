"""
Simplified API for document validation - Direct file upload without SQS.

This API:
1. Accepts file upload directly
2. Validates document (pass/fail only)
3. No borrower identification or cross-validation
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Nodes.config.state_models import PipelineState, IngestionState
from Nodes.nodes.ocr_node import OCR
from Nodes.nodes.classification_node import Classification
from Nodes.nodes.extraction_node import Extract
from Nodes.nodes.validation_check_node import ValidationCheck

app = FastAPI(
    title="LendingWise Document Validation API",
    description="Simplified API for document validation - Upload and validate documents",
    version="2.0.0"
)

# Allowed file extensions
ALLOWED_EXTENSIONS = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.webp']


class ValidationResponse(BaseModel):
    """Response model for document validation"""
    success: bool
    status: str  # "pass" or "fail"
    document_name: str
    document_type: Optional[str] = None
    s3_location: str
    message: str
    details: Optional[Dict[str, Any]] = None


def _validate_file_type(filename: str) -> str:
    """Validate file type and return extension."""
    if not filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file_ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    return file_ext


def _validate_mode(mode: str) -> None:
    """Validate processing mode."""
    if mode not in ["llm", "ocr+llm"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {mode}. Must be 'llm' or 'ocr+llm'"
        )


def _create_ingestion_state(
    file_path: str,
    file_ext: str,
    content_size: int,
    document_name: str,
    mode: str,
    FPCID: str,
    LMRId: str
) -> IngestionState:
    """Create ingestion state for local file processing."""
    content_type = f"image/{file_ext[1:]}" if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp'] else "application/pdf"
    
    return IngestionState(
        s3_bucket="local",
        s3_key=file_path,
        metadata_s3_path=None,
        FPCID=FPCID,
        LMRId=LMRId,
        checklistId=None,
        airecordid=None,
        doc_id=None,
        user_id=None,
        document_name=document_name,
        sqs_receipt_handle=None,
        sqs_message_id="LOCAL",
        document_type=None,
        agent_name="Document Validation Agent",
        agent_type=None,
        tool=mode,
        source_url=None,
        content_type=content_type,
        uploaded_at=datetime.now().isoformat(),
        size_bytes=content_size,
        etag=None,
        prefix_parts=None,
        raw_metadata=None,
    )


def _run_validation_pipeline(state: PipelineState, document_name: str) -> ValidationResponse:
    """Run the validation pipeline and return response."""
    print(f"[API] Running OCR...")
    state = OCR(state)
    
    print(f"[API] Running Classification...")
    state = Classification(state)
    
    if not state.classification or not state.classification.passed:
        return ValidationResponse(
            success=False,
            status="fail",
            document_name=document_name,
            document_type=None,
            s3_location="local",
            message="Document classification failed. Please upload a valid document.",
            details={"reason": state.classification.message if state.classification else "Classification failed"}
        )
    
    print(f"[API] Running Extraction...")
    state = Extract(state)
    
    print(f"[API] Running Validation...")
    state = ValidationCheck(state)
    
    validation_passed = state.extraction and state.extraction.passed
    doc_type = state.classification.detected_doc_type if state.classification else None
    
    if validation_passed:
        return ValidationResponse(
            success=True,
            status="pass",
            document_name=document_name,
            document_type=doc_type,
            s3_location="local",
            message="Document validation passed successfully",
            details={
                "extracted_fields": state.extraction.extracted if state.extraction else {},
                "document_type": doc_type
            }
        )
    else:
        return ValidationResponse(
            success=False,
            status="fail",
            document_name=document_name,
            document_type=doc_type,
            s3_location="local",
            message="Document validation failed",
            details={
                "reason": state.extraction.message if state.extraction else "Validation failed",
                "document_type": doc_type
            }
        )


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Document Validation API", "version": "2.0.0"}


@app.post("/validate", response_model=ValidationResponse)
async def validate_document(
    file: UploadFile = File(..., description="Document file to validate (PDF, JPG, PNG)"),
    document_name: Optional[str] = Form(None, description="Document name (e.g., 'Driver License', 'Passport')"),
    mode: Optional[str] = Form("llm", description="Processing mode: 'llm' or 'ocr+llm'"),
    FPCID: Optional[str] = Form("TEST", description="FPCID identifier"),
    LMRId: Optional[str] = Form("TEST", description="LMRId identifier"),
):
    """
    Validate a document by uploading it directly.
    
    Process:
    1. Save file temporarily
    2. Run OCR to extract text
    3. Classify document type
    4. Extract structured data
    5. Validate document (pass/fail)
    
    Returns validation result with pass/fail status.
    NO S3 upload, NO database save - just validation!
    """
    temp_file = None
    
    try:
        file_ext = _validate_file_type(file.filename)
        _validate_mode(mode)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        
        print(f"[API] Processing file: {file.filename}, mode: {mode}")
        
        # Create pipeline state
        state = PipelineState()
        state.ingestion = _create_ingestion_state(
            file_path=temp_file.name,
            file_ext=file_ext,
            content_size=len(content),
            document_name=document_name or file.filename,
            mode=mode,
            FPCID=FPCID,
            LMRId=LMRId
        )
        
        return _run_validation_pipeline(state, document_name or file.filename)
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Document validation failed: {str(e)}")
    finally:
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                print(f"[WARN] Failed to delete temp file: {e}")


@app.post("/validate-generic")
async def validate_generic_document(
    file: UploadFile = File(..., description="Any document file (PDF, JPG, PNG)"),
    prompt: str = Form(..., description="Your instructions: fields to extract, validation rules, questions"),
    document_hint: Optional[str] = Form(None, description="Optional: Document type hint"),
    mode: Optional[str] = Form("llm", description="Processing mode: 'llm' or 'ocr+llm'"),
):
    """
    Validate ANY document type with custom extraction, validation, and questions.
    
    Accepts any document (bank statements, property papers, invoices, contracts, etc.)
    
    **Your prompt can include:**
    - Fields to extract: `Extract: account_number, balance, transaction_date`
    - Validation rules: `Pass if: Balance > 50000, Document date is within last 3 months`
    - Questions: `Calculate monthly average balance, What is the net income?`
    """
    temp_file = None
    
    try:
        file_ext = _validate_file_type(file.filename)
        _validate_mode(mode)
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        
        print(f"[API] Generic validation for: {file.filename}, mode: {mode}")
        
        from Nodes.nodes.generic_extraction import run_generic_document_pipeline
        
        result = run_generic_document_pipeline(
            file_path=temp_file.name,
            user_prompt=prompt,
            document_hint=document_hint,
            mode=mode
        )
        
        return {
            "success": result["status"] != "error",
            "status": result["status"],
            "score": result["score"],
            "document_type": result.get("document_type"),
            "doc_extracted_json": result.get("doc_extracted_json", {}),
            "reason": result.get("reason", {}),
            "file_name": file.filename,
            "processing_time_ms": result.get("processing_time_ms", 0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generic validation failed: {str(e)}")
    finally:
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                print(f"[WARN] Failed to delete temp file: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.simple_server:app", host="0.0.0.0", port=8001, reload=True)
