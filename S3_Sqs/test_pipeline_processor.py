"""
Test Pipeline Processor - Single file for quick testing without SQS/DB dependencies.

This module provides a standalone endpoint to test the complete document processing pipeline.
It bypasses SQS queue and database operations for rapid testing and development.

Endpoint: POST /processor/process-document
"""

import os
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from urllib.parse import urlparse

from fastapi import HTTPException, status
from pydantic import BaseModel, Field, validator

# Import pipeline components - add project root to path
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Nodes.config.state_models import PipelineState, IngestionState
from Nodes.nodes.ocr_node import OCR
from Nodes.nodes.classification_node import Classification
from Nodes.nodes.extraction_node import Extract
from Nodes.nodes.validation_check_node import ValidationCheck


# ============================================================================
# Request/Response Models
# ============================================================================

class TestProcessRequest(BaseModel):
    """Request model for test pipeline processing."""
    
    FPCID: str = Field(..., description="FPC ID", example="3580")
    s3_file_url: str = Field(..., description="S3 file URL (s3://bucket/key)", example="s3://lendingwise-aiagent/path/to/document.jpg")
    document_name: Optional[str] = Field(default=None, description="Document name (optional - will be auto-detected from OCR if not provided)", example="Driver's License")
    agent_name: str = Field(..., description="Agent name", example="Identity Verification Agent")
    tool: str = Field(..., description="Tool used", example="ocr+llm")
    
    # Optional fields
    LMRId: Optional[str] = Field(default="1", description="LMR ID", example="1")
    checklistId: Optional[str] = Field(default=None, description="Checklist ID", example="163")
    user_id: Optional[str] = Field(default=None, description="User ID", example="12")
    doc_id: Optional[str] = Field(default=None, description="Document ID", example="23")
    
    @validator('s3_file_url')
    def validate_s3_url(cls, v):
        """Validate S3 URL format."""
        if not v.startswith('s3://'):
            raise ValueError('s3_file_url must start with s3://')
        
        parsed = urlparse(v)
        if not parsed.netloc or not parsed.path:
            raise ValueError('Invalid S3 URL format. Expected: s3://bucket/key')
        
        return v
    
    @validator('tool')
    def validate_tool(cls, v):
        """Validate tool value."""
        valid_tools = ['ocr+llm', 'llm']
        if v not in valid_tools:
            raise ValueError(f'tool must be one of: {", ".join(valid_tools)}')
        return v


class TestProcessResponse(BaseModel):
    """Response model for test pipeline processing."""
    
    success: bool
    message: str
    processing_id: str
    
    # Summary report (concise, user-friendly)
    summary_report: Dict[str, Any]
    
    # Input information
    input_data: Dict[str, Any]
    
    # Processing results
    pipeline_results: Dict[str, Any]
    
    # Final status
    overall_status: str  # "pass", "fail", "human_verification_needed"
    
    # Timestamp
    timestamp: str


# ============================================================================
# Helper Functions
# ============================================================================

def parse_s3_url(s3_url: str) -> tuple[str, str]:
    """
    Parse S3 URL into bucket and key.
    
    Args:
        s3_url: S3 URL in format s3://bucket/key
        
    Returns:
        Tuple of (bucket, key)
    """
    parsed = urlparse(s3_url)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    return bucket, key


def create_test_ingestion_state(
    s3_bucket: str,
    s3_key: str,
    FPCID: str,
    LMRId: str,
    document_name: Optional[str],
    agent_name: str,
    tool: str,
    checklistId: Optional[str] = None,
    user_id: Optional[str] = None,
    doc_id: Optional[str] = None
) -> IngestionState:
    """
    Create a test ingestion state without database lookup.
    
    This bypasses the normal ingestion node which polls SQS and fetches DB context.
    """
    # Generate a test airecordid
    airecordid = str(uuid.uuid4())
    
    # Create minimal metadata
    current_time = datetime.now(timezone.utc).isoformat()
    
    # Extract date from S3 key if possible, otherwise use current date
    year = str(datetime.now().year)
    month = f"{datetime.now().month:02d}"
    day = f"{datetime.now().day:02d}"
    
    # Try to extract date from S3 path (format: .../YYYY/MM/DD/...)
    path_parts = s3_key.split('/')
    if len(path_parts) >= 3:
        for i in range(len(path_parts) - 2):
            if path_parts[i].isdigit() and len(path_parts[i]) == 4:  # Year
                if path_parts[i+1].isdigit() and len(path_parts[i+1]) == 2:  # Month
                    if path_parts[i+2].isdigit() and len(path_parts[i+2]) == 2:  # Day
                        year = path_parts[i]
                        month = path_parts[i+1]
                        day = path_parts[i+2]
                        break
    
    # Determine content type from file extension
    ext = os.path.splitext(s3_key)[1].lower()
    content_type_map = {
        '.pdf': 'application/pdf',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif'
    }
    content_type = content_type_map.get(ext, 'application/octet-stream')
    
    # Use placeholder if document_name not provided (will be auto-detected from OCR)
    final_document_name = document_name or "Unknown Document"
    
    # Create raw metadata
    raw_metadata = {
        "FPCID": FPCID,
        "LMRId": LMRId,
        "file_name": os.path.basename(s3_key),
        "s3_bucket": s3_bucket,
        "s3_key": s3_key,
        "uploaded_at": current_time,
        "content_type": content_type,
        "document_name": final_document_name,
        "checklistId": checklistId,
        "airecordid": airecordid,
        "doc_id": doc_id,
        "user_id": user_id,
        "prefix_parts": {"year": year, "month": month, "day": day},
        "_source": "test_pipeline"
    }
    
    # Create ingestion state
    ingestion_state = IngestionState(
        s3_bucket=s3_bucket,
        s3_key=s3_key,
        metadata_s3_path=f"s3://{s3_bucket}/test/metadata/{os.path.basename(s3_key)}.json",
        FPCID=FPCID,
        LMRId=LMRId,
        checklistId=checklistId,
        airecordid=airecordid,
        doc_id=doc_id,
        user_id=user_id,
        document_name=final_document_name,
        document_type="identity" if "identity" in agent_name.lower() else None,
        agent_name=agent_name,
        agent_type=None,
        tool=tool,
        source_url=f"s3://{s3_bucket}/{s3_key}",
        content_type=content_type,
        uploaded_at=current_time,
        size_bytes=None,
        etag=None,
        prefix_parts={"year": year, "month": month, "day": day},
        raw_metadata=raw_metadata
    )
    
    return ingestion_state


def format_pipeline_results(state: PipelineState) -> Dict[str, Any]:
    """
    Format pipeline state into a clean JSON response.
    
    Args:
        state: Final pipeline state
        
    Returns:
        Dictionary with formatted results
    """
    results = {
        "ingestion": None,
        "ocr": None,
        "classification": None,
        "extraction": None,
        "validation": None
    }
    
    # Ingestion results
    if state.ingestion:
        results["ingestion"] = {
            "s3_bucket": state.ingestion.s3_bucket,
            "s3_key": state.ingestion.s3_key,
            "FPCID": state.ingestion.FPCID,
            "LMRId": state.ingestion.LMRId,
            "document_name": state.ingestion.document_name,
            "agent_name": state.ingestion.agent_name,
            "tool": state.ingestion.tool,
            "content_type": state.ingestion.content_type
        }
    
    # OCR results
    if state.ocr:
        results["ocr"] = {
            "mode": state.ocr.mode,
            "doc_category": state.ocr.doc_category,
            "document_name": state.ocr.document_name,
            "ocr_json": state.ocr.ocr_json,
            "detected_doc_type": state.ocr.ocr_json.get("doc_type") if state.ocr.ocr_json else None
        }
    
    # Classification results
    if state.classification:
        results["classification"] = {
            "expected_category": state.classification.expected_category,
            "detected_doc_type": state.classification.detected_doc_type,
            "passed": state.classification.passed,
            "message": state.classification.message
        }
        
        # Add additional classification details if available
        if hasattr(state.classification, '__dict__'):
            extra_fields = {
                "subtype_expected": getattr(state.classification, 'subtype_expected', None),
                "subtype_detected": getattr(state.classification, 'subtype_detected', None),
                "category_match": getattr(state.classification, 'category_match', None),
                "name_mismatch": getattr(state.classification, 'name_mismatch', None)
            }
            results["classification"].update({k: v for k, v in extra_fields.items() if v is not None})
    
    # Extraction results
    if state.extraction:
        results["extraction"] = {
            "passed": state.extraction.passed,
            "message": state.extraction.message,
            "extracted_fields": state.extraction.extracted,
            "field_count": len(state.extraction.extracted) if state.extraction.extracted else 0
        }
    
    # Validation results - get from captured validation state
    validation_info = {
        "validation_performed": False,
        "status": "unknown"
    }
    
    # Check if validation result was captured
    if hasattr(state, '_validation_result'):
        validation_info = state._validation_result
    
    results["validation"] = validation_info
    
    return results


def determine_overall_status(state: PipelineState) -> str:
    """
    Determine overall processing status based on validation results.
    
    Args:
        state: Final pipeline state
        
    Returns:
        Status string: "pass", "fail", or "human_verification_needed"
    """
    # Check validation result first (most authoritative)
    if hasattr(state, '_validation_result'):
        validation_result = state._validation_result
        if validation_result.get('validation_performed'):
            val_status = validation_result.get('status', 'unknown')
            if val_status == 'pass':
                return "pass"
            elif val_status == 'human_verification_needed':
                return "human_verification_needed"
            elif val_status in ['fail', 'failed']:
                return "fail"
    
    # Fallback: check classification and extraction
    # If classification failed, it's a fail
    if state.classification and not state.classification.passed:
        return "fail"
    
    # If extraction failed, it's a fail
    if state.extraction and not state.extraction.passed:
        return "fail"
    
    # If extraction passed, it's a pass
    if state.extraction and state.extraction.passed:
        return "pass"
    
    # Default to unknown
    return "unknown"


def generate_summary_report(state: PipelineState, overall_status: str) -> Dict[str, Any]:
    """
    Generate a concise summary report with final status, extracted data, and reason.
    
    Args:
        state: Final pipeline state
        overall_status: Overall processing status
        
    Returns:
        Dictionary with summary report
    """
    # Get extracted fields
    extracted_data = {}
    if state.extraction and state.extraction.extracted:
        extracted_data = state.extraction.extracted
    
    # Determine reason for pass/fail
    reason = ""
    validation_details = []
    
    if hasattr(state, '_validation_result') and state._validation_result:
        val_result = state._validation_result
        details = val_result.get('details', {})
        
        if overall_status == "pass":
            reason = details.get('message', 'All validation checks passed successfully.')
        elif overall_status == "human_verification_needed":
            reason = details.get('message', 'Document requires human verification due to validation issues.')
            
            # Extract validation issues
            validation_issues = details.get('validation_issues', {})
            if validation_issues:
                errors = validation_issues.get('errors', [])
                warnings = validation_issues.get('warnings', [])
                validation_details = errors + warnings
            
            # Get LLM validation details if available
            llm_validation = details.get('llm_validation', {})
            if llm_validation:
                llm_errors = llm_validation.get('errors', [])
                llm_warnings = llm_validation.get('warnings', [])
                validation_details.extend(llm_errors + llm_warnings)
        
        elif overall_status == "fail":
            reason = details.get('reason', 'Validation failed')
            
            # Get failure details
            if isinstance(details, dict):
                if 'reason' in details:
                    reason = details['reason']
                if 'validation_issues' in details:
                    validation_issues = details['validation_issues']
                    errors = validation_issues.get('errors', [])
                    validation_details = errors
    
    # Fallback reasons if no validation details
    if not reason:
        if overall_status == "fail":
            if state.classification and not state.classification.passed:
                reason = f"Classification failed: {state.classification.message}"
            elif state.extraction and not state.extraction.passed:
                reason = "Extraction failed"
            else:
                reason = "Validation failed"
        elif overall_status == "pass":
            reason = "All checks passed"
        elif overall_status == "human_verification_needed":
            reason = "Document requires human review"
        else:
            reason = "Unknown status"
    
    # Build summary report
    summary = {
        "final_status": overall_status.upper().replace("_", " "),
        "data_extracted": extracted_data,
        "reason": reason
    }
    
    # Add validation details if present
    if validation_details:
        summary["validation_details"] = validation_details[:5]  # Limit to top 5 issues
    
    return summary


# ============================================================================
# Main Processing Function
# ============================================================================

async def process_document_test(request: TestProcessRequest) -> TestProcessResponse:
    """
    Process a document through the complete pipeline without SQS/DB dependencies.
    
    This function:
    1. Parses S3 URL
    2. Creates test ingestion state
    3. Runs OCR node
    4. Runs classification node
    5. Runs extraction node
    6. Runs validation node
    7. Returns formatted results
    
    Args:
        request: Test process request
        
    Returns:
        Test process response with complete results
        
    Raises:
        HTTPException: If processing fails
    """
    processing_id = str(uuid.uuid4())
    
    print(f"\n{'='*80}")
    print(f"🚀 TEST PIPELINE PROCESSING STARTED")
    print(f"Processing ID: {processing_id}")
    print(f"{'='*80}\n")
    
    try:
        # Parse S3 URL
        print(f"[1/6] Parsing S3 URL: {request.s3_file_url}")
        s3_bucket, s3_key = parse_s3_url(request.s3_file_url)
        print(f"  ✓ Bucket: {s3_bucket}")
        print(f"  ✓ Key: {s3_key}\n")
        
        # Validate file extension
        valid_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.gif']
        file_ext = os.path.splitext(s3_key)[1].lower()
        if file_ext not in valid_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {file_ext}. Supported: {', '.join(valid_extensions)}"
            )
        
        # Create initial pipeline state with test ingestion
        print(f"[2/6] Creating test ingestion state...")
        state = PipelineState()
        state.ingestion = create_test_ingestion_state(
            s3_bucket=s3_bucket,
            s3_key=s3_key,
            FPCID=request.FPCID,
            LMRId=request.LMRId or "1",
            document_name=request.document_name,
            agent_name=request.agent_name,
            tool=request.tool,
            checklistId=request.checklistId,
            user_id=request.user_id,
            doc_id=request.doc_id
        )
        print(f"  ✓ Ingestion state created")
        print(f"  ✓ Test airecordid: {state.ingestion.airecordid}\n")
        
        # Set OCR mode environment variable
        os.environ['OCR_MODE'] = request.tool
        
        # Run OCR node
        print(f"[3/6] Running OCR node (mode: {request.tool})...")
        try:
            state = OCR(state)
            print(f"  ✓ OCR completed")
            if state.ocr and state.ocr.ocr_json:
                doc_type = state.ocr.ocr_json.get('doc_type', 'unknown')
                print(f"  ✓ Detected document type: {doc_type}")
                
                # Auto-detect document_name from OCR if not provided
                if not request.document_name and state.ocr:
                    # Get document_name from OCR results (OCR node already extracts it)
                    detected_doc_name = None
                    
                    # First try OCR document_name attribute (already extracted by OCR node)
                    if state.ocr.document_name:
                        detected_doc_name = state.ocr.document_name
                    # Fallback: check OCR structured JSON directly
                    elif state.ocr.ocr_json and isinstance(state.ocr.ocr_json, dict):
                        detected_doc_name = state.ocr.ocr_json.get('document_name')
                    
                    # Update ingestion state with detected document name
                    if detected_doc_name:
                        state.ingestion.document_name = detected_doc_name
                        state.ingestion.raw_metadata['document_name'] = detected_doc_name
                        print(f"  ✓ Auto-detected document name: {detected_doc_name}\n")
                    else:
                        print(f"  ⚠ Could not auto-detect document name from OCR\n")
                else:
                    print()
        except Exception as e:
            print(f"  ✗ OCR failed: {str(e)}\n")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"OCR processing failed: {str(e)}"
            )
        
        # Run classification node
        print(f"[4/6] Running classification node...")
        try:
            state = Classification(state)
            print(f"  ✓ Classification completed")
            if state.classification:
                status_icon = "✓" if state.classification.passed else "✗"
                print(f"  {status_icon} Classification status: {'PASS' if state.classification.passed else 'FAIL'}")
                if not state.classification.passed:
                    print(f"  ✗ Reason: {state.classification.message}\n")
                else:
                    print()
        except Exception as e:
            print(f"  ✗ Classification failed: {str(e)}\n")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Classification failed: {str(e)}"
            )
        
        # Run extraction node (even if classification failed, to get partial results)
        print(f"[5/6] Running extraction node...")
        try:
            # Temporarily disable DB updates for test mode
            original_update_func = None
            try:
                from Nodes.tools import db
                original_update_func = db.update_tblaigents_by_keys
                db.update_tblaigents_by_keys = lambda *args, **kwargs: print("  [TEST MODE] Skipping DB update")
            except:
                pass
            
            state = Extract(state)
            
            # Restore original function
            if original_update_func:
                db.update_tblaigents_by_keys = original_update_func
            
            print(f"  ✓ Extraction completed")
            if state.extraction and state.extraction.extracted:
                field_count = len(state.extraction.extracted)
                print(f"  ✓ Extracted {field_count} fields\n")
        except Exception as e:
            print(f"  ✗ Extraction failed: {str(e)}\n")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Extraction failed: {str(e)}"
            )
        
        # Run validation node
        print(f"[6/6] Running validation node...")
        captured_validation_result = None
        try:
            # Temporarily disable DB updates for test mode but capture validation results
            original_update_func = None
            
            def capture_validation_result(*args, **kwargs):
                nonlocal captured_validation_result
                updates = kwargs.get('updates', {})
                doc_verification_result = updates.get('doc_verification_result')
                if doc_verification_result:
                    try:
                        import json as _json
                        captured_validation_result = _json.loads(doc_verification_result)
                    except:
                        pass
                print("  [TEST MODE] Skipping DB update (captured validation result)")
            
            try:
                from Nodes.tools import db
                original_update_func = db.update_tblaigents_by_keys
                db.update_tblaigents_by_keys = capture_validation_result
            except:
                pass
            
            state = ValidationCheck(state)
            
            # Restore original function
            if original_update_func:
                db.update_tblaigents_by_keys = original_update_func
            
            # Store captured validation result in state
            if captured_validation_result:
                state._validation_result = {
                    "validation_performed": True,
                    "status": captured_validation_result.get("status", "unknown"),
                    "details": captured_validation_result
                }
                print(f"  ✓ Validation completed - Status: {captured_validation_result.get('status', 'unknown').upper()}\n")
            else:
                state._validation_result = {
                    "validation_performed": True,
                    "status": "unknown",
                    "details": None
                }
                print(f"  ✓ Validation completed\n")
        except Exception as e:
            print(f"  ✗ Validation failed: {str(e)}\n")
            # Don't fail the entire pipeline if validation fails
            print(f"  ⚠️  Continuing without validation results\n")
            state._validation_result = {
                "validation_performed": False,
                "status": "unknown",
                "details": None
            }
        
        # Format results
        print(f"[FINAL] Formatting results...")
        pipeline_results = format_pipeline_results(state)
        overall_status = determine_overall_status(state)
        summary_report = generate_summary_report(state, overall_status)
        
        status_icon = "✓" if overall_status == "pass" else "✗" if overall_status == "fail" else "⚠️"
        print(f"  {status_icon} Overall status: {overall_status.upper()}\n")
        
        # Print summary report to console
        print(f"\n{'='*80}")
        print(f"📊 SUMMARY REPORT")
        print(f"{'='*80}")
        print(f"\n🎯 FINAL STATUS: {summary_report['final_status']}")
        print(f"\n📝 DATA EXTRACTED ({len(summary_report['data_extracted'])} fields):")
        for field, value in summary_report['data_extracted'].items():
            print(f"   • {field}: {value}")
        print(f"\n💬 REASON:")
        print(f"   {summary_report['reason']}")
        if 'validation_details' in summary_report and summary_report['validation_details']:
            print(f"\n⚠️  VALIDATION ISSUES:")
            for issue in summary_report['validation_details']:
                print(f"   • {issue}")
        print(f"\n{'='*80}\n")
        
        # Create response
        response = TestProcessResponse(
            success=True,
            message="Pipeline processing completed successfully",
            processing_id=processing_id,
            summary_report=summary_report,
            input_data={
                "FPCID": request.FPCID,
                "LMRId": request.LMRId,
                "s3_file_url": request.s3_file_url,
                "document_name": request.document_name,
                "agent_name": request.agent_name,
                "tool": request.tool,
                "checklistId": request.checklistId,
                "user_id": request.user_id,
                "doc_id": request.doc_id
            },
            pipeline_results=pipeline_results,
            overall_status=overall_status,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        print(f"{'='*80}")
        print(f"✅ TEST PIPELINE PROCESSING COMPLETED")
        print(f"{'='*80}\n")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ TEST PIPELINE PROCESSING FAILED")
        print(f"Error: {str(e)}")
        print(f"{'='*80}\n")
        
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline processing failed: {str(e)}"
        )


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'TestProcessRequest',
    'TestProcessResponse',
    'process_document_test'
]

