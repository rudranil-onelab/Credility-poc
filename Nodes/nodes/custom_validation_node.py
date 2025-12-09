"""
Custom Validation Node for Dynamic Agent API.

This node validates extracted document fields against user-defined prompts.
It REPLACES the default validation logic with user-defined rules.
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any

from ..config.state_models import PipelineState
from ..utils.helpers import log_agent_event
from ..tools.bedrock_client import get_bedrock_client, strip_json_code_fences


def CustomValidation(state: PipelineState, user_prompt: str) -> Dict[str, Any]:
    """
    Validate extracted fields against user's custom prompt.
    
    This node uses LLM to interpret natural language validation rules
    and evaluate them against the extracted document fields.
    
    Args:
        state: PipelineState with extraction results
        user_prompt: User's custom validation rules in natural language
    
    Returns:
        Dict with:
        - status: "pass", "fail", or "error"
        - score: 0-100 (percentage of rules passed)
        - reason: List of human-readable explanations
        - checks: List of individual rule check results
    """
    log_agent_event(state, "Custom Validation", "start")
    
    # Check if extraction was performed
    if state.extraction is None or state.extraction.extracted is None:
        result = {
            "status": "error",
            "score": 0,
            "reason": ["No extracted fields available - document processing may have failed"],
            "checks": []
        }
        log_agent_event(state, "Custom Validation", "completed", {"status": "error", "reason": "no_extraction"})
        return result
    
    extracted_fields = state.extraction.extracted
    document_type = state.classification.detected_doc_type if state.classification else "unknown"
    document_name = state.ocr.document_name if state.ocr else None
    
    # Get Bedrock Claude client
    client = get_bedrock_client()
    
    # Build system prompt with Indian document context
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    system_prompt = f"""You are a document validation expert specializing in INDIAN and international identity documents.

Your task is to validate extracted document fields against USER-DEFINED rules.
The user's rules are the ONLY criteria for pass/fail - ignore any standard validation rules.

You will receive:
1. Extracted fields from a document (JSON)
2. User's validation rules in natural language

═══════════════════════════════════════════════════════════════════════════════
                    INDIAN DOCUMENT CONTEXT
═══════════════════════════════════════════════════════════════════════════════

COMMON INDIAN DOCUMENT FORMATS:
- Aadhaar Number: 12 digits (format: XXXX XXXX XXXX), VID is 16 digits
- PAN Number: 10 characters (format: AAAAA9999A - 5 letters, 4 digits, 1 letter)
- EPIC (Voter ID): Alphanumeric (e.g., ABC1234567)
- GSTIN: 15 characters (2 digits state code + 10 char PAN + 1 digit + Z + 1 alphanumeric)
- Pincode: 6 digits
- Mobile: 10 digits (may have +91 prefix)

INDIAN DATE FORMATS (handle all):
- DD/MM/YYYY (most common in India)
- DD-MM-YYYY
- DD.MM.YYYY
- YYYY-MM-DD (ISO format)
- "Year of Birth" (only year, common in Aadhaar)

AGE CALCULATION (Indian standard):
- Calculate completed years from DOB to today ({today})
- If only "yearOfBirth" is available, assume birthday is January 1st

MULTI-LANGUAGE FIELD SUPPORT:
- Documents may have fields in English AND regional languages
- Look for both: holderName (English) and holderName_Hi (Hindi), holderName_Ta (Tamil), etc.
- When checking names, consider BOTH English and regional language versions

═══════════════════════════════════════════════════════════════════════════════
                    VALIDATION RULES
═══════════════════════════════════════════════════════════════════════════════

IMPORTANT RULES:
- User's rules have ABSOLUTE PRIORITY
- If user says "pass if age > 54", ONLY check that - nothing else matters unless specified
- Calculate dates/ages accurately using today's date: {today}
- Be strict about numeric requirements (e.g., "12 digits" means exactly 12)
- If a required field is missing, that check should fail
- If a field value is empty or null, treat it as missing
- For Aadhaar: Remove spaces when counting digits (XXXX XXXX XXXX = 12 digits)
- For names: Check both English and regional language fields

DOCUMENT-SPECIFIC VALIDATIONS:
- Aadhaar: Check aadhaarNumber format, yearOfBirth/dateOfBirth, gender
- PAN: Check panNumber format (AAAAA9999A)
- Voter ID: Check epicNumber format
- Passport: Check passportNumber, expirationDate
- Driving License: Check licenseNumber, validTillNonTransport/validTillTransport

Return JSON:
{{
  "status": "pass" or "fail",
  "score": 0-100 (percentage of rules passed, rounded),
  "reason": ["human-readable explanation for each check - be specific about values found"],
  "checks": [
    {{
      "rule": "description of the rule being checked",
      "passed": true or false,
      "value": "actual value found in document (or 'NOT FOUND' if missing)",
      "message": "detailed explanation of why it passed or failed"
    }}
  ]
}}

Return ONLY valid JSON, no markdown or explanations outside the JSON."""

    user_message = f"""
DOCUMENT TYPE: {document_type}
DOCUMENT NAME: {document_name or "Unknown"}

EXTRACTED FIELDS:
{json.dumps(extracted_fields, indent=2, ensure_ascii=False)}

USER'S VALIDATION RULES:
{user_prompt}

Validate the document against ALL the user's rules above. 
Check each rule individually and report the result.
Return JSON result.
"""

    try:
        print(f"[Custom Validation] Running validation with Claude via Bedrock...")
        print(f"[Custom Validation] Document type: {document_type}")
        print(f"[Custom Validation] Fields to validate: {list(extracted_fields.keys())}")
        
        response = client.chat_completion(
            messages=[{"role": "user", "content": user_message}],
            system=system_prompt,
            temperature=0,
            max_tokens=4096
        )
        
        # Parse JSON response
        response = strip_json_code_fences(response)
        result = json.loads(response)
        
        # Ensure required fields exist with defaults
        result.setdefault("status", "error")
        result.setdefault("score", 0)
        result.setdefault("reason", [])
        result.setdefault("checks", [])
        
        # Normalize status to lowercase
        result["status"] = result["status"].lower()
        
        # Ensure score is an integer
        result["score"] = int(result["score"])
        
        print(f"[Custom Validation] Result: {result['status']} (score: {result['score']})")
        print(f"[Custom Validation] Checks performed: {len(result['checks'])}")
        
        log_agent_event(state, "Custom Validation", "completed", {
            "status": result["status"],
            "score": result["score"],
            "checks_count": len(result["checks"])
        })
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"[Custom Validation ERROR] Failed to parse LLM response: {e}")
        result = {
            "status": "error",
            "score": 0,
            "reason": [f"Failed to parse validation response: {str(e)}"],
            "checks": []
        }
        log_agent_event(state, "Custom Validation", "error", {"error": str(e)})
        return result
        
    except Exception as e:
        print(f"[Custom Validation ERROR] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        result = {
            "status": "error",
            "score": 0,
            "reason": [f"Validation error: {str(e)}"],
            "checks": [],
            "error": str(e)
        }
        log_agent_event(state, "Custom Validation", "error", {"error": str(e)})
        return result


def run_custom_validation_pipeline(
    file_path: str,
    user_prompt: str,
    mode: str = "ocr+llm",
    tamper_check: bool = False
) -> Dict[str, Any]:
    """
    Run the complete custom validation pipeline on a local file.
    
    This function:
    1. Runs OCR on the document
    2. Classifies the document type
    3. Extracts structured fields
    4. Validates against user's custom rules
    5. Optionally runs tampering detection (if tamper_check=True)
    
    Args:
        file_path: Path to the document file
        user_prompt: User's custom validation rules
        mode: Processing mode ('ocr+llm' or 'llm')
        tamper_check: Enable tampering detection (default: False)
    
    Returns:
        Dict with:
        - status: "pass", "fail", or "error"
        - score: 0-100
        - reason: List of explanations
        - file_name: Original filename
        - doc_extracted_json: All extracted fields
        - document_type: Detected document type
        - processing_details: Additional metadata
        - tampering_score/status/details: Only if tamper_check=True
    """
    import os
    import time
    from ..tools.aws_services import run_textract_local_file
    from .generic_extraction import validate_with_llm
    
    start_time = time.time()
    file_name = os.path.basename(file_path)
    
    print(f"\n{'='*60}")
    print(f"[Custom Pipeline] Starting validation for: {file_name}")
    print(f"[Custom Pipeline] Mode: {mode}")
    print(f"[Custom Pipeline] User prompt: {user_prompt[:100]}...")
    print(f"{'='*60}\n")
    
    try:
        # Step 1: Run AWS Textract OCR
        print("[Custom Pipeline] Step 1/2: Running AWS Textract OCR...")
        textract_result = run_textract_local_file(file_path)
        
        if not textract_result or not textract_result.get("blocks"):
            return {
                "status": "error",
                "score": 0,
                "reason": ["AWS Textract OCR failed - no text extracted"],
                "file_name": file_name,
                "doc_extracted_json": {},
                "document_type": None,
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "tampering_score": None,
                "tampering_status": None,
                "tampering_details": None,
                # OCR extraction quality fields
                "ocr_extraction_status": "fail",
                "ocr_extraction_confidence": 0.0,
                "ocr_extraction_reason": "AWS Textract could not extract any text from the document. The document may be blank, corrupted, or in an unsupported format."
            }
        
        # Calculate OCR extraction quality from Textract confidence scores
        ocr_quality = calculate_textract_confidence(textract_result)
        print(f"[Custom Pipeline] OCR Quality: {ocr_quality['status']} ({ocr_quality['average_confidence']:.1f}% confidence)")
        print(f"[Custom Pipeline] OCR Reason: {ocr_quality['reason']}")
        
        # Extract text from Textract blocks
        ocr_text = extract_text_from_textract(textract_result)
        print(f"[Custom Pipeline] Textract extracted {len(ocr_text)} characters")
        
        # Step 2: Run LLM validation with user prompt
        print("[Custom Pipeline] Step 2/2: Running LLM validation with user prompt...")
        result = validate_with_llm(
            ocr_text=ocr_text,
            user_prompt=user_prompt,
            textract_blocks=textract_result.get("blocks", [])
        )
        
        # ===== VISUAL TAMPERING DETECTION =====
        tampering_score = None
        tampering_status = None
        tampering_details = None
        
        # Only run tampering detection if tamper_check is enabled
        if tamper_check:
            tampering_status = "enabled"  # Add this line
            print("\n" + "=" * 60)
            print("[🔍 TAMPERING DETECTION] Starting visual tampering check...")
            print("=" * 60)
            
            try:
                # Check if it's an image file for tampering detection
                file_ext = os.path.splitext(file_path)[1].lower()
                
                if file_ext in ['.jpg', '.jpeg', '.png']:
                    print(f"[TAMPERING] Image file detected ({file_ext}) - proceeding with tampering detection")
                    
                    # Step 1: Extract image metadata (EXIF, file info)
                    print(f"[TAMPERING] Extracting image metadata...")
                    from ..tools.llm_services import extract_image_metadata
                    
                    image_metadata = extract_image_metadata(file_path)
                    
                    if image_metadata.get("error"):
                        print(f"[TAMPERING] ⚠ Metadata extraction had errors: {image_metadata['error']}")
                    else:
                        exif_count = len(image_metadata.get("exif", {}))
                        indicator_count = len(image_metadata.get("tampering_indicators", []))
                        print(f"[TAMPERING] ✓ Metadata extracted: {exif_count} EXIF fields, {indicator_count} suspicious indicators")
                        
                        if indicator_count > 0:
                            print(f"[TAMPERING] ⚠ METADATA WARNINGS:")
                            for ind in image_metadata.get("tampering_indicators", []):
                                print(f"[TAMPERING]   - [{ind['severity'].upper()}] {ind['description']}")
                    
                    # Step 2: Prepare image for visual analysis
                    print(f"[TAMPERING] Preparing image for visual analysis...")
                    
                    # Convert local file to base64 data URI for tampering detection
                    import base64
                    with open(file_path, "rb") as f:
                        file_bytes = f.read()
                    
                    base64_image = base64.b64encode(file_bytes).decode('utf-8')
                    
                    # Determine MIME type
                    if file_ext in ['.jpg', '.jpeg']:
                        mime_type = 'image/jpeg'
                    elif file_ext == '.png':
                        mime_type = 'image/png'
                    else:
                        mime_type = 'image/jpeg'
                    
                    image_url = f"data:{mime_type};base64,{base64_image}"
                    
                    # Get document type and extracted fields from validation result
                    doc_type = result.get("document_type", "unknown")
                    extracted_fields = result.get("doc_extracted_json", {})
                    
                    print(f"[TAMPERING] Document type: {doc_type}")
                    print(f"[TAMPERING] Calling Claude Vision for tampering analysis...")
                    
                    from ..tools.llm_services import detect_visual_tampering
                    
                    # Run tampering detection with metadata
                    tampering_result = detect_visual_tampering(
                        model="claude-3-5-haiku",  # Model param kept for API compatibility
                        image_url=image_url,
                        doc_type=doc_type,
                        extracted_fields=extracted_fields,
                        image_metadata=image_metadata  # Pass metadata for analysis
                    )
                    
                    print(f"[TAMPERING] Analysis complete, processing results...")
                    
                    if tampering_result:
                        tampering_score = tampering_result.get("risk_score", 0)
                        
                        print(f"[TAMPERING] ✓ Tampering detection completed")
                        print(f"[TAMPERING]   Risk Score: {tampering_score}/100")
                        print(f"[TAMPERING]   Status: {tampering_status}")
                        print(f"[TAMPERING]   Indicators found: {len(tampering_result.get('indicators', []))}")
                        
                        # Store complete tampering result for response
                        tampering_details = tampering_result
                    else:
                        print(f"[TAMPERING] ⚠ No tampering result received")
                        tampering_details = None
                else:
                    print(f"[TAMPERING] Skipping tampering detection for {file_ext} files (images only)")
                    tampering_details = None
            except Exception as e:
                print(f"[TAMPERING] ✗ ERROR: Tampering detection failed: {e}")
                import traceback
                traceback.print_exc()
                # Continue without tampering detection - set defaults
                tampering_score = None
                tampering_status = None
                tampering_details = None
            
            print("=" * 60)
            print(f"[🔍 TAMPERING DETECTION] Completed - Score: {tampering_score}, Status: {tampering_status}")
            print("=" * 60 + "\n")
        else:
            tampering_status = "disabled"
            print("[TAMPERING] Tampering detection is disabled for this agent (tamper_check=False)")
        # ===== END VISUAL TAMPERING DETECTION =====
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Build final result
        final_result = {
            "status": result.get("status", "error"),
            "score": result.get("score", 0),
            "reason": result.get("reason", {}),
            "file_name": file_name,
            "doc_extracted_json": result.get("doc_extracted_json", {}),
            "document_type": result.get("document_type", "unknown"),
            "processing_time_ms": processing_time_ms,
            "tampering_score": tampering_score,
            "tampering_status": tampering_status,
            "tampering_details": tampering_details,  # Include full tampering analysis with metadata
            # OCR extraction quality fields
            "ocr_extraction_status": ocr_quality["status"],
            "ocr_extraction_confidence": ocr_quality["average_confidence"],
            "ocr_extraction_reason": ocr_quality["reason"]
        }
        
        print(f"\n{'='*60}")
        print(f"[Custom Pipeline] Validation complete!")
        print(f"[Custom Pipeline] Status: {final_result['status']}")
        print(f"[Custom Pipeline] Score: {final_result['score']}")
        print(f"[Custom Pipeline] Processing time: {processing_time_ms}ms")
        print(f"{'='*60}\n")
        
        return final_result
        
    except Exception as e:
        print(f"[Custom Pipeline ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return {
            "status": "error",
            "score": 0,
            "reason": [f"Pipeline error: {str(e)}"],
            "file_name": file_name,
            "doc_extracted_json": {},
            "document_type": None,
            "processing_time_ms": processing_time_ms,
            "tampering_score": None,
            "tampering_status": None,
            "tampering_details": None,
            # OCR extraction quality fields
            "ocr_extraction_status": None,
            "ocr_extraction_confidence": None,
            "ocr_extraction_reason": f"Pipeline error occurred: {str(e)}"
        }


def extract_text_from_textract(textract_result: dict) -> str:
    """Extract plain text from Textract blocks."""
    blocks = textract_result.get("blocks", [])
    lines = []
    
    for block in blocks:
        if block.get("BlockType") == "LINE":
            text = block.get("Text", "")
            if text:
                lines.append(text)
    
    return "\n".join(lines)


def calculate_textract_confidence(textract_result: dict, confidence_threshold: float = 90.0) -> Dict[str, Any]:
    """
    Calculate overall OCR extraction quality from Textract confidence scores.
    
    AWS Textract returns a 'Confidence' score (0-100) for each block.
    This function analyzes these scores to determine extraction quality.
    
    Args:
        textract_result: Result from run_textract_local_file
        confidence_threshold: Minimum acceptable confidence (default 90%)
    
    Returns:
        Dict with:
        - status: "pass" or "fail"
        - average_confidence: Average confidence score
        - reason: Human-readable explanation
        - details: Additional metrics
    """
    blocks = textract_result.get("blocks", [])
    
    if not blocks:
        return {
            "status": "fail",
            "average_confidence": 0.0,
            "reason": "No text could be extracted from the document. The document may be blank, corrupted, or in an unsupported format.",
            "details": {
                "total_blocks": 0,
                "line_count": 0,
                "word_count": 0,
                "low_confidence_blocks": 0
            }
        }
    
    # Collect confidence scores by block type
    line_confidences = []
    word_confidences = []
    all_confidences = []
    low_confidence_blocks = []
    
    for block in blocks:
        block_type = block.get("BlockType", "")
        confidence = block.get("Confidence", 0)
        
        if confidence > 0:
            all_confidences.append(confidence)
            
            if block_type == "LINE":
                line_confidences.append(confidence)
                if confidence < confidence_threshold:
                    text = block.get("Text", "")[:50]
                    low_confidence_blocks.append({
                        "type": "LINE",
                        "text": text,
                        "confidence": round(confidence, 2)
                    })
            elif block_type == "WORD":
                word_confidences.append(confidence)
                if confidence < confidence_threshold:
                    text = block.get("Text", "")
                    low_confidence_blocks.append({
                        "type": "WORD",
                        "text": text,
                        "confidence": round(confidence, 2)
                    })
    
    # Calculate averages
    avg_all = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    avg_lines = sum(line_confidences) / len(line_confidences) if line_confidences else 0
    avg_words = sum(word_confidences) / len(word_confidences) if word_confidences else 0
    
    # Count blocks below threshold
    low_conf_count = len([c for c in all_confidences if c < confidence_threshold])
    low_conf_percentage = (low_conf_count / len(all_confidences) * 100) if all_confidences else 100
    
    # Determine status and reason
    if avg_all >= confidence_threshold:
        status = "pass"
        reason = f"Data extracted successfully with {avg_all:.1f}% average confidence. Document is clear and readable."
    elif avg_all >= 70:
        status = "pass"  # Acceptable but with warning
        reason = f"Data extracted with {avg_all:.1f}% average confidence. Some text may be unclear but extraction is acceptable."
    elif avg_all >= 50:
        status = "fail"
        reason = f"Low extraction quality ({avg_all:.1f}% confidence). Document may be blurry, have poor lighting, or contain hard-to-read text. Consider re-uploading a clearer image."
    else:
        status = "fail"
        # Determine specific reason based on analysis
        if len(all_confidences) < 5:
            reason = f"Very little text could be extracted ({avg_all:.1f}% confidence). Document may be mostly blank, heavily damaged, or the image quality is too poor."
        else:
            reason = f"Data extraction failed ({avg_all:.1f}% confidence). Document appears to be blurry, low resolution, or poorly scanned. Please upload a higher quality image."
    
    return {
        "status": status,
        "average_confidence": round(avg_all, 2),
        "reason": reason,
        "details": {
            "total_blocks": len(blocks),
            "line_count": len(line_confidences),
            "word_count": len(word_confidences),
            "avg_line_confidence": round(avg_lines, 2),
            "avg_word_confidence": round(avg_words, 2),
            "low_confidence_percentage": round(low_conf_percentage, 2),
            "low_confidence_samples": low_confidence_blocks[:5]  # First 5 problematic blocks
        }
    }
