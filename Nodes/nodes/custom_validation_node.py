"""
Custom Validation Node for Dynamic Agent API.

This node validates extracted document fields against user-defined prompts.
It REPLACES the default validation logic with user-defined rules.
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any

from openai import OpenAI

from ..config.state_models import PipelineState
from ..utils.helpers import log_agent_event


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
    
    # Get OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        result = {
            "status": "error",
            "score": 0,
            "reason": ["OpenAI API key not configured"],
            "checks": []
        }
        log_agent_event(state, "Custom Validation", "completed", {"status": "error", "reason": "no_api_key"})
        return result
    
    client = OpenAI(api_key=api_key)
    
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
        print(f"[Custom Validation] Running validation with user prompt...")
        print(f"[Custom Validation] Document type: {document_type}")
        print(f"[Custom Validation] Fields to validate: {list(extracted_fields.keys())}")
        
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        
        result = json.loads(response.choices[0].message.content)
        
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
                "tampering_details": None
            }
        
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
                    print(f"[TAMPERING] Calling GPT-4 Vision for tampering analysis...")
                    
                    from ..tools.llm_services import detect_visual_tampering
                    
                    # Run tampering detection with metadata
                    tampering_result = detect_visual_tampering(
                        model="gpt-4o",
                        image_url=image_url,
                        doc_type=doc_type,
                        extracted_fields=extracted_fields,
                        image_metadata=image_metadata  # Pass metadata for analysis
                    )
                    
                    print(f"[TAMPERING] Analysis complete, processing results...")
                    
                    if tampering_result:
                        tampering_score = tampering_result.get("risk_score", 0)
                        
                        # Determine tampering status: pass if risk < 70, fail if >= 70
                        if tampering_score >= 70:
                            tampering_status = "fail"
                        else:
                            tampering_status = "pass"
                        
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
            "tampering_details": tampering_details  # Include full tampering analysis with metadata
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
            "tampering_details": None
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

