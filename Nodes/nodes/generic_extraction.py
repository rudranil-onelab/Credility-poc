"""
Generic Document Extraction - Extracts user-specified fields from ANY document.
Supports custom validation rules and answers user questions.

This module provides:
1. Generic field extraction from any document type
2. User-defined validation rules
3. Answering user questions/calculations
4. Clear score explanation with pass/fail breakdown
"""

import os
import json
import base64
import io
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from openai import OpenAI


def convert_pdf_to_images(file_path: str) -> List[str]:
    """
    Convert PDF to base64-encoded images (one per page).
    
    Returns list of base64 data URIs for each page.
    """
    try:
        # Try using PyMuPDF (fitz) - faster and no external dependencies
        import fitz  # PyMuPDF
        
        doc = fitz.open(file_path)
        images = []
        
        # Limit to first 5 pages for speed
        max_pages = min(len(doc), 5)
        
        for page_num in range(max_pages):
            page = doc.load_page(page_num)
            # Render at 150 DPI for good quality but reasonable size
            mat = fitz.Matrix(150/72, 150/72)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PNG bytes
            img_bytes = pix.tobytes("png")
            base64_data = base64.b64encode(img_bytes).decode('utf-8')
            images.append(f"data:image/png;base64,{base64_data}")
        
        doc.close()
        print(f"[PDF] Converted {len(images)} pages to images using PyMuPDF")
        return images
        
    except ImportError:
        print("[PDF] PyMuPDF not available, trying pdf2image...")
        
        try:
            # Fallback to pdf2image (requires poppler)
            from pdf2image import convert_from_path
            
            # Convert PDF to images
            pages = convert_from_path(file_path, dpi=150, first_page=1, last_page=5)
            images = []
            
            for page in pages:
                # Convert PIL image to base64
                buffer = io.BytesIO()
                page.save(buffer, format='PNG')
                img_bytes = buffer.getvalue()
                base64_data = base64.b64encode(img_bytes).decode('utf-8')
                images.append(f"data:image/png;base64,{base64_data}")
            
            print(f"[PDF] Converted {len(images)} pages to images using pdf2image")
            return images
            
        except ImportError:
            print("[PDF] pdf2image not available either")
            raise ImportError("Please install PyMuPDF (pip install pymupdf) or pdf2image for PDF support")


def prepare_document_for_vision(file_path: str) -> List[str]:
    """
    Prepare document for Vision API.
    
    For images: Returns single base64 data URI
    For PDFs: Converts to images and returns list of data URIs
    
    Returns:
        List of base64 data URIs (one for images, multiple for PDFs)
    """
    from pathlib import Path
    
    file_ext = Path(file_path).suffix.lower()
    
    # Read file
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    
    if file_ext == '.pdf':
        # Convert PDF to images
        return convert_pdf_to_images(file_path)
    else:
        # For images, create data URI directly
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff'
        }
        mime_type = mime_types.get(file_ext, 'image/jpeg')
        base64_data = base64.b64encode(file_bytes).decode('utf-8')
        return [f"data:{mime_type};base64,{base64_data}"]


def extract_knowledge_from_reference_image(
    image_path_or_url: str,
    user_prompt: str = None
) -> Dict[str, Any]:
    """
    Extract knowledge from a reference/training image to enhance the validation prompt.
    
    This performs a one-time analysis of the reference image to learn:
    - Document type and format
    - Expected fields and their locations
    - Field formats and patterns
    - Visual characteristics
    
    Args:
        image_path_or_url: Path to local file or base64 data URL
        user_prompt: Optional user's original prompt for context
    
    Returns:
        Dict with:
        - extracted_knowledge: Text description of what was learned
        - enhanced_prompt: User's prompt + learned knowledge
        - document_type: Detected document type
        - field_patterns: Dict of field names to expected patterns
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Prepare image URL
    if os.path.isfile(image_path_or_url):
        # Local file - convert to base64
        image_urls = prepare_document_for_vision(image_path_or_url)
        image_url = image_urls[0] if image_urls else None
    else:
        # Already a URL or base64
        image_url = image_path_or_url
    
    if not image_url:
        return {
            "success": False,
            "error": "Could not process reference image",
            "extracted_knowledge": "",
            "enhanced_prompt": user_prompt or ""
        }
    
    system_prompt = """You are an expert document analyzer. Your task is to analyze this REFERENCE/TRAINING image 
and extract detailed knowledge that can be used to validate similar documents in the future.

Analyze the document and extract:

1. **DOCUMENT TYPE**: What type of document is this? (e.g., PAN Card, Aadhaar, Passport, Bank Statement)

2. **EXPECTED FIELDS**: List ALL fields visible in the document with their:
   - Field name
   - Expected format/pattern (e.g., "10 alphanumeric characters", "DD/MM/YYYY date")
   - Example value from the reference (redact sensitive info if needed)
   - Location description (e.g., "top-right corner", "below photo")

3. **FIELD VALIDATION RULES**: For each field, what makes it valid?
   - PAN: 10 chars, format AAAAA9999A
   - Aadhaar: 12 digits
   - Dates: specific format
   - Names: should be readable text

4. **VISUAL CHARACTERISTICS**: 
   - Document layout (portrait/landscape)
   - Key visual elements (logo, photo, QR code, hologram)
   - Color scheme
   - Security features visible

5. **VALIDATION TIPS**: What should be checked to verify this is a genuine document?

Return your analysis as JSON:
{
    "document_type": "Document type name",
    "document_description": "Brief description of what this document is",
    "fields": [
        {
            "name": "Field Name",
            "format": "Expected format/pattern",
            "example": "Example value (redacted if sensitive)",
            "location": "Where on document",
            "validation_rule": "How to validate this field"
        }
    ],
    "visual_characteristics": {
        "layout": "portrait/landscape",
        "key_elements": ["list of visual elements"],
        "color_scheme": "description"
    },
    "validation_tips": ["list of validation tips"],
    "knowledge_summary": "A paragraph summarizing everything learned that can be added to a validation prompt"
}"""

    user_content = [
        {
            "type": "text",
            "text": f"""Analyze this REFERENCE document image and extract knowledge for future validation.

{f"User's validation intent: {user_prompt}" if user_prompt else ""}

Extract all details about the document format, fields, and how to validate similar documents."""
        },
        {
            "type": "image_url",
            "image_url": {"url": image_url, "detail": "high"}
        }
    ]
    
    try:
        print("[Knowledge Extraction] Analyzing reference image...")
        
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        )
        
        knowledge = json.loads(response.choices[0].message.content)
        
        # Build enhanced prompt
        knowledge_text = knowledge.get("knowledge_summary", "")
        
        # Create structured knowledge block
        fields_info = ""
        for field in knowledge.get("fields", []):
            fields_info += f"\n   - {field['name']}: {field.get('format', 'N/A')} ({field.get('validation_rule', '')})"
        
        enhanced_knowledge = f"""

═══════════════════════════════════════════════════════════════════════════════
                    LEARNED FROM REFERENCE DOCUMENT
═══════════════════════════════════════════════════════════════════════════════

Document Type: {knowledge.get('document_type', 'Unknown')}
Description: {knowledge.get('document_description', '')}

Expected Fields:{fields_info}

Validation Tips:
{chr(10).join('   - ' + tip for tip in knowledge.get('validation_tips', []))}

Summary: {knowledge_text}
═══════════════════════════════════════════════════════════════════════════════
"""
        
        # Combine user prompt with extracted knowledge
        enhanced_prompt = user_prompt or ""
        if enhanced_prompt:
            enhanced_prompt = enhanced_prompt.strip() + "\n" + enhanced_knowledge
        else:
            enhanced_prompt = enhanced_knowledge
        
        print(f"[Knowledge Extraction] Successfully extracted knowledge for: {knowledge.get('document_type', 'Unknown')}")
        print(f"[Knowledge Extraction] Found {len(knowledge.get('fields', []))} fields")
        
        return {
            "success": True,
            "document_type": knowledge.get("document_type", "Unknown"),
            "extracted_knowledge": enhanced_knowledge,
            "enhanced_prompt": enhanced_prompt,
            "fields": knowledge.get("fields", []),
            "validation_tips": knowledge.get("validation_tips", []),
            "raw_knowledge": knowledge
        }
        
    except Exception as e:
        print(f"[Knowledge Extraction ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "extracted_knowledge": "",
            "enhanced_prompt": user_prompt or ""
        }


def extract_and_validate_generic(
    image_urls: List[str],
    user_prompt: str,
    ocr_text: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Extract user-specified fields and validate against user's rules.
    
    The user prompt can contain:
    1. Fields to extract (e.g., "Extract: account_number, balance, date")
    2. Validation rules (e.g., "Pass if balance > 50000")
    3. Questions/Instructions (e.g., "Calculate monthly average")
    
    Args:
        image_urls: List of base64 data URIs or presigned URLs (one per page)
        user_prompt: User's complete instructions
        ocr_text: Optional OCR text from Textract
    
    Returns:
        Dict with:
        - status: "pass" or "fail"
        - score: 0-100
        - doc_extracted_json: Only the fields user requested
        - reason: {pass_conditions, fail_conditions, user_questions, score_explanation}
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    system_prompt = f"""You are an expert document analyzer and validator.

TODAY'S DATE: {today}

YOUR TASK:
1. Analyze the document image
2. Extract ONLY the fields that user specified in their prompt
3. Validate against user's rules
4. Answer any questions user asked
5. Calculate score based on how many conditions passed/failed

═══════════════════════════════════════════════════════════════════════════════
                         EXTRACTION RULES
═══════════════════════════════════════════════════════════════════════════════

FIELD EXTRACTION:
- Extract ONLY fields that user explicitly requested
- If user says "Extract: account_number, balance" → only return those 2 fields
- If user doesn't specify fields, extract all relevant fields visible in document
- Use exact field names user specified
- For amounts/numbers, return as numbers (not strings)
- For dates, use format visible in document

DOCUMENT TYPES SUPPORTED:
- Bank statements
- Property documents  
- Identity documents (Aadhaar, PAN, Passport, DL, Voter ID, etc.)
- Invoices
- Contracts
- Certificates
- Tax documents
- Insurance documents
- Any other document type

INDIAN DOCUMENT FORMATS:
- Aadhaar Number: 12 digits (format: XXXX XXXX XXXX)
- PAN Number: 10 characters (AAAAA9999A)
- EPIC (Voter ID): Alphanumeric
- GSTIN: 15 characters
- Mobile: 10 digits (may have +91 prefix)
- Pincode: 6 digits

═══════════════════════════════════════════════════════════════════════════════
                         VALIDATION & SCORING
═══════════════════════════════════════════════════════════════════════════════

CRITICAL VALIDATION RULES:
- "clearly readable" / "readable" / "legible" = You can successfully extract the value from the document
- "present" / "exists" / "available" = The field exists and has a non-empty value
- "valid" = The value follows the expected format for that field type
- If you successfully extracted a field value, it IS "clearly readable" and "present" - PASS IT
- Do NOT fail fields that you successfully extracted just because image quality isn't perfect
- Only fail if you genuinely CANNOT read the value or it's missing/empty

HANDLING SPACES AND FORMATTING:
- When user says "after removing spaces" or "without spaces", you MUST remove spaces before validating
- Example: Aadhaar "6952 3994 9634" → remove spaces → "695239949634" = 12 digits = PASS
- Do NOT fail because raw OCR has spaces if user asked to validate "after removing spaces"
- Aadhaar numbers display as "XXXX XXXX XXXX" format but are 12 digits when spaces removed
- PAN numbers are 10 characters, no spaces
- Apply the transformation the user specified, THEN validate

AVOID DUPLICATE CONDITIONS:
- Each user requirement = ONE condition in pass_conditions or fail_conditions
- "Aadhaar is 12 digits (after removing spaces)" = 1 condition, not 2
- Do NOT create multiple conditions from a single user requirement
- Count unique requirements: if user lists 3 things, you should have exactly 3 conditions total

DATE VALIDATION:
- DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD are all valid date formats
- If a date is successfully extracted, it is "valid" and "readable"
- For DOB: any reasonable date in the past is valid (person should be born before today)

SCORING RULES:
- Count total number of validation conditions user specified
- Check each condition and mark as PASSED or FAILED
- Score = (Passed conditions / Total conditions) × 100
- Round to nearest integer

EXAMPLE:
- If user has 4 conditions and 2 pass: Score = 50%
- If user has 3 conditions and all pass: Score = 100%
- If user has 5 conditions and 1 passes: Score = 20%

PASS/FAIL STATUS:
- status = "pass" ONLY if score = 100 (all conditions pass)
- status = "fail" if any condition fails (score < 100)

═══════════════════════════════════════════════════════════════════════════════
                         USER QUESTIONS
═══════════════════════════════════════════════════════════════════════════════

If user asks questions or gives instructions like:
- "Calculate monthly average"
- "What is the total?"
- "Sum all transactions"
- "Find the difference"
- "Calculate net income"

→ Calculate and provide answers in user_questions section

═══════════════════════════════════════════════════════════════════════════════
                         OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════

Return this EXACT JSON structure:
{{
  "status": "pass" or "fail",
  "score": 0-100 (percentage of conditions that passed),
  "document_type": "detected document type",
  "doc_extracted_json": {{
    // ONLY fields user requested
    // Use exact field names from user's prompt
  }},
  "reason": {{
    "pass_conditions": [
      "✓ [Condition description] - PASSED (actual value: X, required: Y)"
    ],
    "fail_conditions": [
      "✗ [Condition description] - FAILED (found: X, required: Y)"
    ],
    "user_questions": [
      "Q: [User's question] → A: [Your answer with calculation if applicable]"
    ],
    "score_explanation": "X out of Y conditions passed = Z% score"
  }}
}}

IMPORTANT FORMATTING:
1. pass_conditions: Start each with "✓", end with "- PASSED", include actual values
2. fail_conditions: Start each with "✗", end with "- FAILED", show actual vs required values
3. user_questions: Start with "Q:" for question, "→ A:" for answer, show calculations
4. score_explanation: Always explain "X out of Y conditions passed = Z%"
5. If no validation conditions specified, extract fields and set score to 100
6. If no questions asked, user_questions should be empty array []

CRITICAL - SCORE MUST BE CONSISTENT:
- The "score" field MUST match your actual calculation
- If 4 out of 4 conditions pass → score MUST be 100
- If 3 out of 4 conditions pass → score MUST be 75
- If 2 out of 4 conditions pass → score MUST be 50
- NEVER set score to a value that contradicts pass_conditions/fail_conditions counts
- If fail_conditions is empty and all conditions passed → score = 100, status = "pass"
- Double-check: count(pass_conditions) / (count(pass_conditions) + count(fail_conditions)) × 100 = score"""

    # Build user message
    user_content = []
    
    # Add instruction text
    page_info = f"({len(image_urls)} page(s))" if len(image_urls) > 1 else ""
    user_content.append({
        "type": "text",
        "text": f"""Analyze this document {page_info} and follow my instructions:

{user_prompt}

REMEMBER:
1. Extract ONLY the fields I mentioned (if I specified any)
2. Validate EACH condition I mentioned separately
3. Calculate score = (passed conditions / total conditions) × 100
4. Answer my questions in user_questions section with "Q:" and "→ A:" format
5. Explain why score is what it is in score_explanation
6. Show actual values vs required values in pass/fail conditions
7. Combine data from ALL pages if multiple pages provided"""
    })
    
    # Add all images (pages)
    for idx, image_url in enumerate(image_urls):
        user_content.append({
            "type": "image_url",
            "image_url": {"url": image_url, "detail": "high"}
        })
    
    if ocr_text:
        user_content.append({
            "type": "text",
            "text": f"\nOCR Data (use for reference):\n{json.dumps(ocr_text, indent=2, ensure_ascii=False)[:3000]}"
        })
    
    try:
        print(f"[Generic] Processing with GPT-4o Vision...")
        
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Ensure proper structure
        result.setdefault("status", "fail")
        result.setdefault("score", 0)
        result.setdefault("document_type", "unknown")
        result.setdefault("doc_extracted_json", {})
        
        # Ensure reason has all required sections
        if "reason" not in result or not isinstance(result["reason"], dict):
            result["reason"] = {}
        
        result["reason"].setdefault("pass_conditions", [])
        result["reason"].setdefault("fail_conditions", [])
        result["reason"].setdefault("user_questions", [])
        result["reason"].setdefault("score_explanation", "Unable to calculate score")
        
        # Normalize values
        result["status"] = result["status"].lower()
        
        # CRITICAL FIX: Recalculate score from actual pass/fail conditions
        # This prevents LLM inconsistency where score doesn't match the conditions
        pass_count = len(result["reason"]["pass_conditions"])
        fail_count = len(result["reason"]["fail_conditions"])
        total_conditions = pass_count + fail_count
        
        if total_conditions > 0:
            calculated_score = round((pass_count / total_conditions) * 100)
            llm_score = int(result["score"])
            
            # If LLM score doesn't match calculated score, use calculated score
            if calculated_score != llm_score:
                print(f"[Generic] Score mismatch detected: LLM said {llm_score}%, but {pass_count}/{total_conditions} conditions passed = {calculated_score}%")
                print(f"[Generic] Correcting score from {llm_score}% to {calculated_score}%")
                result["score"] = calculated_score
                result["reason"]["score_explanation"] = f"{pass_count} out of {total_conditions} conditions passed = {calculated_score}% score"
            else:
                result["score"] = llm_score
        else:
            # No conditions specified, default to 100
            result["score"] = 100
        
        # Ensure status matches score logic
        if result["score"] == 100 and fail_count == 0:
            result["status"] = "pass"
        elif result["score"] < 100 or fail_count > 0:
            result["status"] = "fail"
        
        print(f"[Generic] Status: {result['status']}")
        print(f"[Generic] Score: {result['score']}%")
        print(f"[Generic] Passed: {len(result['reason']['pass_conditions'])} conditions")
        print(f"[Generic] Failed: {len(result['reason']['fail_conditions'])} conditions")
        print(f"[Generic] Questions answered: {len(result['reason']['user_questions'])}")
        
        return result
        
    except Exception as e:
        print(f"[Generic ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "score": 0,
            "document_type": "unknown",
            "doc_extracted_json": {},
            "reason": {
                "pass_conditions": [],
                "fail_conditions": [f"✗ Processing error: {str(e)} - FAILED"],
                "user_questions": [],
                "score_explanation": "Error occurred during processing"
            }
        }


def run_generic_document_pipeline(
    file_path: str,
    user_prompt: str,
    document_hint: str = None,
    mode: str = "llm"
) -> Dict[str, Any]:
    """
    Run complete generic document pipeline.
    
    This pipeline:
    1. Reads any document (image or PDF)
    2. Converts PDFs to images for Vision API
    3. Extracts user-specified fields using GPT Vision
    4. Validates against user's custom rules
    5. Answers user's questions
    
    Args:
        file_path: Path to document file (PDF, JPG, PNG)
        user_prompt: User's complete instructions including:
            - Fields to extract
            - Validation rules
            - Questions to answer
        document_hint: Optional hint about document type
        mode: 'llm' (Vision only) or 'ocr+llm' (Textract + Vision)
    
    Returns:
        Complete validation result with:
        - status: pass/fail
        - score: 0-100
        - doc_extracted_json: requested fields only
        - reason: {pass_conditions, fail_conditions, user_questions, score_explanation}
    """
    import time
    from pathlib import Path
    
    start_time = time.time()
    file_name = os.path.basename(file_path)
    file_ext = Path(file_path).suffix.lower()
    
    print(f"\n{'='*70}")
    print(f"[Generic Pipeline] Processing: {file_name}")
    print(f"[Generic Pipeline] File type: {file_ext}")
    print(f"[Generic Pipeline] Mode: {mode}")
    if document_hint:
        print(f"[Generic Pipeline] Document hint: {document_hint}")
    print(f"{'='*70}\n")
    
    try:
        # Prepare document for Vision API (handles PDF conversion)
        print("[Generic Pipeline] Preparing document for Vision API...")
        image_urls = prepare_document_for_vision(file_path)
        print(f"[Generic Pipeline] Document prepared: {len(image_urls)} page(s)")
        
        # Add document hint if provided
        full_prompt = user_prompt
        if document_hint:
            full_prompt = f"Document Type Hint: {document_hint}\n\n{user_prompt}"
        
        # OCR for PDFs if mode is ocr+llm (optional enhancement)
        ocr_text = None
        # Note: For now, we rely on GPT Vision which works well for most documents
        # Textract can be added later for complex PDFs if needed
        
        # Extract and validate using GPT-4o Vision
        print("[Generic Pipeline] Running extraction and validation with GPT-4o Vision...")
        result = extract_and_validate_generic(
            image_urls=image_urls,
            user_prompt=full_prompt,
            ocr_text=ocr_text
        )
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        result["file_name"] = file_name
        result["processing_time_ms"] = processing_time_ms
        result["pages_processed"] = len(image_urls)
        
        print(f"\n{'='*70}")
        print(f"[Result] Status: {result['status'].upper()}")
        print(f"[Result] Score: {result['score']}%")
        print(f"[Result] {result['reason'].get('score_explanation', '')}")
        print(f"[Result] Extracted fields: {list(result['doc_extracted_json'].keys())}")
        print(f"[Result] Processing time: {processing_time_ms}ms")
        print(f"{'='*70}\n")
        
        return result
        
    except FileNotFoundError:
        return {
            "status": "error",
            "score": 0,
            "document_type": "unknown",
            "file_name": file_name,
            "doc_extracted_json": {},
            "reason": {
                "pass_conditions": [],
                "fail_conditions": [f"✗ File not found: {file_path} - FAILED"],
                "user_questions": [],
                "score_explanation": "Error: File not found"
            },
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }
    
    except ImportError as e:
        return {
            "status": "error",
            "score": 0,
            "document_type": "unknown",
            "file_name": file_name,
            "doc_extracted_json": {},
            "reason": {
                "pass_conditions": [],
                "fail_conditions": [f"✗ Missing dependency: {str(e)} - FAILED"],
                "user_questions": [],
                "score_explanation": "Error: Please install required packages (pip install pymupdf)"
            },
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        processing_time_ms = int((time.time() - start_time) * 1000)
        return {
            "status": "error",
            "score": 0,
            "document_type": "unknown",
            "file_name": file_name,
            "doc_extracted_json": {},
            "reason": {
                "pass_conditions": [],
                "fail_conditions": [f"✗ Pipeline error: {str(e)} - FAILED"],
                "user_questions": [],
                "score_explanation": "Error during processing"
            },
            "processing_time_ms": processing_time_ms
        }


def validate_with_s3_file(
    s3_bucket: str,
    s3_key: str,
    user_prompt: str,
    document_hint: str = None
) -> Dict[str, Any]:
    """
    Validate a document from S3 with user-defined rules.
    
    Note: For S3 files, we use presigned URLs which work for images.
    For PDFs in S3, consider downloading first and using run_generic_document_pipeline.
    
    Args:
        s3_bucket: S3 bucket name
        s3_key: S3 object key
        user_prompt: User's validation rules and questions
        document_hint: Optional document type hint
    
    Returns:
        Validation result
    """
    import time
    import tempfile
    from pathlib import Path
    from ..tools.aws_services import generate_presigned_url, download_from_s3
    
    start_time = time.time()
    file_name = os.path.basename(s3_key)
    file_ext = Path(s3_key).suffix.lower()
    
    print(f"[Generic S3] Processing: s3://{s3_bucket}/{s3_key}")
    
    try:
        # For PDFs, download and process locally (Vision API doesn't accept PDF URLs)
        if file_ext == '.pdf':
            print("[Generic S3] PDF detected - downloading for local processing...")
            
            # Download to temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_path = temp_file.name
            temp_file.close()
            
            try:
                download_from_s3(s3_bucket, s3_key, temp_path)
                
                # Use local pipeline
                result = run_generic_document_pipeline(
                    file_path=temp_path,
                    user_prompt=user_prompt,
                    document_hint=document_hint
                )
                
                result["s3_location"] = f"s3://{s3_bucket}/{s3_key}"
                return result
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        # For images, use presigned URL directly
        presigned_url = generate_presigned_url(s3_bucket, s3_key, expiration=300)
        
        if not presigned_url:
            return {
                "status": "error",
                "score": 0,
                "document_type": "unknown",
                "file_name": file_name,
                "doc_extracted_json": {},
                "reason": {
                    "pass_conditions": [],
                    "fail_conditions": ["✗ Failed to generate S3 presigned URL - FAILED"],
                    "user_questions": [],
                    "score_explanation": "Error: Could not access S3 file"
                }
            }
        
        # Add document hint to prompt
        full_prompt = user_prompt
        if document_hint:
            full_prompt = f"Document Type Hint: {document_hint}\n\n{user_prompt}"
        
        # Extract and validate (single image)
        result = extract_and_validate_generic(
            image_urls=[presigned_url],  # Wrap in list
            user_prompt=full_prompt
        )
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        result["file_name"] = file_name
        result["s3_location"] = f"s3://{s3_bucket}/{s3_key}"
        result["processing_time_ms"] = processing_time_ms
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "score": 0,
            "document_type": "unknown",
            "file_name": file_name,
            "doc_extracted_json": {},
            "reason": {
                "pass_conditions": [],
                "fail_conditions": [f"✗ S3 processing error: {str(e)} - FAILED"],
                "user_questions": [],
                "score_explanation": "Error during S3 processing"
            },
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }


def validate_with_llm(
    ocr_text: str,
    user_prompt: str,
    textract_blocks: List[Dict] = None
) -> Dict[str, Any]:
    """
    Validate document using LLM with AWS Textract OCR results.
    
    This function:
    1. Takes OCR text from AWS Textract
    2. Sends to GPT-4 with user's validation prompt
    3. Returns extraction and validation results
    
    Args:
        ocr_text: Plain text extracted by Textract
        user_prompt: User's validation rules and extraction instructions
        textract_blocks: Raw Textract blocks for additional context
    
    Returns:
        Dict with status, score, doc_extracted_json, reason
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    # Build key-value pairs from Textract if available
    key_value_pairs = {}
    tables_data = []
    
    if textract_blocks:
        # Extract KEY_VALUE_SET pairs
        key_map = {}
        value_map = {}
        
        for block in textract_blocks:
            block_id = block.get("Id", "")
            block_type = block.get("BlockType", "")
            
            if block_type == "KEY_VALUE_SET":
                entity_types = block.get("EntityTypes", [])
                if "KEY" in entity_types:
                    key_map[block_id] = block
                elif "VALUE" in entity_types:
                    value_map[block_id] = block
        
        # Match keys to values
        for key_id, key_block in key_map.items():
            key_text = ""
            value_text = ""
            
            # Get key text
            for rel in key_block.get("Relationships", []):
                if rel.get("Type") == "CHILD":
                    for child_id in rel.get("Ids", []):
                        for block in textract_blocks:
                            if block.get("Id") == child_id and block.get("BlockType") == "WORD":
                                key_text += block.get("Text", "") + " "
                
                if rel.get("Type") == "VALUE":
                    for value_id in rel.get("Ids", []):
                        if value_id in value_map:
                            value_block = value_map[value_id]
                            for vrel in value_block.get("Relationships", []):
                                if vrel.get("Type") == "CHILD":
                                    for child_id in vrel.get("Ids", []):
                                        for block in textract_blocks:
                                            if block.get("Id") == child_id and block.get("BlockType") == "WORD":
                                                value_text += block.get("Text", "") + " "
            
            key_text = key_text.strip()
            value_text = value_text.strip()
            
            if key_text:
                key_value_pairs[key_text] = value_text
        
        # Extract tables
        for block in textract_blocks:
            if block.get("BlockType") == "TABLE":
                tables_data.append("Table detected in document")
    
    system_prompt = f"""You are an expert document analyzer and validator.

TODAY'S DATE: {today}

YOUR TASK:
1. Analyze the document text extracted by AWS Textract
2. Extract fields that user specified in their prompt
3. Validate against user's rules
4. Answer any questions user asked
5. Calculate score based on how many conditions passed/failed

═══════════════════════════════════════════════════════════════════════════════
                         EXTRACTION RULES
═══════════════════════════════════════════════════════════════════════════════

FIELD EXTRACTION:
- Extract ONLY fields that user explicitly requested
- If user doesn't specify fields, extract all relevant fields visible in document
- Use exact field names user specified
- For amounts/numbers, return as numbers (not strings)
- For dates, use format visible in document

DOCUMENT TYPES SUPPORTED:
- Bank statements
- Property documents (tax bills, deeds, etc.)
- Identity documents (Aadhaar, PAN, Passport, DL, Voter ID, etc.)
- Invoices
- Contracts
- Certificates
- Tax documents
- Insurance documents
- Any other document type

═══════════════════════════════════════════════════════════════════════════════
                         VALIDATION & SCORING
═══════════════════════════════════════════════════════════════════════════════

CRITICAL VALIDATION RULES:
- "clearly readable" / "readable" / "legible" = You can successfully extract the value from the OCR text
- "present" / "exists" / "available" = The field exists and has a non-empty value in the OCR output
- "valid" = The value follows the expected format for that field type
- If you found a field value in the OCR text, it IS "clearly readable" and "present" - PASS IT
- Do NOT fail fields that exist in the OCR output just because you're unsure
- Only fail if the value is genuinely missing, empty, or malformed

HANDLING SPACES AND FORMATTING:
- When user says "after removing spaces" or "without spaces", you MUST remove spaces before validating
- Example: Aadhaar "6952 3994 9634" → remove spaces → "695239949634" = 12 digits = PASS
- Do NOT fail because raw OCR has spaces if user asked to validate "after removing spaces"
- Aadhaar numbers display as "XXXX XXXX XXXX" format but are 12 digits when spaces removed
- PAN numbers are 10 characters, no spaces
- Apply the transformation the user specified, THEN validate

AVOID DUPLICATE CONDITIONS:
- Each user requirement = ONE condition in pass_conditions or fail_conditions
- "Aadhaar is 12 digits (after removing spaces)" = 1 condition, not 2
- Do NOT create multiple conditions from a single user requirement
- Count unique requirements: if user lists 3 things, you should have exactly 3 conditions total

DATE VALIDATION:
- DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD are all valid date formats
- If a date is found in OCR output, it is "valid" and "readable"
- For DOB: any reasonable date in the past is valid (person should be born before today)

SCORING RULES:
- Count total number of validation conditions user specified
- Check each condition and mark as PASSED or FAILED
- Score = (Passed conditions / Total conditions) × 100
- Round to nearest integer

PASS/FAIL STATUS:
- status = "pass" ONLY if score = 100 (all conditions pass)
- status = "fail" if any condition fails (score < 100)

═══════════════════════════════════════════════════════════════════════════════
                         OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════

Return this EXACT JSON structure:
{{
  "status": "pass" or "fail",
  "score": 0-100 (percentage of conditions that passed),
  "document_type": "detected document type",
  "doc_extracted_json": {{
    // ONLY fields user requested or all relevant fields if not specified
  }},
  "reason": {{
    "pass_conditions": [
      "✓ [Condition description] - PASSED (actual value: X, required: Y)"
    ],
    "fail_conditions": [
      "✗ [Condition description] - FAILED (found: X, required: Y)"
    ],
    "user_questions": [
      "Q: [User's question] → A: [Your answer with calculation if applicable]"
    ],
    "score_explanation": "X out of Y conditions passed = Z% score"
  }}
}}"""

    # Build user message with OCR data
    user_message = f"""USER'S VALIDATION PROMPT:
{user_prompt}

═══════════════════════════════════════════════════════════════════════════════
                    DOCUMENT TEXT (from AWS Textract OCR)
═══════════════════════════════════════════════════════════════════════════════

{ocr_text}

"""
    
    if key_value_pairs:
        user_message += f"""
═══════════════════════════════════════════════════════════════════════════════
                    KEY-VALUE PAIRS (from Textract Forms)
═══════════════════════════════════════════════════════════════════════════════

{json.dumps(key_value_pairs, indent=2, ensure_ascii=False)}
"""
    
    user_message += """
═══════════════════════════════════════════════════════════════════════════════

Based on the document text above, extract the requested fields and validate against the user's rules.
Return your response as valid JSON."""

    try:
        print(f"[LLM] Sending to GPT-4o for validation...")
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        result = json.loads(response.choices[0].message.content)
        
        print(f"[LLM] Validation complete - Status: {result.get('status')}, Score: {result.get('score')}")
        
        return result
        
    except Exception as e:
        print(f"[LLM] Error during validation: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "status": "error",
            "score": 0,
            "document_type": "unknown",
            "doc_extracted_json": {},
            "reason": {
                "pass_conditions": [],
                "fail_conditions": [f"✗ LLM validation error: {str(e)} - FAILED"],
                "user_questions": [],
                "score_explanation": "Error during LLM validation"
            }
        }
