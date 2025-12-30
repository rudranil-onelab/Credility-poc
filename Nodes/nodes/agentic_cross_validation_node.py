"""
Agentic Cross Validation Node for Cross-Document Validation.

This node performs agentic cross-document validation by analyzing consistency of
identities, document numbers, and key information across main and supporting documents.
"""

import json
import time
import base64
from typing import Dict, Any, List, Optional, Tuple
from ..tools.bedrock_client import get_bedrock_client, strip_json_code_fences


def pdf_to_image_base64(pdf_path: str) -> Tuple[str, str]:
    """
    Convert PDF to image (first page) and return base64 encoded data.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        Tuple of (base64_data, media_type)
    """
    try:
        from pdf2image import convert_from_path
        from PIL import Image
        import io
        
        # Convert first page of PDF to image
        images = convert_from_path(pdf_path, first_page=1, last_page=1)
        
        if not images:
            raise Exception("Failed to convert PDF to image")
        
        # Get first page
        img = images[0]
        
        # Convert to JPEG format in memory
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        
        # Encode to base64
        image_data = base64.b64encode(buffer.read()).decode('utf-8')
        
        return image_data, 'image/jpeg'
        
    except ImportError:
        print("[PDF Conversion] pdf2image not installed. Attempting alternative method...")
        # Fallback to PyMuPDF if available
        try:
            import fitz  # PyMuPDF
            import io
            from PIL import Image
            
            doc = fitz.open(pdf_path)
            page = doc[0]  # First page
            
            # Render page to image (higher resolution for better quality)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Convert to JPEG
            buffer = io.BytesIO()
            img.convert('RGB').save(buffer, format='JPEG', quality=95)
            buffer.seek(0)
            
            # Encode to base64
            image_data = base64.b64encode(buffer.read()).decode('utf-8')
            
            doc.close()
            
            return image_data, 'image/jpeg'
            
        except ImportError:
            raise Exception("PDF conversion requires pdf2image or PyMuPDF (fitz) library. Please install with: pip install pdf2image or pip install PyMuPDF")
    except Exception as e:
        raise Exception(f"Failed to convert PDF to image: {str(e)}")


def encode_image_to_base64(file_path: str) -> Tuple[str, str]:
    """
    Encode an image file to base64. Converts PDFs to images first.
    
    Args:
        file_path: Path to the image/PDF file
    
    Returns:
        Tuple of (base64_data, media_type)
    """
    import os
    
    # Determine media type from file extension
    ext = os.path.splitext(file_path)[1].lower()
    
    # Handle PDFs - convert to image first
    if ext == '.pdf':
        print(f"[Image Encoding] Converting PDF to image for Claude vision API...")
        return pdf_to_image_base64(file_path)
    
    # Handle image files directly
    media_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    media_type = media_type_map.get(ext, 'image/jpeg')
    
    # Read and encode file
    with open(file_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    return image_data, media_type


def extract_identifiable_fields(extracted_json: Dict[str, Any], document_type: str) -> Dict[str, Any]:
    """
    Extract key identifiable fields that should match across documents.
    
    Args:
        extracted_json: Extracted document fields
        document_type: Type of document (e.g., "PAN Card", "Tax Return")
    
    Returns:
        Dict with standardized field names for cross-validation
    """
    identifiable_fields = {
        "raw_data": extracted_json,
        "document_type": document_type,
        "extracted_fields": {}
    }
    
    # Common identifier patterns to look for (case-insensitive)
    field_patterns = {
        "name": ["name", "holder_name", "applicant_name", "pan_holder_name", "full_name", 
                 "first_name", "pan_name", "holderName", "applicantName", "assessee_name",
                 "patient_name", "insured_name", "policyholder_name"],
        "pan": ["pan", "pan_number", "panNumber", "pan_no", "pan_id"],
        "aadhaar": ["aadhaar", "aadhaar_number", "aadhaarNumber", "aadhaar_no"],
        "date_of_birth": ["date_of_birth", "dob", "dateOfBirth", "year_of_birth", "yearOfBirth"],
        "father_name": ["father_name", "fatherName", "father_s_name"],
        "gender": ["gender", "sex"],
        "address": ["address", "permanent_address", "residential_address", "presentAddress"],
        "mobile": ["mobile", "phone", "phoneNumber", "mobile_number"],
        "email": ["email", "email_id"],
        "gstin": ["gstin", "gst_number"],
        "tan": ["tan", "tan_number"],
        "udyam": ["udyam", "udyam_number"],
        "income": ["income", "gross_income", "annual_income", "net_income"],
        "employer": ["employer", "company_name", "employer_name", "organization"],
        "policy_number": ["policy_number", "policyNumber", "policy_no", "insurance_policy"],
        "claim_amount": ["claim_amount", "claimAmount", "amount_claimed", "bill_amount"],
        "medical_amount": ["medical_amount", "bill_amount", "treatment_cost", "total_amount"]
    }
    
    # Search for matching fields (case-insensitive)
    for standardized_key, search_patterns in field_patterns.items():
        for extracted_key, extracted_value in extracted_json.items():
            key_lower = extracted_key.lower()
            # Check if any pattern matches this key
            for pattern in search_patterns:
                if pattern.lower() in key_lower:
                    identifiable_fields["extracted_fields"][standardized_key] = {
                        "original_key": extracted_key,
                        "value": extracted_value
                    }
                    break
    
    return identifiable_fields


def run_agentic_cross_validation_pipeline(
    main_file_path: str,
    main_extracted_json: Dict[str, Any],
    main_document_type: str,
    supporting_file_paths: List[str],
    supporting_extracted_jsons: List[Dict[str, Any]],
    supporting_document_types: List[str],
    user_prompt: str,
    mode: str = "ocr+llm",
    supporting_descriptions: Optional[List[str]] = None,
    cross_validation_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run agentic cross-document validation by cross-validating the main document with supporting documents.
    
    This function:
    1. Sends main document image + supporting document images directly to Claude
    2. Uses Claude's vision capabilities to analyze all documents together
    3. Detects inconsistencies, contradictions, and potential fraud
    4. Provides detailed reports on each validation check
    
    NOTE: Supporting documents are sent as raw images WITHOUT OCR to improve accuracy.
    PDFs are automatically converted to images.
    
    Args:
        main_file_path: Path to main document file
        main_extracted_json: Extracted fields from main document (from main validation)
        main_document_type: Type of main document
        supporting_file_paths: List of paths to supporting document files
        supporting_extracted_jsons: List of extracted fields from supporting documents (may be empty)
        supporting_document_types: List of document types for supporting documents
        user_prompt: User's validation rules
        mode: Processing mode ('ocr+llm' or 'llm')
        supporting_descriptions: Optional list of descriptions for each supporting document
        cross_validation_prompt: Optional custom instructions for cross-validation
    
    Returns:
        Dict with:
        - risk_score: 0-100 (higher = more likely fraud or cross-document error)
        - status: "pass", "suspicious", or "fail"
        - consistency_checks: List of field consistency validations
        - contradictions: List of contradictions found
        - document_agreement: Summary of document agreement
        - overall_message: Human-readable summary
        - warnings: List of warning flags
        - processing_time_ms: Time taken for analysis
    """
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"[AGENTIC CROSS VALIDATION] Starting cross-validation with DIRECT IMAGE ANALYSIS")
    print(f"[AGENTIC CROSS VALIDATION] Main document: {main_document_type}")
    print(f"[AGENTIC CROSS VALIDATION] Supporting documents: {len(supporting_file_paths)}")
    print(f"[AGENTIC CROSS VALIDATION] Mode: Sending raw images directly to Claude (NO OCR)")
    print(f"[AGENTIC CROSS VALIDATION] PDFs will be automatically converted to images")
    if cross_validation_prompt:
        print(f"[AGENTIC CROSS VALIDATION] Custom prompt: {cross_validation_prompt[:80]}...")
    print(f"{'='*70}\n")
    
    try:
        # Get Claude client for LLM analysis
        client = get_bedrock_client()
        
        # Build comprehensive prompt for agentic cross-document detection
        today = time.strftime("%Y-%m-%d")
        
        system_prompt = """You are an expert in agentic cross-document validation and consistency analysis with deep knowledge of Indian documents and cross-document validation.

Your task is to perform intelligent cross-validation across multiple documents to detect potential fraud or inconsistencies.

IMPORTANT: You are receiving the ACTUAL DOCUMENT IMAGES. Analyze them directly without relying on pre-extracted data.

AGENTIC CROSS VALIDATION PRINCIPLES:
1. **Identity Consistency**: All documents should refer to the SAME person
   - Names must match (allow for minor spelling variations, initials, order changes)
   - Dates of birth must be identical
   - Identifiable numbers (PAN, Aadhaar, etc.) must match across documents
   
2. **Document Agreement**: Supporting documents should corroborate main document information
   - If main document shows income, supporting docs should show similar income ranges
   - Employment details should be consistent with employer in supporting docs
   - Address information should match
   - For insurance claims: medical bills should match claim amounts and policy details
   
3. **Contradiction Detection**: Flag any conflicting information
   - Same field with different values across documents = RED FLAG
   - Missing supporting information when it should exist = YELLOW FLAG
   - Impossible date combinations = RED FLAG
   
4. **Document Relationship Validation**:
   - PAN Card + Tax Return: PAN number must match, name must match, income should be consistent
   - PAN Card + Form 16: PAN must match, employer name should match, income should match
   - PAN Card + Bank Statement: Name must match, identity should align
   - Aadhaar + PAN: Name should match (allow spelling variations), DOB must match
   - Insurance Policy + Medical Bill: Patient name must match, amounts should align, dates should be logical
   - Insurance Claim + Aadhaar: Name and identity should match
   
5. **Risk Scoring**:
   - 0-20: Very Low Risk (all documents consistent, perfect match)
   - 21-40: Low Risk (minor inconsistencies, likely data entry errors)
   - 41-60: Medium Risk (some contradictions, requires clarification)
   - 61-80: High Risk (significant inconsistencies, suspicious patterns)
   - 81-100: Very High Risk (clear contradictions, likely fraudulent)

Return JSON with detailed cross-validation results."""

        # Prepare supporting documents descriptions
        supporting_docs_descriptions = []
        for i in range(len(supporting_file_paths)):
            desc = {
                "document_index": i + 1,
                "type": supporting_document_types[i] if i < len(supporting_document_types) else "Unknown"
            }
            if supporting_descriptions and i < len(supporting_descriptions):
                desc["user_description"] = supporting_descriptions[i]
            supporting_docs_descriptions.append(desc)

        # Build custom validation instructions section
        custom_validation_section = ""
        if cross_validation_prompt:
            custom_validation_section = f"""

CUSTOM CROSS-VALIDATION INSTRUCTIONS FROM USER:
{cross_validation_prompt}

IMPORTANT: Apply these custom instructions during your cross-validation analysis.
These instructions should guide your validation logic and risk assessment.
"""

        # Build main document context (from OCR extraction)
        main_fields = extract_identifiable_fields(main_extracted_json, main_document_type)

        user_text = f"""Please perform agentic cross-document validation analysis on the images provided.

MAIN DOCUMENT (Image 1):
- Type: {main_document_type}
- Previously extracted key fields (for context): {json.dumps(main_fields.get('extracted_fields', {}), indent=2, ensure_ascii=False)}

SUPPORTING DOCUMENTS (Images 2-{len(supporting_file_paths) + 1}):
{json.dumps(supporting_docs_descriptions, indent=2, ensure_ascii=False)}

USER'S MAIN DOCUMENT VALIDATION RULES:
{user_prompt}
{custom_validation_section}

Perform the following analysis BY EXAMINING THE ACTUAL DOCUMENT IMAGES:

1. **Identity Verification**: Are all documents about the SAME person?
   - Check name consistency (allow spelling variations)
   - Check date of birth match
   - Check unique identifiers (PAN, Aadhaar, etc.)
   - Score: 0-20 points

2. **Document Relationship**: Do supporting documents validate the main document?
   - Check if relationships make sense (e.g., PAN in Tax Return)
   - Check if fields are consistent (e.g., PAN number, employer)
   - Check if information corroborates (e.g., income levels, claim amounts)
   - Pay attention to user descriptions of supporting documents
   - Score: 0-20 points

3. **Contradiction Detection**: Are there any conflicting values?
   - List all contradictions found
   - For each, explain which documents conflict
   - Flag severity (critical/high/medium/low)
   - Score deduction: 10-20 points per critical contradiction

4. **Missing Supporting Information**: Is required information missing?
   - If main doc has PAN, supporting docs should reference it
   - If age shown in main doc, supporting docs should have consistent dates
   - For insurance: check if medical bills match policy details
   - Score deduction: 5-10 points per missing critical info

5. **Custom Validation Rules**: Apply any custom cross-validation instructions provided
   - If user specified custom rules, evaluate them carefully
   - Explain how documents meet or fail custom criteria

6. **Final Risk Assessment**:
   - Compile all findings into a single cross-document risk score (0-100)
   - Higher score = higher cross-document risk
   - Status: "pass" (0-30), "suspicious" (31-70), "fail" (71-100)

Return ONLY valid JSON with this structure:
{{
  "risk_score": <0-100>,
  "status": "<pass|suspicious|fail>",
  "identity_verification": {{
    "is_same_person": <true|false>,
    "confidence": <0-100>,
    "findings": [<list of findings>],
    "score": <0-20>
  }},
  "document_relationship": {{
    "relationships_valid": <true|false>,
    "findings": [<list of findings>],
    "score": <0-20>
  }},
  "consistency_checks": [
    {{
      "field_name": "<standardized field name>",
      "main_value": "<value in main document>",
      "supporting_values": [{{"document_index": <num>, "value": "<value>"}}],
      "status": "<consistent|inconsistent|missing>",
      "message": "<explanation>"
    }}
  ],
  "contradictions": [
    {{
      "field_name": "<field with contradiction>",
      "conflicting_documents": [{{"index": <num>, "type": "<type>", "value": "<value>"}}],
      "severity": "<critical|high|medium|low>",
      "explanation": "<why this is concerning>"
    }}
  ],
  "missing_information": [
    {{
      "field": "<missing field name>",
      "expected_in": "<document type>",
      "severity": "<high|medium|low>",
      "explanation": "<why it should be present>"
    }}
  ],
  "custom_validation_results": [
    {{
      "rule": "<custom rule being checked>",
      "passed": <true|false>,
      "explanation": "<detailed explanation>"
    }}
  ],
  "document_agreement": {{
    "total_documents": <number>,
    "documents_in_agreement": <number>,
    "agreement_percentage": <0-100>,
    "message": "<summary of agreement>"
  }},
  "warnings": [<list of warning messages>],
  "overall_message": "<human-readable summary of cross-document risk and key findings>"
}}

IMPORTANT:
- Analyze the ACTUAL IMAGES provided, not just the extracted fields
- Be thorough in your analysis
- Flag even minor inconsistencies that could indicate fraud or cross-document risk
- Consider document relationships (what documents should contain)
- Consider user-provided descriptions of supporting documents
- Apply custom cross-validation instructions if provided
- Do NOT approve documents with critical contradictions
- Return ONLY valid JSON"""

        print("[AGENTIC CROSS VALIDATION] Encoding images for Claude...")
        
        # Encode main document image
        main_image_data = None
        main_media_type = None
        try:
            main_image_data, main_media_type = encode_image_to_base64(main_file_path)
            print(f"[AGENTIC CROSS VALIDATION] Main document encoded: {main_media_type}")
        except Exception as e:
            print(f"[AGENTIC CROSS VALIDATION] Warning: Could not encode main document: {e}")
        
        # Encode supporting document images
        supporting_images = []
        for i, file_path in enumerate(supporting_file_paths):
            try:
                image_data, media_type = encode_image_to_base64(file_path)
                supporting_images.append((image_data, media_type))
                print(f"[AGENTIC CROSS VALIDATION] Supporting document {i+1} encoded: {media_type}")
            except Exception as e:
                print(f"[AGENTIC CROSS VALIDATION] Warning: Could not encode supporting document {i+1}: {e}")
        
        # Build message content with all images
        message_content = []
        
        # Add main document image first
        if main_image_data:
            message_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": main_media_type,
                    "data": main_image_data
                }
            })
        
        # Add all supporting document images
        for image_data, media_type in supporting_images:
            message_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data
                }
            })
        
        # Add text prompt
        message_content.append({
            "type": "text",
            "text": user_text
        })
        
        print(f"[AGENTIC CROSS VALIDATION] Sending {len(message_content) - 1} images to Claude for analysis...")
        
        # Send to Claude with all images
        response = client.chat_completion(
            messages=[{"role": "user", "content": message_content}],
            system=system_prompt,
            temperature=0,
            max_tokens=8192
        )
        
        # Parse response
        response = strip_json_code_fences(response)
        agentic_cross_validation_result = json.loads(response)
        
        # Ensure required fields exist
        agentic_cross_validation_result.setdefault("risk_score", 50)
        agentic_cross_validation_result.setdefault("status", "suspicious")
        agentic_cross_validation_result.setdefault("consistency_checks", [])
        agentic_cross_validation_result.setdefault("contradictions", [])
        agentic_cross_validation_result.setdefault("document_agreement", {})
        agentic_cross_validation_result.setdefault("overall_message", "Analysis completed")
        agentic_cross_validation_result.setdefault("warnings", [])
        agentic_cross_validation_result.setdefault("custom_validation_results", [])
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        agentic_cross_validation_result["processing_time_ms"] = processing_time_ms
        
        # Log summary
        print(f"\n[AGENTIC CROSS VALIDATION] Analysis complete!")
        print(f"[AGENTIC CROSS VALIDATION] Risk Score: {agentic_cross_validation_result['risk_score']}/100")
        print(f"[AGENTIC CROSS VALIDATION] Status: {agentic_cross_validation_result['status']}")
        print(f"[AGENTIC CROSS VALIDATION] Contradictions found: {len(agentic_cross_validation_result.get('contradictions', []))}")
        if cross_validation_prompt:
            print(f"[AGENTIC CROSS VALIDATION] Custom rules evaluated: {len(agentic_cross_validation_result.get('custom_validation_results', []))}")
        print(f"[AGENTIC CROSS VALIDATION] Processing time: {processing_time_ms}ms")
        print(f"{'='*70}\n")
        
        return agentic_cross_validation_result
        
    except json.JSONDecodeError as e:
        print(f"[Agentic Cross Validation ERROR] Failed to parse LLM response: {e}")
        return {
            "risk_score": 0,
            "status": "error",
            "consistency_checks": [],
            "contradictions": [],
            "document_agreement": {},
            "overall_message": f"Error analyzing documents: {str(e)}",
            "warnings": [f"Failed to parse analysis response: {str(e)}"],
            "custom_validation_results": [],
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }
        
    except Exception as e:
        print(f"[Agentic Cross Validation ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "risk_score": 0,
            "status": "error",
            "consistency_checks": [],
            "contradictions": [],
            "document_agreement": {},
            "overall_message": f"Error analyzing documents: {str(e)}",
            "warnings": [f"Agentic cross validation error: {str(e)}"],
            "custom_validation_results": [],
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }