"""
LLM utilities for OpenAI integration and document processing.
"""

import json
from typing import Dict, Any, Optional

from ..config.settings import OPENAI_API_KEY, OPENAI_MODEL, ROUTE_LABELS


def strip_json_code_fences(s: str) -> str:
    """Remove JSON code fences from string."""
    s = s.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl+1:]
        if s.endswith("```"):
            s = s[:-3].strip()
    return s


def chat_json(model: str, system_text: str, user_payload: dict) -> dict:
    """
    Try strict JSON mode. If it fails or returns malformed JSON, fall back safely.
    Always return a dict (possibly empty) — never raise here.
    Supports both new SDK (OpenAI) and legacy openai.ChatCompletion.
    """
    system_msg = (
        system_text
        + "\n\nReturn a single JSON object only. Do not include any extra text."
    )
    user_msg = (
        "You MUST return a single JSON object only (JSON). No prose, no code fences.\n\n"
        "Payload follows as JSON:\n"
        + json.dumps(user_payload, ensure_ascii=False)
    )

    if not OPENAI_API_KEY:
        return {}

    # Preferred: new SDK
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=OPENAI_API_KEY)
        try:
            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                temperature=0,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
            content = resp.choices[0].message.content
            return json.loads(content)
        except Exception:
            # fallback without response_format
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system_msg + "\n(You must still return JSON.)"},
                        {"role": "user", "content": user_msg},
                    ],
                )
                raw = resp.choices[0].message.content
                raw = strip_json_code_fences(raw)
                return json.loads(raw)
            except Exception:
                return {}
    except Exception:
        # Legacy openai
        try:
            import openai  # type: ignore
            openai.api_key = OPENAI_API_KEY
            try:
                resp = openai.ChatCompletion.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                )
                raw = resp["choices"][0]["message"]["content"]
                raw = strip_json_code_fences(raw)
                return json.loads(raw)
            except Exception:
                return {}
        except Exception:
            return {}


def remove_raw_text_fields(obj: Any) -> Any:
    """Remove raw_text fields from object."""
    if isinstance(obj, dict):
        return {k: remove_raw_text_fields(v) for k, v in obj.items() if k != "raw_text"}
    if isinstance(obj, list):
        return [remove_raw_text_fields(x) for x in obj]
    return obj


def classify_via_image(model: str, image_url: str) -> str:
    """Classify document type via image analysis. INDIAN DOCUMENTS PRIMARY FOCUS."""
    system = f"""You are an expert document classifier specializing in INDIAN identity documents.

INDIAN DOCUMENT TYPES (Primary - look for these first):
- aadhaar: Aadhaar Card - 12-digit UID number (XXXX XXXX XXXX), UIDAI logo, "आधार", "Unique Identification Authority"
- pan_card: PAN Card - 10-char ID (AAAAA9999A format), "Income Tax Department", "आयकर विभाग", "Permanent Account Number"
- voter_id: Voter ID/EPIC - "Election Commission of India", "भारत निर्वाचन आयोग", EPIC number
- indian_passport: Indian Passport - "Republic of India", "भारत गणराज्य", dark blue, MRZ lines
- indian_driving_license: Indian Driving License - State RTO issued, "Transport Department", "परिवहन"
- ration_card: Ration Card - "Public Distribution", food/ration related
- gst_certificate: GST Certificate - GSTIN number (15 chars), "Goods and Services Tax"
- caste_certificate: Caste Certificate - SC/ST/OBC categories, "जाति प्रमाण पत्र"
- income_certificate: Income Certificate - Annual income, "आय प्रमाण पत्र"
- marksheet: Educational Marksheet - Board exam results, student marks

OTHER DOCUMENT TYPES:
- identity: Generic identity document (non-Indian)
- bank_statement: Bank statements, account summaries
- property: Property documents, deeds
- loan: Loan documents
- unknown: Cannot confidently classify

CLASSIFICATION RULES:
1. Look for Hindi/regional Indian text (Devanagari script: ा े ी ो ू)
2. Look for Indian government logos (Ashoka emblem, UIDAI, Income Tax Dept)
3. Check for Indian number formats (Aadhaar: XXXX XXXX XXXX, PAN: AAAAA9999A)
4. If you see "Government of India" or "भारत सरकार", it's an Indian document
5. For Indian documents, choose the SPECIFIC type (aadhaar, pan_card, etc.), NOT generic "identity"

Available labels: {ROUTE_LABELS}
Return ONLY: {{"doc_type": "<label>"}}"""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": "Classify this document. Look carefully for Indian document indicators (Hindi text, UIDAI logo, Income Tax Dept, Aadhaar number format, PAN format, etc.). Return {\"doc_type\": \"<label>\"}."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]},
            ],
        )
        content = resp.choices[0].message.content
        out = json.loads(content)
        label = out.get("doc_type", "unknown")
        # Accept the label if it's in ROUTE_LABELS, otherwise try to map it
        if label in ROUTE_LABELS:
            return label
        # Try to map common variations
        label_lower = label.lower()
        if "aadhaar" in label_lower or "aadhar" in label_lower:
            return "aadhaar"
        if "pan" in label_lower:
            return "pan_card"
        if "voter" in label_lower or "epic" in label_lower:
            return "voter_id"
        if "passport" in label_lower and "india" in label_lower:
            return "indian_passport"
        if "driv" in label_lower and ("india" in label_lower or "license" in label_lower):
            return "indian_driving_license"
        if "identity" in label_lower or "id" in label_lower:
            return "identity"
        return "unknown"
    except Exception as e:
        print(f"[classify_via_image ERROR] {e}")
        return "unknown"


def extract_via_image(model: str, doc_type: str, image_url: str, prompts_by_type: Dict[str, str]) -> Dict[str, Any]:
    """Extract data from document image."""
    system = (
        prompts_by_type.get(doc_type, prompts_by_type["unknown"]) +
        "\nReturn ONE JSON object only."
    )
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract all structured data from this document image per the rules."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]},
            ],
        )
        content = resp.choices[0].message.content
        try:
            return remove_raw_text_fields(json.loads(content))
        except Exception:
            return {}
    except Exception:
        return {}


def extract_image_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract EXIF and other metadata from an image file.
    
    Args:
        file_path: Path to the image file
    
    Returns:
        Dict with metadata including:
        - exif: EXIF data (camera, software, timestamps, etc.)
        - basic: Basic file metadata (size, format, dimensions)
        - tampering_indicators: Pre-analyzed suspicious patterns
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        import os
        from datetime import datetime
        
        metadata = {
            "exif": {},
            "basic": {},
            "tampering_indicators": []
        }
        
        # Get basic file info
        file_stats = os.stat(file_path)
        metadata["basic"] = {
            "file_size_bytes": file_stats.st_size,
            "file_modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            "file_created": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
        }
        
        # Open image and extract EXIF
        with Image.open(file_path) as img:
            metadata["basic"]["format"] = img.format
            metadata["basic"]["mode"] = img.mode
            metadata["basic"]["size"] = img.size
            metadata["basic"]["width"] = img.width
            metadata["basic"]["height"] = img.height
            
            # Extract EXIF data
            exif_data = img.getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    try:
                        # Convert bytes to string if needed
                        if isinstance(value, bytes):
                            value = value.decode('utf-8', errors='ignore')
                        metadata["exif"][str(tag)] = str(value)
                    except Exception:
                        pass
        
        # Analyze for suspicious patterns
        editing_software = [
            "photoshop", "gimp", "paint.net", "affinity", "pixlr",
            "adobe", "lightroom", "snapseed", "picsart", "canva",
            "photoscape", "fotor", "befunky"
        ]
        
        software_field = metadata["exif"].get("Software", "").lower()
        if any(editor in software_field for editor in editing_software):
            metadata["tampering_indicators"].append({
                "type": "editing_software_detected",
                "severity": "high",
                "description": f"Image editing software detected in metadata: {metadata['exif'].get('Software')}",
                "value": metadata["exif"].get("Software")
            })
        
        # Check if metadata is suspiciously absent (stripped)
        if not metadata["exif"] or len(metadata["exif"]) < 3:
            metadata["tampering_indicators"].append({
                "type": "missing_metadata",
                "severity": "medium",
                "description": "EXIF metadata is missing or has been stripped (common in edited documents)",
                "value": "No EXIF data found"
            })
        
        # Check for timestamp inconsistencies
        if "DateTime" in metadata["exif"] and "DateTimeOriginal" in metadata["exif"]:
            if metadata["exif"]["DateTime"] != metadata["exif"]["DateTimeOriginal"]:
                metadata["tampering_indicators"].append({
                    "type": "timestamp_mismatch",
                    "severity": "medium",
                    "description": "Modified timestamp differs from original timestamp (suggests editing)",
                    "value": f"Original: {metadata['exif']['DateTimeOriginal']}, Modified: {metadata['exif']['DateTime']}"
                })
        
        return metadata
        
    except Exception as e:
        print(f"[extract_image_metadata ERROR] {e}")
        return {
            "exif": {},
            "basic": {},
            "tampering_indicators": [],
            "error": str(e)
        }


def detect_visual_tampering(
    model: str,
    image_url: str,
    doc_type: str,
    extracted_fields: Dict[str, Any],
    image_metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Detect visual tampering indicators using GPT-4 Vision and metadata analysis.
    
    Analyzes document image for:
    - Font inconsistencies
    - Alignment issues
    - Color mismatches
    - Security feature anomalies
    - Editing artifacts
    - Layout inconsistencies
    - Metadata anomalies (EXIF tampering indicators)
    
    Args:
        model: Model to use (e.g., "gpt-4o")
        image_url: URL or base64 data URI of the image
        doc_type: Document type (e.g., "aadhaar", "pan_card")
        extracted_fields: Extracted document fields for context
        image_metadata: Optional image metadata (EXIF, file info, etc.)
    
    Returns:
        Dict with tampering detection results
    """
    # Build metadata context for the prompt
    metadata_context = ""
    metadata_indicators = []
    
    if image_metadata:
        exif = image_metadata.get("exif", {})
        basic = image_metadata.get("basic", {})
        tampering_indicators = image_metadata.get("tampering_indicators", [])
        
        if exif or basic or tampering_indicators:
            metadata_context = "\n\n8. **IMAGE METADATA ANALYSIS**:"
            
            if tampering_indicators:
                metadata_context += f"\n   SUSPICIOUS METADATA PATTERNS DETECTED ({len(tampering_indicators)}):"
                for indicator in tampering_indicators:
                    metadata_context += f"\n   - {indicator['severity'].upper()}: {indicator['description']}"
                    metadata_indicators.append(indicator)
            
            if exif:
                metadata_context += "\n   EXIF Data Available:"
                if "Software" in exif:
                    metadata_context += f"\n   - Software: {exif['Software']}"
                if "Make" in exif:
                    metadata_context += f"\n   - Camera Make: {exif['Make']}"
                if "Model" in exif:
                    metadata_context += f"\n   - Camera Model: {exif['Model']}"
                if "DateTime" in exif:
                    metadata_context += f"\n   - Date/Time: {exif['DateTime']}"
                
                # Check if metadata is suspiciously minimal
                if len(exif) < 5:
                    metadata_context += "\n   ⚠️ WARNING: Very minimal EXIF data (possible metadata stripping)"
            else:
                metadata_context += "\n   ⚠️ WARNING: No EXIF metadata found (common in edited/stripped images)"
            
            if basic:
                metadata_context += f"\n   File Info: {basic.get('format')} format, {basic.get('width')}x{basic.get('height')}px"
    
    system_prompt = f"""You are an expert document forensics analyst specializing in detecting tampered or manipulated documents.

Analyze this {doc_type} document image carefully for signs of visual tampering or manipulation.

CHECK FOR THESE INDICATORS:

1. **Security Feature Anomalies**:
   - Missing watermarks, logos, or holograms that should be present
   - Security features that look suspicious or poorly integrated
   - Missing QR codes or barcodes (for documents that require them)
   - Security patterns that appear inconsistent

2. **Editing Artifacts**:
   - Visible editing boundaries (sharp edges, pixelation)
   - Cloned or duplicated regions (copy-paste artifacts)
   - Blur patches indicating content was erased and replaced
   - Compression artifacts in specific regions (suggesting re-editing)
   - Shadow inconsistencies (text shadows don't match lighting)

3. **Photo Quality Issues**:
   - Excessive blur in specific regions (not overall photo quality)
   - Multiple compression artifacts suggesting multiple edits
   - Quality variations across the document

4. **Content Consistency** (compare with extracted fields):
   - Visual layout doesn't match expected document template
   - Fields are in unexpected positions for this document type
{metadata_context}

METADATA TAMPERING INDICATORS:
- Editing software in EXIF (Photoshop, GIMP, etc.) = HIGH RISK
- Missing/stripped EXIF data = MEDIUM RISK (common in edited documents)
- Timestamp inconsistencies in EXIF = MEDIUM RISK
- Unusual software field values = MEDIUM RISK

IMPORTANT RULES:
- Distinguish between poor photo quality (acceptable) and editing artifacts (suspicious)
- Real documents may have some blur from photography - that's normal
- Focus on INCONSISTENCIES within the document, not overall quality
- Missing security features are more suspicious than poor quality photos
- Metadata tampering indicators should INCREASE risk score significantly

Return JSON with this EXACT structure:
{{
  "tampering_detected": true or false,
  "confidence": "high|medium|low",
  "risk_score": 0-100,
  "indicators": [
    {{
      "type": "security_feature|editing_artifact|layout|quality|metadata",
      "severity": "high|medium|low",
      "description": "Clear description of what was found",
      "location": "where on document (e.g., 'name field', 'date section', 'top-right', 'metadata')"
    }}
  ],
  "summary": "Brief overall assessment"
}}

Return ONLY valid JSON, no markdown or explanations."""

    # Build user prompt with metadata info
    metadata_summary = ""
    if image_metadata and image_metadata.get("tampering_indicators"):
        metadata_summary = f"\n\nIMAGE METADATA WARNINGS ({len(image_metadata['tampering_indicators'])}):\n"
        for ind in image_metadata['tampering_indicators']:
            metadata_summary += f"- {ind['description']}\n"
    
    user_prompt = f"""Analyze this {doc_type} document for visual tampering indicators.

Extracted fields for context:
{json.dumps(extracted_fields, indent=2, ensure_ascii=False)[:500]}
{metadata_summary}
Look carefully at the document image and identify any visual inconsistencies, editing artifacts, or security feature anomalies that suggest tampering.
Consider both the visual appearance AND the metadata analysis above."""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]},
            ],
        )
        
        content = resp.choices[0].message.content
        result = json.loads(content)
        
        # Ensure default values
        result.setdefault("tampering_detected", False)
        result.setdefault("confidence", "low")
        result.setdefault("risk_score", 0)
        result.setdefault("indicators", [])
        result.setdefault("summary", "No tampering detected")
        
        # Add metadata-based indicators to the result
        if metadata_indicators:
            for meta_ind in metadata_indicators:
                result["indicators"].append({
                    "type": "metadata",
                    "severity": meta_ind["severity"],
                    "description": meta_ind["description"],
                    "location": "metadata"
                })
            
            # Increase risk score for metadata indicators
            # High severity: +25 points, Medium: +15, Low: +5
            metadata_risk_boost = sum(
                25 if ind["severity"] == "high" else 15 if ind["severity"] == "medium" else 5
                for ind in metadata_indicators
            )
            result["risk_score"] = min(100, result["risk_score"] + metadata_risk_boost)
            
            # If risk score is high, mark as tampering detected
            if result["risk_score"] >= 70:
                result["tampering_detected"] = True
                if result["confidence"] == "low":
                    result["confidence"] = "medium"
        
        return result
        
    except Exception as e:
        print(f"[detect_visual_tampering ERROR] {e}")
        import traceback
        traceback.print_exc()
        # Return safe defaults on error
        return {
            "tampering_detected": False,
            "confidence": "low",
            "risk_score": 0,
            "indicators": [],
            "summary": f"Tampering detection failed: {str(e)}",
            "error": str(e)
        }