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

from ..tools.bedrock_client import get_bedrock_client, strip_json_code_fences


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
    client = get_bedrock_client()
    
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
IMPORTANT: Return ONLY valid JSON, no markdown or explanations.

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

    user_text = f"""Analyze this REFERENCE document image and extract knowledge for future validation.

{f"User's validation intent: {user_prompt}" if user_prompt else ""}

Extract all details about the document format, fields, and how to validate similar documents."""
    
    try:
        print("[Knowledge Extraction] Analyzing reference image with Claude Vision...")
        
        result = client.chat_json_with_image(
            system=system_prompt,
            user_text=user_text,
            image_data=image_url,
            temperature=0
        )
        
        knowledge = result
        
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


def merge_knowledge_from_multiple_images(
    image_paths: List[str],
    user_prompt: str,
    per_image_contexts: List[str] = None
) -> Dict[str, Any]:
    """
    Extract knowledge from multiple reference images and merge into a single consolidated prompt.
    
    Handles:
    - Deduplication of repeated fields/rules across images
    - Resolution of contradictory information
    - Consolidation into a coherent validation prompt
    - Per-image specific context (e.g., "name is beside photo" vs "name is below photo")
    
    Args:
        image_paths: List of paths to reference images (up to 5)
        user_prompt: User's original validation description (centralized rules)
        per_image_contexts: Optional list of context/description for each image
                           (e.g., ["Front of PAN - name beside photo", "Back of PAN - name below photo"])
    
    Returns:
        Dict with:
        - success: bool
        - enhanced_prompt: Final merged prompt
        - extracted_knowledge: Consolidated knowledge from all images
        - per_image_knowledge: List of knowledge extracted from each image
        - contradictions_found: List of any contradictions detected and how they were resolved
        - document_type: Detected document type
        - unique_fields: List of all unique fields found
    """
    client = get_bedrock_client()
    
    # Ensure per_image_contexts is a list of same length as image_paths
    if per_image_contexts is None:
        per_image_contexts = [None] * len(image_paths)
    elif len(per_image_contexts) < len(image_paths):
        # Pad with None if not enough contexts provided
        per_image_contexts = per_image_contexts + [None] * (len(image_paths) - len(per_image_contexts))
    
    # Step 1: Extract knowledge from each image with its SPECIFIC context
    per_image_knowledge = []
    successful_extractions = []
    
    print(f"[Multi-Image Training] Processing {len(image_paths)} reference images...")
    
    for idx, image_path in enumerate(image_paths, start=1):
        print(f"[Multi-Image Training] Extracting knowledge from image {idx}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        # Build image-specific prompt with its own context
        image_specific_prompt = user_prompt
        image_context = per_image_contexts[idx - 1] if idx - 1 < len(per_image_contexts) else None
        
        if image_context:
            image_specific_prompt = f"{user_prompt}\n\n📌 SPECIFIC CONTEXT FOR THIS IMAGE:\n{image_context}"
            print(f"[Multi-Image Training] Image {idx} context: {image_context[:80]}...")
        
        result = extract_knowledge_from_reference_image(
            image_path_or_url=image_path,
            user_prompt=image_specific_prompt
        )
        
        per_image_knowledge.append({
            "image_index": idx,
            "image_path": os.path.basename(image_path),
            "image_context": image_context,  # Store the specific context used
            "success": result.get("success", False),
            "knowledge": result.get("extracted_knowledge", ""),
            "document_type": result.get("document_type", "Unknown"),
            "fields": result.get("fields", []),
            "validation_tips": result.get("validation_tips", []),
            "error": result.get("error")
        })
        
        if result.get("success"):
            successful_extractions.append({
                "index": idx,
                "image_context": image_context,  # Include context in successful extractions
                "knowledge": result.get("extracted_knowledge", ""),
                "document_type": result.get("document_type", "Unknown"),
                "fields": result.get("fields", []),
                "validation_tips": result.get("validation_tips", []),
                "raw_knowledge": result.get("raw_knowledge", {})
            })
            print(f"[Multi-Image Training] ✓ Image {idx}: Extracted {len(result.get('fields', []))} fields")
        else:
            print(f"[Multi-Image Training] ✗ Image {idx}: Failed - {result.get('error', 'Unknown error')}")
    
    # If no successful extractions, return original prompt
    if not successful_extractions:
        print("[Multi-Image Training] No successful extractions from any image")
        return {
            "success": False,
            "enhanced_prompt": user_prompt,
            "extracted_knowledge": None,
            "per_image_knowledge": per_image_knowledge,
            "contradictions_found": [],
            "document_type": "Unknown",
            "unique_fields": [],
            "error": "Failed to extract knowledge from any reference image"
        }
    
    # If only one successful extraction, use it directly (no merging needed)
    if len(successful_extractions) == 1:
        print("[Multi-Image Training] Only 1 successful extraction, using directly without merging")
        return {
            "success": True,
            "enhanced_prompt": f"{user_prompt}\n{successful_extractions[0]['knowledge']}",
            "extracted_knowledge": successful_extractions[0]['knowledge'],
            "per_image_knowledge": per_image_knowledge,
            "contradictions_found": [],
            "document_type": successful_extractions[0]['document_type'],
            "unique_fields": [f.get("name", "") for f in successful_extractions[0].get('fields', [])]
        }
    
    # Step 2: Merge knowledge from multiple images using LLM
    print(f"[Multi-Image Training] Merging knowledge from {len(successful_extractions)} images...")
    
    # Check if any user context was provided
    has_user_context = any(ext.get('image_context') for ext in successful_extractions)
    context_count = sum(1 for ext in successful_extractions if ext.get('image_context'))
    
    merge_prompt = f"""You are a document analysis expert. You have been given knowledge extracted from {len(successful_extractions)} reference images of the same document type.

Your task is to MERGE and CONSOLIDATE this knowledge into a SINGLE, coherent set of validation rules and field definitions.

═══════════════════════════════════════════════════════════════════════════════
                    ⚠️ CRITICAL PRIORITY WEIGHTING SYSTEM ⚠️
═══════════════════════════════════════════════════════════════════════════════

When building the final consolidated knowledge, follow this STRICT priority order:

{"**USER'S REFERENCE CONTEXT IS PROVIDED** - Apply these weights:" if has_user_context else "**NO USER CONTEXT PROVIDED** - Use 100% system analysis"}

{'''
📊 WEIGHT DISTRIBUTION:
┌─────────────────────────────────────────────────────────────────────────────┐
│  70% WEIGHT → USER'S REFERENCE CONTEXT (what user explicitly said)          │
│              - This is the PRIMARY source of truth                          │
│              - User's descriptions about each image are MOST IMPORTANT      │
│              - Example: "No field labels present" or "Field labels present" │
│              - Include these VERBATIM in layout variations                  │
│                                                                             │
│  30% WEIGHT → SYSTEM'S VISUAL ANALYSIS (auto-detected by AI)                │
│              - Secondary information only                                   │
│              - Positions, layouts detected automatically                    │
│              - Only use to SUPPLEMENT user context, not replace it          │
└─────────────────────────────────────────────────────────────────────────────┘

🚨 IMPORTANT RULES:
1. If user says "no field labels present" → This MUST appear in layout variations
2. If user says "field labels are present" → This MUST appear in layout variations  
3. User's exact words should be preserved in the output
4. System-detected positions are SECONDARY to user's context
5. When user context conflicts with system analysis, USER CONTEXT WINS
''' if has_user_context else '''
📊 WEIGHT DISTRIBUTION:
┌─────────────────────────────────────────────────────────────────────────────┐
│  100% WEIGHT → SYSTEM'S VISUAL ANALYSIS                                     │
│               - Since no user context provided, use all detected info       │
│               - Positions, layouts, field locations                         │
│               - Visual characteristics observed                             │
└─────────────────────────────────────────────────────────────────────────────┘
'''}

═══════════════════════════════════════════════════════════════════════════════
                         MERGING RULES
═══════════════════════════════════════════════════════════════════════════════

1. **DEDUPLICATION**: 
   - If the same field appears in multiple images, include it ONLY ONCE
   - Merge field descriptions - take the most complete/detailed version

2. **CONTRADICTION RESOLUTION**: 
   - If images have conflicting information:
     a) USER CONTEXT always takes priority over system analysis
     b) If both are user context, include BOTH as valid variations
   - Document ALL contradictions found with your resolution

3. **FIELD CONSOLIDATION**:
   - Same field with different names = ONE field
   - Merge validation rules for the same field

4. **VALIDATION RULES**:
   - Combine all unique validation rules from all images
   - Remove duplicate rules (same meaning, different wording)

5. **LAYOUT VARIATIONS** (MOST IMPORTANT):
   - Each image's USER CONTEXT defines a valid layout variation
   - Include user's EXACT descriptions as layout variant descriptions
   - Example: If user said "no field labels present" for Image 1:
     → Layout Variant 1: "No field labels present - values appear without labels like 'Name:', 'Father's Name:'"
   - Example: If user said "field labels are present" for Image 2:
     → Layout Variant 2: "Field labels are present - values appear with labels like 'Name:', 'Father's Name:'"

═══════════════════════════════════════════════════════════════════════════════
                    KNOWLEDGE FROM EACH REFERENCE IMAGE
═══════════════════════════════════════════════════════════════════════════════

"""
    
    for extraction in successful_extractions:
        # Include image-specific context PROMINENTLY if provided (70% weight)
        image_context_section = ""
        if extraction.get('image_context'):
            image_context_section = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  🎯 USER'S CONTEXT FOR THIS IMAGE (70% WEIGHT - PRIMARY SOURCE)               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  {extraction['image_context']}
╚═══════════════════════════════════════════════════════════════════════════════╝

⚠️ IMPORTANT: The above user context MUST be included in layout_variations!
   Use the user's EXACT description as the variant description.

"""
        else:
            image_context_section = """
[No user context provided for this image - use 100% system analysis]

"""
        
        merge_prompt += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMAGE {extraction['index']} - Document Type: {extraction['document_type']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{image_context_section}
SYSTEM'S VISUAL ANALYSIS (30% WEIGHT - Secondary):
- Fields Found: {json.dumps([f.get('name', 'Unknown') for f in extraction.get('fields', [])], ensure_ascii=False)}
- Validation Tips: {json.dumps(extraction.get('validation_tips', [])[:3], ensure_ascii=False)}

FULL KNOWLEDGE (for reference):
{extraction['knowledge'][:500]}...

"""
    
    merge_prompt += """
═══════════════════════════════════════════════════════════════════════════════
                              YOUR TASK
═══════════════════════════════════════════════════════════════════════════════

Analyze all the knowledge above and return a JSON object with:

{
    "document_type": "The detected document type (use most common across images)",
    "document_description": "Brief description of what this document is",
    
    "consolidated_fields": [
        {
            "name": "Canonical field name",
            "aliases": ["other names this field appeared as"],
            "format": "Expected format/pattern (most detailed version)",
            "validation_rule": "How to validate (merged from all images)",
            "location": "Where on document",
            "required": true/false,
            "appeared_in_images": [1, 2, 3]
        }
    ],
    
    "validation_rules": [
        {
            "rule": "Description of the validation rule",
            "source_images": [1, 2],
            "priority": "required" or "recommended" or "optional"
        }
    ],
    
    "contradictions": [
        {
            "field_or_rule": "What had conflicting info",
            "image_1_value": "Value from image 1",
            "image_2_value": "Value from image 2",
            "resolution": "Which value was chosen and why",
            "confidence": "high" / "medium" / "low"
        }
    ],
    
    "visual_characteristics": {
        "layout": "portrait/landscape (most common)",
        "key_elements": ["merged list of visual elements"],
        "security_features": ["merged list of security features"]
    },
    
    "layout_variations": [
        {
            "variant_name": "Descriptive name for this layout variant",
            "user_context": "EXACT text from user's context for this image (REQUIRED if provided)",
            "description": "Full description combining user context (70%) + system analysis (30%)",
            "source_image": 1,
            "distinguishing_features": ["Key features that identify this variant"],
            "has_field_labels": true/false,
            "notes": "Any additional notes about this variant"
        }
    ],
    
    "consolidated_knowledge_summary": "A comprehensive paragraph that MUST include: 1) All user-provided context about each image, 2) Layout variations with user's exact descriptions, 3) What makes each variant valid. The user's context should be prominently featured."
}

⚠️ CRITICAL REMINDER FOR layout_variations:
- If user provided context for an image, that context MUST appear in the corresponding layout_variation
- Use the user's EXACT words in the "user_context" and "description" fields
- Example: If user said "no field labels present", include: "user_context": "no field labels present"

Return ONLY valid JSON."""

    try:
        system_content = "You are an expert at analyzing and consolidating document validation rules from multiple reference images. Always return valid JSON. Be thorough in deduplication and contradiction resolution. IMPORTANT: Return ONLY valid JSON, no markdown or explanations."
        
        response = client.chat_completion(
            messages=[{"role": "user", "content": merge_prompt}],
            system=system_content,
            temperature=0
        )
        
        content = strip_json_code_fences(response)
        merged_result = json.loads(content)
        
        # Build the final consolidated knowledge string
        consolidated_knowledge = f"""
═══════════════════════════════════════════════════════════════════════════════
        CONSOLIDATED KNOWLEDGE FROM {len(successful_extractions)} REFERENCE IMAGES
═══════════════════════════════════════════════════════════════════════════════

Document Type: {merged_result.get('document_type', 'Unknown')}
Description: {merged_result.get('document_description', '')}

"""
        
        # Add consolidated fields
        consolidated_knowledge += "EXPECTED FIELDS:\n"
        unique_fields = []
        for field in merged_result.get("consolidated_fields", []):
            field_name = field.get("name", "Unknown")
            unique_fields.append(field_name)
            aliases = field.get("aliases", [])
            alias_str = f" (also known as: {', '.join(aliases)})" if aliases else ""
            required_str = "REQUIRED" if field.get("required", True) else "OPTIONAL"
            
            consolidated_knowledge += f"""
   • {field_name}{alias_str}
     Format: {field.get('format', 'N/A')}
     Validation: {field.get('validation_rule', 'N/A')}
     Location: {field.get('location', 'N/A')}
     Status: {required_str}
"""
        
        # Add validation rules
        validation_rules = merged_result.get("validation_rules", [])
        if validation_rules:
            consolidated_knowledge += "\nVALIDATION RULES:\n"
            for rule in validation_rules:
                priority = rule.get("priority", "required").upper()
                consolidated_knowledge += f"   • [{priority}] {rule.get('rule', '')}\n"
        
        # Add visual characteristics
        visual = merged_result.get("visual_characteristics", {})
        if visual:
            consolidated_knowledge += f"""
VISUAL CHARACTERISTICS:
   • Layout: {visual.get('layout', 'N/A')}
   • Key Elements: {', '.join(visual.get('key_elements', []))}
   • Security Features: {', '.join(visual.get('security_features', []))}
"""
        
        # Add layout variations if any - PRIORITIZE USER CONTEXT
        layout_variations = merged_result.get("layout_variations", [])
        if layout_variations:
            consolidated_knowledge += "\n" + "="*70 + "\n"
            consolidated_knowledge += "🎯 LAYOUT VARIATIONS (Accept ALL these layouts as valid):\n"
            consolidated_knowledge += "="*70 + "\n"
            for lv in layout_variations:
                variant_name = lv.get('variant_name', f"Variant from Image {lv.get('source_image', '?')}")
                user_context = lv.get('user_context', '')
                has_labels = lv.get('has_field_labels')
                
                consolidated_knowledge += f"""
   📋 {variant_name} (from Image {lv.get('source_image', '?')}):
"""
                # Prominently display user context if available (70% weight)
                if user_context:
                    consolidated_knowledge += f"""      🎯 USER'S DESCRIPTION: "{user_context}"
"""
                
                # Add field labels info if available
                if has_labels is not None:
                    label_text = "Field labels ARE present (e.g., 'Name:', 'Father's Name:')" if has_labels else "Field labels are NOT present (values appear without labels)"
                    consolidated_knowledge += f"""      📝 Field Labels: {label_text}
"""
                
                consolidated_knowledge += f"""      📍 Description: {lv.get('description', 'N/A')}
      ✓ Distinguishing features: {', '.join(lv.get('distinguishing_features', []))}
"""
                if lv.get('notes'):
                    consolidated_knowledge += f"""      📌 Notes: {lv.get('notes')}
"""
        
        # Add contradiction notes if any
        contradictions = merged_result.get("contradictions", [])
        if contradictions:
            consolidated_knowledge += "\nNOTES - RESOLVED CONTRADICTIONS:\n"
            for c in contradictions:
                consolidated_knowledge += f"""   • {c.get('field_or_rule', 'Unknown')}:
     - Image values: "{c.get('image_1_value', '')}" vs "{c.get('image_2_value', '')}"
     - Resolution: {c.get('resolution', 'N/A')}
     - Confidence: {c.get('confidence', 'medium')}
"""
        
        # Add summary
        consolidated_knowledge += f"""
SUMMARY:
{merged_result.get('consolidated_knowledge_summary', '')}

═══════════════════════════════════════════════════════════════════════════════
"""
        
        # Build final enhanced prompt with 60/40 weighting
        # 60% weight to main description (user's validation rules)
        # 40% weight to reference image learning
        enhanced_prompt = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    📋 MAIN VALIDATION RULES (60% WEIGHT)                      ║
║                    These are the PRIMARY validation criteria                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝

{user_prompt}

╔═══════════════════════════════════════════════════════════════════════════════╗
║              📚 REFERENCE IMAGE KNOWLEDGE (40% WEIGHT)                        ║
║              Use this to understand document structure & variations           ║
╚═══════════════════════════════════════════════════════════════════════════════╝

{consolidated_knowledge}

╔═══════════════════════════════════════════════════════════════════════════════╗
║                         ⚠️ VALIDATION PRIORITY                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
1. FIRST: Check all conditions in MAIN VALIDATION RULES (60% weight)
2. THEN: Use REFERENCE IMAGE KNOWLEDGE to understand valid document variations
3. Accept documents that match ANY of the layout variations described above
4. If a document matches a layout variation, it should NOT be failed for layout differences
"""
        
        print(f"[Multi-Image Training] ✓ Successfully merged knowledge from {len(successful_extractions)} images")
        print(f"[Multi-Image Training] Found {len(unique_fields)} unique fields")
        print(f"[Multi-Image Training] Resolved {len(contradictions)} contradictions")
        
        return {
            "success": True,
            "enhanced_prompt": enhanced_prompt,
            "extracted_knowledge": consolidated_knowledge,
            "per_image_knowledge": per_image_knowledge,
            "contradictions_found": contradictions,
            "document_type": merged_result.get("document_type", "Unknown"),
            "unique_fields": unique_fields,
            "consolidated_fields": merged_result.get("consolidated_fields", []),
            "validation_rules": merged_result.get("validation_rules", [])
        }
        
    except Exception as e:
        print(f"[Multi-Image Training] Error merging knowledge: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: concatenate all knowledge without intelligent merging
        print("[Multi-Image Training] Falling back to simple concatenation...")
        fallback_knowledge = f"""
═══════════════════════════════════════════════════════════════════════════════
        KNOWLEDGE FROM {len(successful_extractions)} REFERENCE IMAGES (NOT MERGED)
═══════════════════════════════════════════════════════════════════════════════
"""
        
        all_fields = []
        for extraction in successful_extractions:
            fallback_knowledge += f"""
━━━ From Image {extraction['index']} ({extraction['document_type']}) ━━━
{extraction['knowledge']}
"""
            all_fields.extend([f.get("name", "") for f in extraction.get('fields', [])])
        
        # Deduplicate field names at least
        unique_fields = list(set(all_fields))
        
        return {
            "success": True,
            "enhanced_prompt": f"{user_prompt}\n{fallback_knowledge}",
            "extracted_knowledge": fallback_knowledge,
            "per_image_knowledge": per_image_knowledge,
            "contradictions_found": [],
            "document_type": successful_extractions[0]['document_type'],
            "unique_fields": unique_fields,
            "merge_warning": f"Intelligent merge failed ({str(e)}), using concatenated knowledge"
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
    client = get_bedrock_client()
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
    
    # Build user text with OCR data
    user_text = f"""Analyze this document and follow my instructions:

{user_prompt}

REMEMBER:
1. Extract ONLY the fields I mentioned (if I specified any)
2. Validate EACH condition I mentioned separately
3. Calculate score = (passed conditions / total conditions) × 100
4. Answer my questions in user_questions section with "Q:" and "→ A:" format
5. Explain why score is what it is in score_explanation
6. Show actual values vs required values in pass/fail conditions"""

    if ocr_text:
        user_text += f"\n\nOCR Data (use for reference):\n{json.dumps(ocr_text, indent=2, ensure_ascii=False)[:3000]}"
    
    try:
        print(f"[Generic] Processing with Claude Vision...")
        
        # Use first image for vision analysis (Claude handles one image at a time)
        # For multi-page documents, we concatenate OCR text
        image_url = image_urls[0] if image_urls else None
        
        if image_url:
            result = client.chat_json_with_image(
                system=system_prompt + "\n\nIMPORTANT: Return ONLY valid JSON, no markdown or explanations.",
                user_text=user_text,
                image_data=image_url,
                temperature=0
            )
        else:
            # No image, use text-only
            response = client.chat_completion(
                messages=[{"role": "user", "content": user_text}],
                system=system_prompt + "\n\nIMPORTANT: Return ONLY valid JSON, no markdown or explanations.",
                temperature=0
            )
            result = json.loads(strip_json_code_fences(response))
        
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
    2. Sends to Claude with user's validation prompt
    3. Returns extraction and validation results
    
    Args:
        ocr_text: Plain text extracted by Textract
        user_prompt: User's validation rules and extraction instructions
        textract_blocks: Raw Textract blocks for additional context
    
    Returns:
        Dict with status, score, doc_extracted_json, reason
    """
    client = get_bedrock_client()
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
        print(f"[LLM] Sending to Claude for validation...")
        
        response = client.chat_completion(
            messages=[{"role": "user", "content": user_message}],
            system=system_prompt + "\n\nIMPORTANT: Return ONLY valid JSON, no markdown or explanations.",
            temperature=0
        )
        
        content = strip_json_code_fences(response)
        result = json.loads(content)
        
        # ===== SCORE CORRECTION LOGIC =====
        # Recalculate score from actual pass/fail conditions to fix LLM miscalculations
        if isinstance(result.get("reason"), dict):
            pass_conditions = result["reason"].get("pass_conditions", [])
            fail_conditions = result["reason"].get("fail_conditions", [])
            pass_count = len(pass_conditions)
            fail_count = len(fail_conditions)
            total_conditions = pass_count + fail_count
            
            if total_conditions > 0:
                calculated_score = round((pass_count / total_conditions) * 100)
                llm_score = int(result.get("score", 0))
                
                # If LLM score doesn't match calculated score, use calculated score
                if calculated_score != llm_score:
                    print(f"[LLM] Score mismatch detected: LLM said {llm_score}%, but {pass_count}/{total_conditions} conditions passed = {calculated_score}%")
                    print(f"[LLM] Correcting score from {llm_score}% to {calculated_score}%")
                    result["score"] = calculated_score
                    result["reason"]["score_explanation"] = f"{pass_count} out of {total_conditions} conditions passed = {calculated_score}% score"
            
            # Ensure status matches score logic
            if result["score"] == 100 and fail_count == 0:
                result["status"] = "pass"
            elif result["score"] < 100 or fail_count > 0:
                result["status"] = "fail"
        # ===== END SCORE CORRECTION =====
        
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
