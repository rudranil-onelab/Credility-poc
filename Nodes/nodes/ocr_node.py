"""
OCR node for document processing and data extraction.
"""

import os
import json
from typing import Dict, Any, Optional, List

from ..config.state_models import PipelineState, OCRState
from ..config.settings import OCR_MODE, DOC_CATEGORY
from ..tools.aws_services import run_textract_async_s3, run_analyze_id_s3, run_textract_local_file
from ..tools.ocr_processing import (
    group_blocks_by_page, resolve_kv_pairs_from_page_blocks,
    cells_from_page_blocks, lines_words_from_page_blocks,
    route_document_type_from_ocr, analyze_id_to_kvs
)
from ..tools.llm_services import chat_json, classify_via_image, extract_via_image, remove_raw_text_fields
from ..utils.helpers import get_filename_without_extension, log_agent_event


# Document type prompts
PROMPTS_BY_TYPE: Dict[str, str] = {
    "bank_statement": (
        "You are a financial document parser for BANK STATEMENTS. "
        "Input JSON contains arrays: 'lines', 'cells', and 'kvs'. "
        "Return a single JSON object with anything present (do not limit to a fixed field list). "
        "Prefer structured keys where obvious (e.g., statement_period, balances, transactions). "
        "Additionally, set a top-level field 'document_name' to the exact document title/name as mentioned in the document itself when present; otherwise omit. "
        "Rules: Preserve text exactly; do not normalize; include a 'key_values' array if you see labeled pairs; "
        "omit fields not present. Output a single JSON object only."
    ),
    "identity": (
        "You parse identity documents (U.S. driver license/state ID/passport plus INDIAN IDs like Aadhaar, PAN, Voter ID, Indian Passport, Driving Licence, GST certificate). "
        "Input JSON has 'lines', 'cells', and 'kvs'. "
        "Extract ALL information present. "
        "Return ONE JSON object where each field name is a JSON key and the field's value is the JSON value. "
        "\n\nCRITICAL NAME EXTRACTION RULES:\n"
        "- ALWAYS extract names in the correct order: firstName, middleName (if present), lastName\n"
        "- firstName: The person's given/first name (e.g., 'DENNIS' in 'DENNIS DEAN MCKINLEY')\n"
        "- middleName: The person's middle name or initial (e.g., 'DEAN' in 'DENNIS DEAN MCKINLEY')\n"
        "- lastName: The person's family/surname (e.g., 'MCKINLEY' in 'DENNIS DEAN MCKINLEY')\n"
        "- If the document shows 'NAME: DENNIS DEAN MCKINLEY', extract as: {\"firstName\":\"DENNIS\", \"middleName\":\"DEAN\", \"lastName\":\"MCKINLEY\"}\n"
        "- If only two names appear (e.g., 'DENNIS MCKINLEY'), extract as: {\"firstName\":\"DENNIS\", \"lastName\":\"MCKINLEY\"} (omit middleName)\n"
        "- If the document has separate fields like 'FN:', 'MN:', 'LN:' or 'Given Name', 'Surname', use those to determine the correct mapping\n"
        "- NEVER swap firstName and lastName\n"
        "- NEVER put middleName in lastName position\n"
        "\n\nINDIAN DOCUMENT FIELD RULES:\n"
        "- Aadhaar: capture aadhaarNumber (12 digit), yearOfBirth/dateOfBirth, gender, address lines, fatherName/motherName/husbandName, and VID if printed.\n"
        "- PAN: capture panNumber (AAAAA9999A format), holderName, fatherName, motherName, dateOfBirth, signature, photo reference, QR data if shown.\n"
        "- Voter ID (EPIC): capture epicNumber, serialNumber, partNumber, assemblyConstituency, relationType (e.g., Father/Husband), gender, address, age/dateOfBirth.\n"
        "- Indian Passport: capture passportNumber, surname/givenNames, nationality, dateOfBirth, placeOfBirth, issueDate, expirationDate, fileNumber, issuingAuthority.\n"
        "- Indian Driving Licence: capture licenseNumber, issuingRTO, issueDate, validTillNonTransport, validTillTransport, address, bloodGroup, vehicle classes (COV).\n"
        "- GST/other Indian IDs: capture gstin, tradeName, legalName, constitutionOfBusiness, registration dates, addresses.\n"
        "- Always include both English and Hindi (or other script) text exactly as printed; do NOT translate or drop Hindi lines. Example: \"पिता/Father's Name\" should keep the Hindi text.\n"
        "- Map bilingual labels to the canonical key but preserve the exact value string, including Devanagari characters.\n"
        "- Recognize field labels like \"UIDAI\", \"Unique Identification Authority of India\", \"EPIC No\", \"GSTIN\", \"DL No\", \"PAN\", \"VID\".\n"
        "\n\nMULTILINGUAL & FORMAT RULES:\n"
        "- Documents may contain Hindi + English. Capture BOTH scripts verbatim in the value. If a field shows two scripts, concatenate them in one string separated by a space or newline.\n"
        "- Preserve numerals exactly as printed (Aadhaar spacing, PAN capitalization, passport prefixes, DL formatting).\n"
        "- If a field appears with only a symbol (like a heart icon for Organ Donor), include that symbol as its value.\n"
        "- Do not wrap fields as {key:..., value:...} objects — instead use plain JSON key:value pairs.\n"
        "- Do not invent fields; only include what is clearly present.\n"
        "- Additionally, set a top-level field 'document_name' to the exact document title/name as mentioned in the document itself when present; otherwise omit.\n"
        "- Output JSON only, no prose."
    ),
    "property": (
        "You are an exhaustive parser for PROPERTY-related documents (appraisals, deeds, plats, surveys, covenants, tax records). "
        "Input has 'lines', 'cells', and 'kvs'. Extract EVERYTHING present. Return ONE JSON object only.\n\n"
        "Sections (include when present):\n"
        "  title_block: address, parcel/APN, legal description, borrower/owner/lender, zoning, tax info, subdivision, district/county/state, dates, registry numbers.\n"
        "  valuation: approaches (cost/sales/income), opinion of value, effective date, exposure/marketing time, reconciliations.\n"
        "  site: lot size, utilities, zoning compliance, easements, hazards, topography, influences.\n"
        "  improvements: year built, style, condition (C-ratings), renovations, construction details (foundation, roof, HVAC, windows, floors), amenities (garages, decks, fireplaces).\n"
        "  sales_history: full chain with dates, prices, document types, grantors/grantees, book/page.\n"
        "  comparables: reconstruct comparable tables into arrays with adjustments, net/gross, distances, remarks.\n"
        "  key_values: all labeled pairs as {key, value}.\n"
        "  approvals: signers, roles, license numbers, expirations, certifications, supervisory details.\n"
        "  maps_legends: captions, scales, legends, directional notes.\n"
        "  notes: disclaimers, limiting conditions, free text not captured elsewhere.\n\n"
        "Additionally, set a top-level field 'document_name' to the exact document title/name as mentioned in the document itself when present; otherwise omit.\n"
        "Rules: Preserve text EXACTLY; do not normalize; reconstruct tables; include checkboxes and symbols as-is; no prose."
    ),
    "entity": (
        "You are an exhaustive parser for ENTITY/BUSINESS documents (formation, amendments, certificates, annual reports). "
        "Input has 'lines', 'cells', and 'kvs'. Extract EVERYTHING. Return ONE JSON object only.\n\n"
        "Sections (include when present):\n"
        "  header: document titles, form names/codes, jurisdiction, filing office.\n"
        "  entity_profile: legal name(s), prior names/DBAs, entity type/class, jurisdiction of organization, domestication/foreign registration details, formation date, duration.\n"
        "  identifiers: EIN, state ID, SOS#, file#, control#, NAICS, DUNS.\n"
        "  registered_agent: name, ID, addresses, consent statements.\n"
        "  addresses: principal office, mailing, records office (each as full exact text).\n"
        "  management: organizers, incorporators, members/managers, directors, officers (names, roles, addresses, terms).\n"
        "  ownership_capital: shares/units/classes, par value, authorized/issued, ownership table (reconstruct from cells).\n"
        "  purpose_powers: stated purpose, limitations, special provisions.\n"
        "  compliance: annual reports, franchise tax, effective dates, delayed effectiveness.\n"
        "  approvals: signatures, seals, notary blocks, certifications, filing acknowledgments, dates/times.\n"
        "  key_values: every labeled pair as {key, value}.\n"
        "  tables: any tables reconstructed from 'cells'.\n"
        "  notes: free text not captured elsewhere.\n\n"
        "Additionally, set a top-level field 'document_name' to the exact document title/name as mentioned in the document itself when present; otherwise omit.\n"
        "Rules: Preserve text exactly; reconstruct tables; include checkboxes/symbols; no prose."
    ),
    "loan": (
        "You are an exhaustive parser for LOAN documents (notes, disclosures, deeds of trust, LE/CD, riders). "
        "Input has 'lines', 'cells', and 'kvs'. Extract EVERYTHING. Return ONE JSON object only.\n\n"
        "Sections (include when present):\n"
        "  parties: borrower(s), lender, trustee, servicer, MERS, guarantors (names, addresses).\n"
        "  loan_terms: principal, interest rate, APR/APY, rate type, index/margin, caps, payment schedule, maturity, amortization, prepayment, late fees, escrow, balloon, ARM disclosures.\n"
        "  collateral: property address/legal, lien position, riders/addenda.\n"
        "  fees_costs: itemized fees, finance charges, points, credits (reconstruct tables).\n"
        "  disclosures: TILA/RESPA sections, right to cancel, servicing transfer, privacy, HMDA, ECOA.\n"
        "  compliance_numbers: loan #, application #, NMLS IDs, case numbers, MIC/endorsements.\n"
        "  signatures_notary: signature lines, notary acknowledgments, seals, dates/times.\n"
        "  key_values: every labeled pair as {key, value}.\n"
        "  tables: payment schedules, fee tables, escrow analyses reconstructed from 'cells'.\n"
        "  notes: any free text not captured elsewhere.\n\n"
        "Additionally, set a top-level field 'document_name' to the exact document title/name as mentioned in the document itself when present; otherwise omit.\n"
        "Rules: Preserve text exactly; reconstruct tables; include checkboxes/symbols; no prose."
    ),
    "unknown": (
        "You are a cautious yet exhaustive parser for UNKNOWN document types. "
        "Input has 'lines', 'cells', and 'kvs'. Extract EVERYTHING visible without guessing meaning. Return ONE JSON object only.\n\n"
        "Output shape MUST match prior expectations:\n"
        "  key_values: array of {key, value} for any labeled pairs you can see.\n"
        "  free_text: ordered array of textual lines exactly as shown.\n"
        "Additionally (when present):\n"
        "  tables: reconstructed tables from 'cells' (array of row objects).\n"
        "  checkmarks: array of {label, status} for selection elements with 'SELECTED' or 'NOT_SELECTED'.\n"
        "  notes: any content that is ambiguous or uncategorized.\n\n"
        "Additionally, set a top-level field 'document_name' to the exact document title/name as mentioned in the document itself when present; otherwise omit.\n"
        "Rules: Preserve text exactly; do not normalize; no prose."
    ),
}


def llm_extract_page(doc_type: str, page_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract data from page using LLM."""
    llm_input = {
        "lines": page_data.get("lines", []),
        "cells": page_data.get("cells", []),
        "kvs":   page_data.get("kvs", []),
    }
    system = PROMPTS_BY_TYPE.get(doc_type, PROMPTS_BY_TYPE["unknown"])
    out = chat_json("gpt-4o", system, llm_input) or {}
    return remove_raw_text_fields(out)


def convert_pdf_to_image_url(local_path: str) -> str:
    """
    Convert first page of PDF to base64 image URL for Vision API.
    
    Returns:
        Base64 data URI of the first page as PNG image
    """
    import base64
    import io
    
    try:
        # Try PyMuPDF first (faster, no external deps)
        import fitz  # PyMuPDF
        
        doc = fitz.open(local_path)
        page = doc.load_page(0)  # First page
        
        # Render at 150 DPI for good quality
        mat = fitz.Matrix(150/72, 150/72)
        pix = page.get_pixmap(matrix=mat)
        
        img_bytes = pix.tobytes("png")
        doc.close()
        
        base64_data = base64.b64encode(img_bytes).decode('utf-8')
        print(f"[PDF] Converted PDF to image using PyMuPDF")
        return f"data:image/png;base64,{base64_data}"
        
    except ImportError:
        print("[PDF] PyMuPDF not available, trying pdf2image...")
        
        try:
            from pdf2image import convert_from_path
            
            pages = convert_from_path(local_path, dpi=150, first_page=1, last_page=1)
            if pages:
                buffer = io.BytesIO()
                pages[0].save(buffer, format='PNG')
                img_bytes = buffer.getvalue()
                base64_data = base64.b64encode(img_bytes).decode('utf-8')
                print(f"[PDF] Converted PDF to image using pdf2image")
                return f"data:image/png;base64,{base64_data}"
            else:
                raise RuntimeError("pdf2image returned no pages")
                
        except ImportError:
            raise ImportError("Please install PyMuPDF (pip install pymupdf) for PDF support")


def run_pipeline_local(local_path: str, mode: str = "ocr+llm") -> Dict[str, Any]:
    """
    Run OCR pipeline on a local file without S3.
    
    For PDFs with ocr+llm mode: Uses AWS Textract (faster & more accurate)
    For images: Uses LLM vision API directly
    """
    import base64
    from pathlib import Path
    
    print(f"[OCR LOCAL] Processing file: {local_path}")
    print(f"[OCR LOCAL] Mode: {mode}")
    
    # Determine file type
    file_ext = Path(local_path).suffix.lower()
    print(f"[OCR LOCAL] File type: {file_ext}")
    
    # For PDFs with ocr+llm mode, use Textract (faster & more accurate)
    if file_ext == '.pdf' and mode == "ocr+llm":
        print("[OCR LOCAL] PDF with OCR+LLM mode - using AWS Textract...")
        return run_pipeline_local_textract(local_path, mode)
    
    # For images or LLM-only mode, use Vision API
    if file_ext == '.pdf':
        # PDF with LLM-only mode - convert to image
        print("[OCR LOCAL] PDF with LLM mode - converting to image for Vision API...")
        try:
            image_url = convert_pdf_to_image_url(local_path)
        except ImportError as e:
            print(f"[OCR LOCAL ERROR] {e}")
            raise RuntimeError(f"PDF processing requires PyMuPDF: pip install pymupdf")
        except Exception as e:
            print(f"[OCR LOCAL ERROR] Failed to convert PDF: {e}")
            raise RuntimeError(f"Failed to convert PDF to image: {e}")
    else:
        # Read image file and encode to base64
        with open(local_path, "rb") as f:
            file_bytes = f.read()
        
        base64_image = base64.b64encode(file_bytes).decode('utf-8')
        
        if file_ext in ['.jpg', '.jpeg']:
            mime_type = 'image/jpeg'
        elif file_ext == '.png':
            mime_type = 'image/png'
        elif file_ext == '.gif':
            mime_type = 'image/gif'
        elif file_ext == '.webp':
            mime_type = 'image/webp'
        else:
            mime_type = 'image/jpeg'  # Default to JPEG
        
        image_url = f"data:{mime_type};base64,{base64_image}"
    
    # Use LLM vision for classification and extraction
    try:
        # Step 1: Classify document type
        print("[OCR LOCAL] Step 1: Classifying document type...")
        doc_type = classify_via_image("gpt-4o", image_url)
        print(f"[OCR LOCAL] Detected document type: {doc_type}")
        
        # Step 2: Extract data based on document type
        print("[OCR LOCAL] Step 2: Extracting structured data...")
        
        # Use the extract_via_image function with correct parameters
        structured_data = extract_via_image("gpt-4o", doc_type, image_url, PROMPTS_BY_TYPE)
        
        # Add document type to structured data
        structured_data["doc_type"] = doc_type
        structured_data["mode"] = mode
        
        print(f"[OCR LOCAL] Extraction complete. Fields extracted: {len(structured_data)}")
        
        return {
            "doc_type": doc_type,
            "mode": mode,
            "result_path": "local",
            "name_no_ext": Path(local_path).stem,
            "structured": structured_data,
            "ocr_text": ""
        }
    
    except Exception as e:
        print(f"[OCR LOCAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Local OCR processing failed: {e}")


def run_pipeline_local_textract(local_path: str, mode: str = "ocr+llm") -> Dict[str, Any]:
    """
    Run OCR pipeline on a local PDF using AWS Textract.
    
    This is faster and more accurate than Vision API for PDFs.
    """
    from pathlib import Path
    
    print(f"[OCR LOCAL TEXTRACT] Processing PDF: {local_path}")
    
    try:
        # Run Textract on local file
        raw = run_textract_local_file(local_path)
        
        # Build per-page simplified view
        blocks = raw.get("blocks", [])
        pages_full = group_blocks_by_page(blocks)
        
        simplified = {"pages": {}}
        for page_num, page_blocks in pages_full.items():
            lw = lines_words_from_page_blocks(page_blocks)
            cells = cells_from_page_blocks(page_blocks)
            kvs = resolve_kv_pairs_from_page_blocks(page_blocks)
            
            simplified["pages"][page_num] = {
                "lines": lw["lines"],
                "words": lw["words"],
                "cells": cells,
                "kvs": kvs,
            }
        
        # Route document type from OCR
        doc_type = route_document_type_from_ocr(simplified)
        print(f"[OCR LOCAL TEXTRACT] Document type: {doc_type}")
        
        # Extract page-by-page via LLM
        all_structured: Dict[str, Any] = {"doc_type": doc_type}
        doc_name_candidates: List[str] = []
        
        for page_num in sorted(simplified["pages"].keys(), key=lambda x: int(x)):
            page_data = simplified["pages"][page_num]
            print(f"[OCR LOCAL TEXTRACT] Extracting page {page_num} as '{doc_type}'...")
            extracted = llm_extract_page(doc_type, page_data)
            
            # Hoist document_name
            if isinstance(extracted, dict):
                dn = extracted.get("document_name")
                if dn:
                    doc_name_candidates.append(dn)
                    extracted.pop("document_name", None)
            
            all_structured[str(page_num)] = extracted
        
        # Set top-level document_name
        if doc_name_candidates:
            all_structured["document_name"] = doc_name_candidates[0]
        
        print(f"[OCR LOCAL TEXTRACT] Extraction complete. Pages processed: {len(simplified['pages'])}")
        
        return {
            "doc_type": doc_type,
            "mode": mode,
            "result_path": "local",
            "name_no_ext": Path(local_path).stem,
            "structured": all_structured,
            "ocr_text": ""
        }
        
    except Exception as e:
        print(f"[OCR LOCAL TEXTRACT ERROR] {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Textract processing failed: {e}")


def run_pipeline(bucket: str, key: str, mode: str = "ocr+llm") -> Dict[str, Any]:
    """
    Execute the OCR extraction pipeline.
    """
    simplified: Dict[str, Any] = {"pages": {}}
    name_no_ext = get_filename_without_extension(key)

    if mode == "ocr+llm":
        # 1) Textract (Tables + Forms)
        raw = run_textract_async_s3(bucket, key)

        # 2) Build per-page simplified view FROM ORIGINAL BLOCKS so relationships remain available
        blocks = raw.get("blocks", [])
        pages_full = group_blocks_by_page(blocks)

        simplified = {"pages": {}}
        for page_num, page_blocks in pages_full.items():
            lw = lines_words_from_page_blocks(page_blocks)
            cells = cells_from_page_blocks(page_blocks)
            kvs = resolve_kv_pairs_from_page_blocks(page_blocks)

            simplified["pages"][page_num] = {
                "lines": lw["lines"],
                "words": lw["words"],
                "cells": cells,
                "kvs": kvs,
            }

        # 3) Route document type from OCR ONLY (page 1)
        doc_type = route_document_type_from_ocr(simplified)
        print(f"[Router] Document type: {doc_type}")

        # 4) OPTIONAL: If identity, try AnalyzeID and MERGE KVs into page 1
        if doc_type == "identity":
            try:
                aid = run_analyze_id_s3(bucket, key)
                aid_kvs = analyze_id_to_kvs(aid)
                first_page_key = sorted(simplified["pages"].keys(), key=lambda x: int(x))[0]
                simplified["pages"][first_page_key].setdefault("kvs", [])
                simplified["pages"][first_page_key]["kvs"].extend(aid_kvs)
            except Exception as e:
                print(f"[AnalyzeID] skipped or failed: {e}")

        image_extracted = {}
    else:  # LLM-only mode
        lower_key = key.lower()
        if lower_key.endswith(".pdf"):
            raw = run_textract_async_s3(bucket, key)
            blocks = raw.get("blocks", [])
            pages_full = group_blocks_by_page(blocks)
            simplified = {"pages": {}}
            for page_num, page_blocks in pages_full.items():
                lw = lines_words_from_page_blocks(page_blocks)
                cells = cells_from_page_blocks(page_blocks)
                kvs = resolve_kv_pairs_from_page_blocks(page_blocks)
                simplified["pages"][page_num] = {
                    "lines": lw["lines"],
                    "words": lw["words"],
                    "cells": cells,
                    "kvs": kvs,
                }
            doc_type = route_document_type_from_ocr(simplified)
            print(f"[Router] Document type: {doc_type}")
            image_extracted = {}
        else:
            # Handle image files with LLM vision
            from ..tools.aws_services import generate_presigned_url
            image_url = generate_presigned_url(bucket, key)
            
            if image_url:
                doc_type = classify_via_image("gpt-4o", image_url)
                print(f"[Router] Document type: {doc_type}")
                image_extracted = extract_via_image("gpt-4o", doc_type, image_url, PROMPTS_BY_TYPE)
                simplified = {"pages": {1: {"lines": [], "words": [], "cells": [], "kvs": []}}}
            else:
                doc_type = "unknown"
                simplified = {"pages": {1: {"lines": [], "words": [], "cells": [], "kvs": []}}}
                image_extracted = {}

    # 5) Extract page-by-page via LLM (works for both modes) + HOIST document_name
    all_structured: Dict[str, Any] = {"doc_type": doc_type}
    top_level_doc_name: Optional[str] = None
    doc_name_candidates: List[str] = []

    for page_num in sorted(simplified["pages"].keys(), key=lambda x: int(x)):
        page_data = simplified["pages"][page_num]
        print(f"[LLM] Extracting page {page_num} as '{doc_type}'...")
        if mode == "llm" and page_num == 1 and image_extracted:
            extracted = image_extracted
        else:
            extracted = llm_extract_page(doc_type, page_data)

        # Hoist `document_name` out of page result (first non-empty wins)
        if isinstance(extracted, dict):
            dn = extracted.get("document_name")
            if dn:
                doc_name_candidates.append(dn)
                extracted.pop("document_name", None)

        all_structured[str(page_num)] = extracted

    # Decide on a single top-level document_name
    if doc_name_candidates:
        top_level_doc_name = doc_name_candidates[0]
        all_structured["document_name"] = top_level_doc_name

    # Do not save OCR outputs locally or to S3 as per requirement
    return {
        "doc_type": doc_type,
        "structured": all_structured,
        "name_no_ext": name_no_ext,
        "mode": mode,
    }


def OCR(state: PipelineState) -> PipelineState:
    """
    Process document through OCR pipeline.
    """
    if state.ingestion is None:
        raise ValueError("Ingestion state missing; run Ingestion node first.")

    # Prefer doc category from ingestion document_type
    doc_category = (state.ingestion.document_type or "").strip()
    if not doc_category:
        # Fallback to env for manual runs
        doc_category = DOC_CATEGORY

    bucket = state.ingestion.s3_bucket
    key = state.ingestion.s3_key
    # Use mode from ingestion state if available, otherwise use OCR_MODE
    mode = state.ingestion.tool or OCR_MODE

    if not bucket or not key:
        raise ValueError("Missing S3 bucket/key from ingestion.")

    print(f"\n=== PROCEEDING WITH DOCUMENT PROCESSING ===")
    print(f"[OCR] Mode: {mode}")
    log_agent_event(state, "OCR", "start")

    # Check if this is a local file (no S3)
    if bucket == "local":
        print(f"[OCR] Processing local file directly: {key}")
        result = run_pipeline_local(key, mode)
    else:
        # Run OCR pipeline with the requested mode
        result = run_pipeline(bucket, key, mode)
    structured = result.get("structured", {})
    
    # Check if OCR failed to extract meaningful data
    ocr_extraction_failed = False
    error_message = None
    
    # Skip validation for local processing (it returns flat structure, not pages)
    if bucket != "local":
        # First, check if structured data is empty or has minimal content
        if not structured or len(structured) <= 1:  # Only has doc_type
            ocr_extraction_failed = True
            error_message = "Please upload a valid document with good quality. OCR was unable to extract any data from the document."
        else:
            # Check if any page has meaningful content
            has_content = False
            total_fields = 0
            non_empty_fields = 0
            
            for page_key, page_data in structured.items():
                if page_key in ["doc_type", "document_name", "mode"]:
                    continue
                if isinstance(page_data, dict):
                    # Check if page has any meaningful fields (more than just doc_type)
                    if len(page_data) > 0:
                        total_fields += len(page_data)
                        # Check if there are actual field values (not just empty strings)
                        for field_value in page_data.values():
                            if field_value and str(field_value).strip():
                                non_empty_fields += 1
                                has_content = True
            
            # If no content found or very few fields extracted, consider it a failure
            if not has_content or (total_fields > 0 and non_empty_fields < 2):
                ocr_extraction_failed = True
                error_message = "Please upload a valid document with good quality. OCR was unable to extract meaningful data from the document."
    else:
        # For local processing, just check if we have any fields besides metadata
        non_meta_fields = {k: v for k, v in structured.items() if k not in ["doc_type", "document_name", "mode"]}
        if not non_meta_fields or len(non_meta_fields) == 0:
            ocr_extraction_failed = True
            error_message = "Please upload a valid document with good quality. OCR was unable to extract any data from the document."
        else:
            print(f"[OCR LOCAL] Successfully extracted {len(non_meta_fields)} fields from document")
    
    # If OCR extraction failed, store error in structured data for downstream processing
    if ocr_extraction_failed:
        structured["ocr_extraction_failed"] = True
        structured["ocr_error_message"] = error_message
        print(f"[ERROR] OCR extraction failed: {error_message}")
        log_agent_event(state, "OCR", "failed", {"error": error_message})

    # Populate OCR state
    state.ocr = OCRState(
        bucket=bucket,
        key=key,
        mode=mode,
        doc_category=doc_category,
        document_name=structured.get("document_name"),
        ocr_json=structured,
    )
    log_agent_event(state, "OCR", "completed", {"doc_type": result.get("doc_type")})
    return state
