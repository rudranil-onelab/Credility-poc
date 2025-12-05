"""
Classification node for document type validation.
INDIAN DOCUMENTS FIRST with multi-language support (22+ languages).
Adds strict mismatch check between provided document_name and detected identity subtype.
"""

import os
import re
from typing import Optional, Dict, Any

from ..config.state_models import PipelineState, ClassificationState
from ..utils.helpers import log_agent_event
from ..utils.language_utils import (
    detect_languages, detect_scripts, detect_document_by_keywords,
    MULTILINGUAL_KEYWORDS, LANGUAGE_CODES
)


def _map_display_name_to_identity_subtype(name: Optional[str]) -> str:
    """
    Map a human document display name (from metadata/DB) to a normalized
    identity subtype. INDIAN DOCUMENTS ARE CHECKED FIRST.
    Supports multi-language document names (Hindi, Tamil, Telugu, etc.)
    """
    n = (name or "").strip().lower()
    if not n:
        return "other_identity"

    # ==========================================
    # INDIAN IDENTITY DOCUMENTS (Check First)
    # ==========================================
    
    # Aadhaar Card (आधार कार्ड)
    if "aadhaar" in n or "aadhar" in n or "आधार" in n or "uidai" in n:
        return "aadhaar"
    
    # PAN Card (पैन कार्ड)
    if ("pan" in n and ("card" in n or "permanent account" in n or "number" in n)) or "पैन" in n:
        return "pan_card"
    
    # Voter ID / EPIC (मतदाता पहचान पत्र)
    if ("voter" in n and "id" in n) or "epic" in n or "मतदाता" in n or "election commission" in n:
        if "registration" not in n:  # US voter registration is different
            return "voter_id"
    
    # Indian Passport (भारतीय पासपोर्ट)
    if "passport" in n and ("indian" in n or "india" in n or "भारतीय" in n or "republic of india" in n):
        return "indian_passport"
    
    # Indian Driving License (ड्राइविंग लाइसेंस)
    if (("driver" in n or "driving" in n) and ("license" in n or "licence" in n)):
        if "indian" in n or "india" in n or "rto" in n or "dto" in n or "परिवहन" in n:
            return "indian_driving_license"
    
    # Ration Card (राशन कार्ड)
    if "ration" in n and "card" in n:
        return "ration_card"
    
    # NREGA Job Card (नरेगा जॉब कार्ड)
    if "nrega" in n or "mgnrega" in n or ("job" in n and "card" in n):
        return "nrega_job_card"
    
    # GST Certificate
    if "gst" in n or "gstin" in n or ("goods" in n and "services" in n and "tax" in n):
        return "gst_certificate"
    
    # Indian Birth Certificate (जन्म प्रमाण पत्र)
    if ("birth" in n and "certificate" in n) and ("india" in n or "जन्म" in n):
        return "indian_birth_certificate"
    
    # Indian Marriage Certificate (विवाह प्रमाण पत्र)
    if ("marriage" in n and "certificate" in n) and ("india" in n or "विवाह" in n):
        return "indian_marriage_certificate"
    
    # Caste Certificate (जाति प्रमाण पत्र)
    if "caste" in n and "certificate" in n:
        return "caste_certificate"
    
    # Income Certificate (आय प्रमाण पत्र)
    if "income" in n and "certificate" in n:
        return "income_certificate"
    
    # Domicile Certificate (अधिवास प्रमाण पत्र)
    if "domicile" in n and "certificate" in n:
        return "domicile_certificate"
    
    # Marksheet / Educational (अंकपत्र)
    if "marksheet" in n or "mark sheet" in n or "अंकपत्र" in n:
        return "marksheet"
    
    # Degree Certificate
    if "degree" in n and "certificate" in n:
        return "degree_certificate"
    
    # Bank Passbook
    if "passbook" in n or ("bank" in n and "pass" in n):
        return "bank_passbook"
    
    # Disability Certificate
    if "disability" in n and "certificate" in n:
        return "disability_certificate"
    
    # Pension Card / PPO
    if "pension" in n or "ppo" in n:
        return "pension_card"

    # ==========================================
    # U.S. IDENTITY DOCUMENTS (Secondary)
    # ==========================================
    
    # U.S. Driving License
    if ("driver" in n or "driving" in n) and ("license" in n or "licence" in n):
        if "mobile" in n or "mdl" in n:
            return "mobile_drivers_license"
        return "driving_license"
    
    if ("state" in n and ("id" in n or "identification" in n)) and "driver" not in n:
        if "real" in n:
            return "real_id"
        return "state_id"
    
    # U.S. Passport Documents
    if "passport" in n and "card" in n:
        return "passport_card"
    if "passport" in n and "card" not in n:
        return "passport"
    
    # U.S. Birth and Vital Records
    if ("birth" in n and "certificate" in n) or "birth_cert" in n:
        return "birth_certificate"
    if ("marriage" in n and "certificate" in n) or "marriage_cert" in n:
        return "marriage_certificate"
    if ("divorce" in n and ("decree" in n or "certificate" in n)):
        return "divorce_decree"
    
    # Social Security
    if ("social" in n and "security" in n) or "ssn" in n or "ss_card" in n:
        return "social_security_card"
    
    # Immigration Documents
    if ("permanent" in n and "resident" in n) or "green_card" in n or "prc" in n or "i-551" in n:
        return "permanent_resident_card"
    if ("naturalization" in n and "certificate" in n) or "n-550" in n or "n-570" in n:
        return "certificate_of_naturalization"
    if ("citizenship" in n and "certificate" in n) or "n-560" in n or "n-561" in n:
        return "certificate_of_citizenship"
    if ("employment" in n and "authorization" in n) or "ead" in n or "i-766" in n:
        return "employment_authorization_document"
    if "i-94" in n or ("arrival" in n and "departure" in n):
        return "form_i94"
    if ("visa" in n and ("us" in n or "american" in n)) or "h1b" in n or "h-1b" in n:
        return "us_visa"
    if ("reentry" in n and "permit" in n) or "i-327" in n:
        return "reentry_permit"
    
    # Military and Government IDs
    if ("military" in n and "id" in n) or "cac" in n or "common_access" in n:
        return "military_id"
    if ("veteran" in n and "id" in n) or "vic" in n:
        return "veteran_id"
    if ("tribal" in n and "id" in n) or "tribal_card" in n:
        return "tribal_id"
    if ("global" in n and "entry" in n) or "nexus" in n:
        return "global_entry_card"
    if ("tsa" in n and "precheck" in n) or "precheck" in n:
        return "tsa_precheck_card"
    if ("voter" in n and ("registration" in n or "card" in n)):
        return "voter_registration"
    
    # Professional and Educational
    if ("professional" in n and "license" in n) or ("license" in n and any(prof in n for prof in ["medical", "legal", "contractor", "nursing", "teaching"])):
        return "professional_license"
    if ("student" in n and "id" in n) or "student_card" in n:
        return "student_id"
    
    # Financial and Proof Documents
    if ("utility" in n and "bill" in n) or any(util in n for util in ["electric", "gas", "water", "internet", "cable"]):
        return "utility_bill"
    if ("lease" in n and "agreement" in n) or "rental_agreement" in n:
        return "lease_agreement"
    if ("bank" in n and "statement" in n) or "account_statement" in n:
        return "bank_statement"
    if ("insurance" in n and "card" in n) or any(ins in n for ins in ["health_insurance", "auto_insurance"]):
        return "insurance_card"
    if ("voided" in n and "check" in n) or "void_check" in n:
        return "voided_check"
    if ("direct" in n and "deposit" in n) or "dd_form" in n:
        return "direct_deposit"
    
    # Consular and International
    if ("consular" in n and "id" in n) or "matricula" in n:
        return "consular_id"
    
    # Digital IDs
    if ("digital" in n and "id" in n) or any(platform in n for platform in ["id.me", "login.gov"]):
        return "digital_id"
    
    # If it's any kind of identity document but we can't determine the specific type
    if any(identity_term in n for identity_term in ["id", "identification", "license", "card", "certificate", "document"]):
        return "identity_document"
    
    return "other_identity"


def _guess_identity_subtype_from_ocr(ocr_json: Dict[str, Any]) -> str:
    """
    Guess identity subtype from OCR-structured JSON.
    INDIAN DOCUMENTS ARE CHECKED FIRST with multi-language keyword support.
    Uses enhanced heuristics and LLM classification with comprehensive document type support.
    """
    # Prefer top-level page 1 extracted fields if present
    page1 = None
    if isinstance(ocr_json, dict):
        page1 = ocr_json.get("1") or ocr_json.get(1)
        if not isinstance(page1, dict):
            # Some pipelines might store directly under page-less keys; use entire object
            page1 = ocr_json

    # Compose a compact summary string for heuristic and/or LLM
    keys = []
    values = []
    if isinstance(page1, dict):
        for k, v in list(page1.items())[:50]:  # cap to avoid huge prompts
            try:
                ks = str(k).lower()
                vs = str(v).lower()
            except Exception:
                continue
            keys.append(ks)
            if len(vs) < 120:
                values.append(vs)
    text_blob = " ".join(keys + values)

    # Also check for raw text with regional scripts
    raw_text = ""
    if isinstance(page1, dict):
        for k, v in page1.items():
            raw_text += f" {k} {v}"

    # Enhanced heuristics for Indian and U.S. identity documents (cheap, no network)
    blob = text_blob
    
    # ==========================================
    # INDIAN IDENTITY DOCUMENTS (Check First)
    # Multi-language keyword detection
    # ==========================================
    
    # First, try multi-language keyword detection
    detected_doc = detect_document_by_keywords(raw_text)
    if detected_doc:
        print(f"[Classification] Detected document type via multi-language keywords: {detected_doc}")
        return detected_doc
    
    # Aadhaar Card (आधार कार्ड)
    aadhaar_number_patterns = [
        r"\b[2-9][0-9]{3}\s[0-9]{4}\s[0-9]{4}\b",
        r"\b[2-9][0-9]{11}\b",
    ]
    if (
        ("government of india" in blob and ("aadhaar" in blob or "aadhar" in blob))
        or any(term in blob for term in ["aadhaar", "aadhar", "uidai", "vid", "आधार", "यूआईडीएआई"])
        or any(re.search(pattern, blob) for pattern in aadhaar_number_patterns)
        or ("year of birth" in blob and ("aadhaar" in blob or "unique identification" in blob))
        or "unique identification authority" in blob
    ):
        return "aadhaar"

    # PAN Card (पैन कार्ड)
    pan_patterns = [
        r"\b[a-z]{3}[abcfghljptf][a-z][0-9]{4}[a-z]\b",  # Standard format
        r"\b[a-z]{3}[abcfghljptf][a-z]-[0-9]{4}-[a-z]\b",  # Hyphenated format
    ]
    if (
        ("income tax department" in blob and "permanent account number" in blob)
        or ("govt of india" in blob and "pan" in blob)
        or "permanent account number" in blob
        or any(re.search(pattern, blob) for pattern in pan_patterns)
        or any(term in blob for term in ["आयकर विभाग", "स्थायी खाता संख्या", "पैन"])
    ):
        return "pan_card"

    # Voter ID / EPIC (मतदाता पहचान पत्र)
    if (
        "election commission of india" in blob
        or "elector photo identity card" in blob
        or "epic no" in blob
        or ("voter" in blob and "identity card" in blob)
        or ("serial no" in blob and "part no" in blob and "assembly constituency" in blob)
        or any(term in blob for term in ["निर्वाचन आयोग", "मतदाता", "ईपीआईसी"])
    ):
        return "voter_id"

    # Indian Driving License (ड्राइविंग लाइसेंस)
    if (
        "indian union driving licence" in blob
        or "ministry of road transport" in blob
        or "transport department" in blob
        or ("driving licence no" in blob and "india" in blob)
        or ("driving license no" in blob and "india" in blob)
        or ("dl no" in blob and ("rto" in blob or "dto" in blob))
        or any(term in blob for term in ["परिवहन विभाग", "ड्राइविंग लाइसेंस", "आरटीओ"])
    ):
        return "indian_driving_license"

    # Indian Passport (भारतीय पासपोर्ट)
    if (
        ("republic of india" in blob and "passport" in blob)
        or ("passport no" in blob and "type" in blob and "nation" in blob and "india" in blob)
        or any(term in blob for term in ["भारत गणराज्य", "पासपोर्ट"])
    ):
        return "indian_passport"

    # GST Certificate
    if (
        "goods and services tax" in blob
        or "gstin" in blob
        or "gst certificate" in blob
        or ("gst" in blob and "registration" in blob)
        or any(term in blob for term in ["वस्तु एवं सेवा कर", "जीएसटीआईएन"])
    ):
        return "gst_certificate"
    
    # Ration Card (राशन कार्ड)
    if (
        "ration card" in blob
        or "public distribution" in blob
        or "food supply" in blob
        or any(term in blob for term in ["राशन कार्ड", "सार्वजनिक वितरण"])
    ):
        return "ration_card"
    
    # NREGA Job Card (नरेगा जॉब कार्ड)
    if (
        "nrega" in blob or "mgnrega" in blob
        or "job card" in blob
        or "rural employment" in blob
    ):
        return "nrega_job_card"
    
    # Indian Birth Certificate (जन्म प्रमाण पत्र)
    if (
        ("birth" in blob and "certificate" in blob and ("registrar" in blob or "india" in blob))
        or any(term in blob for term in ["जन्म प्रमाण पत्र", "जन्म पंजीकरण"])
    ):
        return "indian_birth_certificate"
    
    # Indian Marriage Certificate (विवाह प्रमाण पत्र)
    if (
        ("marriage" in blob and "certificate" in blob and ("registrar" in blob or "india" in blob))
        or any(term in blob for term in ["विवाह प्रमाण पत्र", "विवाह पंजीकरण"])
    ):
        return "indian_marriage_certificate"
    
    # Caste Certificate (जाति प्रमाण पत्र)
    if (
        "caste certificate" in blob
        or ("caste" in blob and "certificate" in blob)
        or any(term in blob for term in ["जाति प्रमाण पत्र", "अनुसूचित जाति", "अनुसूचित जनजाति"])
    ):
        return "caste_certificate"
    
    # Income Certificate (आय प्रमाण पत्र)
    if (
        "income certificate" in blob
        or ("income" in blob and "certificate" in blob)
        or any(term in blob for term in ["आय प्रमाण पत्र", "वार्षिक आय"])
    ):
        return "income_certificate"
    
    # Domicile Certificate (अधिवास प्रमाण पत्र)
    if (
        "domicile certificate" in blob
        or ("domicile" in blob and "certificate" in blob)
        or ("residence" in blob and "certificate" in blob)
        or any(term in blob for term in ["अधिवास प्रमाण पत्र", "निवास प्रमाण पत्र"])
    ):
        return "domicile_certificate"
    
    # Marksheet / Educational (अंकपत्र)
    if (
        "marksheet" in blob or "mark sheet" in blob
        or ("board" in blob and ("exam" in blob or "result" in blob))
        or any(term in blob for term in ["अंकपत्र", "परीक्षा परिणाम"])
    ):
        return "marksheet"
    
    # Degree Certificate
    if (
        ("degree" in blob and "certificate" in blob)
        or "convocation" in blob
        or ("university" in blob and "certificate" in blob)
    ):
        return "degree_certificate"
    
    # Bank Passbook
    if (
        "passbook" in blob
        or ("bank" in blob and "pass" in blob and "book" in blob)
    ):
        return "bank_passbook"
    
    # Disability Certificate
    if (
        "disability certificate" in blob
        or ("disability" in blob and "certificate" in blob)
        or "pwd certificate" in blob
    ):
        return "disability_certificate"
    
    # Pension Card / PPO
    if (
        "pension" in blob
        or "ppo" in blob
        or "pensioner" in blob
    ):
        return "pension_card"

    # ==========================================
    # U.S. IDENTITY DOCUMENTS (Secondary)
    # ==========================================
    
    # U.S. Driving License
    if any(w in blob for w in ["driver", "driving"]) and "license" in blob:
        if "mobile" in blob or "mdl" in blob:
            return "mobile_drivers_license"
        return "driving_license"
    if ("state" in blob and ("id" in blob or "identification" in blob)) and "driver" not in blob:
        if "real" in blob:
            return "real_id"
        return "state_id"
    
    # Passport Documents
    if "passport card" in blob:
        return "passport_card"
    if "passport" in blob:
        return "passport"
    
    # Birth and Vital Records
    if ("birth" in blob and "certificate" in blob) or "birth_cert" in blob:
        return "birth_certificate"
    if ("marriage" in blob and "certificate" in blob):
        return "marriage_certificate"
    if ("divorce" in blob and ("decree" in blob or "certificate" in blob)):
        return "divorce_decree"
    
    # Social Security
    if ("social" in blob and "security" in blob) or "ssn" in blob:
        return "social_security_card"
    
    # Immigration Documents
    if ("permanent" in blob and "resident" in blob) or "green_card" in blob or "i-551" in blob:
        return "permanent_resident_card"
    if ("naturalization" in blob and "certificate" in blob) or any(form in blob for form in ["n-550", "n-570"]):
        return "certificate_of_naturalization"
    if ("citizenship" in blob and "certificate" in blob) or any(form in blob for form in ["n-560", "n-561"]):
        return "certificate_of_citizenship"
    if ("employment" in blob and "authorization" in blob) or "ead" in blob or "i-766" in blob:
        return "employment_authorization_document"
    if "i-94" in blob or ("arrival" in blob and "departure" in blob):
        return "form_i94"
    if ("visa" in blob and ("us" in blob or "american" in blob)) or "h1b" in blob or "h-1b" in blob:
        return "us_visa"
    if ("reentry" in blob and "permit" in blob) or "i-327" in blob:
        return "reentry_permit"
    
    # Military and Government IDs
    if ("military" in blob and "id" in blob) or "cac" in blob or "common_access" in blob:
        return "military_id"
    if ("veteran" in blob and "id" in blob) or "vic" in blob:
        return "veteran_id"
    if ("tribal" in blob and "id" in blob):
        return "tribal_id"
    if ("global" in blob and "entry" in blob) or "nexus" in blob:
        return "global_entry_card"
    if ("tsa" in blob and "precheck" in blob) or "precheck" in blob:
        return "tsa_precheck_card"
    if ("voter" in blob and ("registration" in blob or "card" in blob)):
        return "voter_registration"
    
    # Professional and Educational
    if ("professional" in blob and "license" in blob) or ("license" in blob and any(prof in blob for prof in ["medical", "legal", "contractor", "nursing", "teaching"])):
        return "professional_license"
    if ("student" in blob and "id" in blob):
        return "student_id"
    
    # Financial and Proof Documents
    if ("utility" in blob and "bill" in blob) or any(util in blob for util in ["electric", "gas", "water", "internet", "cable"]):
        return "utility_bill"
    if ("lease" in blob and "agreement" in blob) or "rental_agreement" in blob:
        return "lease_agreement"
    if ("bank" in blob and "statement" in blob):
        return "bank_statement"
    if ("insurance" in blob and "card" in blob):
        return "insurance_card"
    if ("voided" in blob and "check" in blob):
        return "voided_check"
    if ("direct" in blob and "deposit" in blob):
        return "direct_deposit"
    
    # Consular and Digital IDs
    if ("consular" in blob and "id" in blob) or "matricula" in blob:
        return "consular_id"
    if ("digital" in blob and "id" in blob) or any(platform in blob for platform in ["id.me", "login.gov"]):
        return "digital_id"

    # LLM-based classification (comprehensive document type support)
    try:
        from openai import OpenAI  # type: ignore
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("no key")
        client = OpenAI(api_key=api_key)
        
        # Comprehensive list of all supported identity document types
        allowed_types = [
            "driving_license", "mobile_drivers_license", "state_id", "real_id",
            "passport", "passport_card", "indian_passport",
            "birth_certificate", "marriage_certificate", "divorce_decree",
            "social_security_card", "permanent_resident_card", "certificate_of_naturalization",
            "certificate_of_citizenship", "employment_authorization_document", "form_i94", "us_visa",
            "reentry_permit", "military_id", "veteran_id", "tribal_id", "global_entry_card",
            "tsa_precheck_card", "voter_registration", "professional_license", "student_id",
            "utility_bill", "lease_agreement", "bank_statement", "insurance_card",
            "voided_check", "direct_deposit", "consular_id", "digital_id",
            "aadhaar", "pan_card", "voter_id", "indian_driving_license", "gst_certificate",
            "identity_document", "other_identity"
        ]
        
        system = (
            "You are an expert classifier for U.S. and Indian identity documents. Return JSON only.\n"
            f"Supported document types: {', '.join(allowed_types)}\n"
            "Analyze the OCR data and classify the document to the most specific type.\n"
            "Consider document layout, field names, issuing authorities, and content patterns."
        )
        
        payload = {
            "document_name": ocr_json.get("document_name"),
            "page1_fields_preview": dict(list(page1.items())[:30]) if isinstance(page1, dict) else {},
            "text_content_sample": text_blob[:500]  # First 500 chars for context
        }
        
        resp = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": (
                    "Classify this U.S. identity document to one of the supported types. "
                    "Respond as {\"subtype\":\"document_type\", \"confidence\":\"high|medium|low\", \"reasoning\":\"brief explanation\"}.\n"
                    + str(payload)
                )},
            ],
        )
        content = resp.choices[0].message.content
        import json as _json
        result = _json.loads(content) or {}
        label = result.get("subtype", "other_identity")
        
        # Validate against allowed types
        if label in allowed_types:
            return label
        else:
            # If LLM returned an unexpected type, try to map it or fall back
            return "identity_document" if any(term in label for term in ["id", "license", "card", "certificate"]) else "other_identity"
            
    except Exception as e:
        # Silent fallback to generic identity document if it looks like an ID document
        if any(term in blob for term in ["id", "license", "card", "certificate", "document", "number"]):
            return "identity_document"
        return "other_identity"


def _generate_classification_failure_reason(message: str, doc_type: str, ingestion_name: str, actual_document_name: str = None) -> str:
    """
    Generate a user-friendly 2-line reason for classification failure using AI.
    """
    try:
        from openai import OpenAI
        import json as _json
        
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            # Fallback to basic reason
            if "mismatch" in message.lower():
                return f"Document type mismatch detected.\nPlease upload the correct document type."
            elif "expired" in message.lower():
                return "Your document has expired.\nPlease upload a current, valid document."
            else:
                return "Document validation failed.\nPlease upload a clear, valid document."
        
        client = OpenAI(api_key=api_key)
        
        system_prompt = """You are a user experience expert who creates clear, friendly, and actionable messages for document verification failures.

Your task is to generate a user-friendly reason message in EXACTLY 2 lines (maximum 150 characters total) that explains why the document was rejected.

Guidelines:
1. Be clear and direct - users need to understand immediately what went wrong
2. Be friendly and professional - avoid technical jargon
3. Be actionable - tell users exactly what to do next
4. Use simple language - write for a general audience
5. MUST be exactly 2 lines, each line under 75 characters
6. Write as plain text without quotes or special characters

Examples:

Document type mismatch:
You uploaded a Passport but selected Driver's License
Please upload the correct document type

Wrong document uploaded:
The document doesn't match what you specified
Please verify and upload the correct document

Blurry or unclear image:
The image quality is too low to verify your document
Please upload a clearer photo with all text visible

Document expired:
Your document has expired and cannot be accepted
Please upload a current, valid document

Return ONLY the 2-line message as plain text, nothing else. Do NOT use quotes around the text."""

        user_prompt = f"""Generate a user-friendly 2-line reason for this classification failure:

Expected Document: {ingestion_name}
Detected Document: {actual_document_name or "Unknown"}
Failure Message: {message}

Return EXACTLY 2 lines (under 75 characters each) that clearly explain what went wrong and what the user should do."""

        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        reason = resp.choices[0].message.content.strip()
        
        # Ensure it's exactly 2 lines
        lines = [line.strip() for line in reason.split('\n') if line.strip()]
        if len(lines) > 2:
            reason = '\n'.join(lines[:2])
        elif len(lines) == 1:
            reason = lines[0] + "\nPlease upload the correct document."
        else:
            reason = '\n'.join(lines)
        
        print(f"[AI REASON] Classification failure reason: {reason.replace(chr(10), ' | ')}")
        
        return reason
        
    except Exception as e:
        print(f"[WARN] Failed to generate AI reason for classification: {e}")
        # Fallback
        if "mismatch" in message.lower():
            return "Document type doesn't match what you selected.\nPlease upload the correct document type."
        elif "expired" in message.lower():
            return "Your document has expired.\nPlease upload a current, valid document."
        else:
            return "Document validation failed.\nPlease upload a clear, valid document."


def _extract_actual_document_name_from_ocr(ocr_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use LLM to intelligently extract and analyze the actual document from OCR content.
    Returns comprehensive document analysis including type, confidence, and key indicators.
    """
    # Get OCR content
    page1 = None
    if isinstance(ocr_json, dict):
        page1 = ocr_json.get("1") or ocr_json.get(1)
        if not isinstance(page1, dict):
            page1 = ocr_json

    # Compose text for analysis
    text_content = []
    if isinstance(page1, dict):
        for k, v in list(page1.items())[:50]:  # Get more fields for better analysis
            try:
                text_content.append(f"{k}: {v}")
            except Exception:
                continue
    
    text_blob = "\n".join(text_content)
    
    # Use LLM for intelligent document analysis
    try:
        from openai import OpenAI
        import json as _json
        
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return {"document_type": "Unknown Document", "confidence": "low", "reasoning": "No API key"}
        
        client = OpenAI(api_key=api_key)
        
        system_prompt = """You are an expert document classifier with deep knowledge of identity documents worldwide.

Analyze the OCR text and identify the document type with high precision. Consider:
1. Official headers and titles
2. Issuing authority names
3. Document layout and structure
4. Field names and patterns
5. Security features mentioned
6. Country-specific formats

Supported document types:
- Driver's License / Driving License
- State ID / State Identification Card
- Passport
- Passport Card
- Social Security Card
- Aadhaar Card (India)
- PAN Card (India)
- Voter ID Card (India)
- Indian Passport
- Aadhaar Card (India)
- Indian Driving License
- PAN Card (India)
- GST Certificate (India)
- Birth Certificate
- Marriage Certificate
- Divorce Decree
- Permanent Resident Card (Green Card)
- Employment Authorization Document (EAD)
- Military ID
- Veteran ID
- Professional License
- Student ID
- Utility Bill
- Bank Statement
- Lease Agreement
- Insurance Card
- Voter Registration
- Tribal ID
- Certificate of Naturalization
- Certificate of Citizenship

Return JSON with:
{
  "document_type": "exact document name",
  "confidence": "high|medium|low",
  "reasoning": "brief explanation of key indicators found",
  "issuing_authority": "detected authority if any",
  "country": "detected country",
  "key_fields_found": ["list of key fields that helped identify"]
}"""

        user_prompt = f"""Analyze this OCR data and identify the document type:

{text_blob[:2000]}

Provide detailed classification with confidence level and reasoning."""

        resp = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        result = _json.loads(resp.choices[0].message.content)
        print(f"[LLM CLASSIFICATION] Document: {result.get('document_type')}, Confidence: {result.get('confidence')}")
        print(f"[LLM CLASSIFICATION] Reasoning: {result.get('reasoning')}")
        
        return result
        
    except Exception as e:
        print(f"[WARN] LLM document classification failed: {e}")
        import traceback
        traceback.print_exc()
        return {"document_type": "Unknown Document", "confidence": "low", "reasoning": str(e)}


def Classification(state: PipelineState) -> PipelineState:
    """
    Classify document type, validate against expected category, and enforce
    a strict name-vs-identity-subtype check. If mismatched, mark as fail and
    instruct user to re-upload the correct document.
    """
    if state.ocr is None:
        raise ValueError("Nothing ingested; run OCR node first.")

    log_agent_event(state, "Document Classification", "start")

    # Check if OCR extraction failed
    ocr_json = state.ocr.ocr_json or {}
    if ocr_json.get("ocr_extraction_failed"):
        error_message = ocr_json.get("ocr_error_message", "Please upload a valid document with good quality. OCR was unable to extract data from the document.")
        
        # Generate AI-powered user-friendly reason
        ai_reason = _generate_classification_failure_reason(
            message=error_message,
            doc_type="unknown",
            ingestion_name=state.ingestion.document_name if state.ingestion else "Unknown Document",
            actual_document_name=None
        )
        
        # Mark classification as failed
        state.classification = ClassificationState(
            expected_category=state.ocr.doc_category or "",
            detected_doc_type="unknown",
            passed=False,
            message=error_message,
        )
        
        # Store failure result to database
        try:
            import json
            from datetime import datetime, timezone
            from ..tools.db import update_tblaigents_by_keys
            
            # Format ai_reason as list if it's a string
            if isinstance(ai_reason, str):
                # Split by newline or pipe separator to create list
                reason_lines = [line.strip() for line in ai_reason.replace('|', '\n').split('\n') if line.strip()]
                # Remove empty strings
                reason_lines = [line for line in reason_lines if line]
            else:
                reason_lines = ai_reason if isinstance(ai_reason, list) else [str(ai_reason)]
                # Remove empty strings
                reason_lines = [line for line in reason_lines if line]
            
            # Ensure we have at least one reason line
            if not reason_lines:
                reason_lines = ["Document validation failed", "Please upload a clear, valid document"]
            
            doc_verification_result_json = json.dumps({
                "score": 0,
                "stats": {
                    "score": 0,
                    "matched_fields": 0,
                    "mismatched_fields": 0
                },
                "reason": reason_lines,
                "details": []
            })
            
            if state.ingestion:
                update_tblaigents_by_keys(
                    FPCID=state.ingestion.FPCID,
                    airecordid=state.ingestion.airecordid,
                    updates={
                        "document_status": "fail",
                        "doc_verification_result": doc_verification_result_json,
                        "cross_validation": False,
                        "checklistId": state.ingestion.checklistId if hasattr(state.ingestion, 'checklistId') else None,
                        "doc_id": state.ingestion.doc_id if hasattr(state.ingestion, 'doc_id') else None,
                        "document_type": state.ingestion.document_type if hasattr(state.ingestion, 'document_type') else None,
                        "file_s3_location": f"s3://{state.ingestion.s3_bucket}/{state.ingestion.s3_key}" if hasattr(state.ingestion, 's3_bucket') and hasattr(state.ingestion, 's3_key') else None,
                        "metadata_s3_path": state.ingestion.metadata_s3_path if hasattr(state.ingestion, 'metadata_s3_path') else None,
                    },
                    document_name=state.ingestion.document_name,
                    LMRId=state.ingestion.LMRId,
                )
                print(f"[✓] OCR extraction failure result saved to database")
        except Exception as e:
            print(f"[WARN] Failed to save OCR extraction failure result to database: {e}")
        
        log_agent_event(state, "Document Classification", "completed", {
            "passed": False,
            "reason": "ocr_extraction_failed"
        })
        return state

    # Coarse category (bank_statement/identity/property/...) detected by OCR routing
    doc_category = (state.ocr.doc_category or "").strip().lower()
    doc_type = (ocr_json.get("doc_type") or "").strip().lower()

    # If ingestion agent is Identity Verification Agent, force expected to identity
    try:
        agent_name = (state.ingestion.agent_name or "").strip().lower() if state.ingestion else ""
    except Exception:
        agent_name = ""
    if agent_name == "identity verification agent":
        doc_category = "identity"
    elif not doc_category and doc_type:
        # If no expected category provided (e.g., missing from metadata), treat expected as detected
        doc_category = doc_type

    # First-level pass/fail on coarse category
    category_match = (doc_category == doc_type) and (doc_type != "")

    # Enhanced comprehensive document validation
    name_mismatch = False
    subtype_detected = None
    subtype_expected = None
    detailed_reason = None
    metadata_db_name_mismatch = False
    actual_document_mismatch = False
    llm_validated = False  # Track if LLM performed validation

    ingestion_name = (state.ingestion.document_name if state.ingestion else None) or ""
    
    # 1. Check for mismatch between metadata document_name and DB document_name
    if state.ingestion and state.ingestion.raw_metadata:
        metadata_doc_name = state.ingestion.raw_metadata.get("document_name", "").strip()
        db_doc_name = ingestion_name.strip()
        
        if metadata_doc_name and db_doc_name and metadata_doc_name.lower() != db_doc_name.lower():
            metadata_db_name_mismatch = True
            detailed_reason = (
                f"Document name mismatch: Metadata indicates '{metadata_doc_name}' but "
                f"database context shows '{db_doc_name}'. Please ensure the document name "
                f"is consistent or re-upload with the correct document type."
            )
            print(f"[ERROR] Metadata/DB document name mismatch: '{metadata_doc_name}' vs '{db_doc_name}'")
    
    # 2. Extract actual document name from OCR content and validate using LLM
    actual_document_analysis = None
    actual_document_name = None
    if doc_type == "identity" and not metadata_db_name_mismatch:
        actual_document_analysis = _extract_actual_document_name_from_ocr(ocr_json or {})
        actual_document_name = actual_document_analysis.get("document_type", "Unknown Document")
        confidence = actual_document_analysis.get("confidence", "low")
        reasoning = actual_document_analysis.get("reasoning", "")
        
        print(f"[INFO] LLM Document Analysis:")
        print(f"  - Detected Type: '{actual_document_name}'")
        print(f"  - Confidence: {confidence}")
        print(f"  - Reasoning: {reasoning}")
        
        # Only validate if confidence is medium or high
        if confidence in ["medium", "high"] and actual_document_name != "Unknown Document":
            # Use LLM to check if documents match
            try:
                from openai import OpenAI
                import json as _json
                
                api_key = os.environ.get("OPENAI_API_KEY", "")
                if api_key:
                    client = OpenAI(api_key=api_key)
                    
                    match_prompt = f"""Compare these two document types and determine if they match:

Claimed Document: "{ingestion_name}"
Detected Document: "{actual_document_name}"

Context: {reasoning}

Consider:
1. Exact matches (e.g., "Driver's License" = "Driving License")
2. Common variations (e.g., "Aadhar" = "Aadhaar")
3. Abbreviations (e.g., "DL" = "Driver's License")
4. Regional differences (e.g., "Licence" vs "License")
5. Related documents (e.g., "Passport Card" is a type of "Passport")

Return JSON:
{{
  "is_match": true/false,
  "confidence": "high|medium|low",
  "reasoning": "explanation of why they match or don't match"
}}"""

                    resp = client.chat.completions.create(
                        model="gpt-4o",
                        response_format={"type": "json_object"},
                        temperature=0,
                        messages=[
                            {"role": "system", "content": "You are an expert at comparing document types and identifying matches, considering variations, abbreviations, and regional differences."},
                            {"role": "user", "content": match_prompt}
                        ]
                    )
                    
                    match_result = _json.loads(resp.choices[0].message.content)
                    is_valid_match = match_result.get("is_match", False)
                    match_confidence = match_result.get("confidence", "low")
                    match_reasoning = match_result.get("reasoning", "")
                    
                    print(f"[LLM MATCH] Is Match: {is_valid_match}, Confidence: {match_confidence}")
                    print(f"[LLM MATCH] Reasoning: {match_reasoning}")
                    
                    # Only flag mismatch if LLM is confident it's not a match
                    if not is_valid_match and match_confidence in ["high", "medium"]:
                        actual_document_mismatch = True
                        detailed_reason = (
                            f"Document content mismatch: You specified '{ingestion_name}' but the uploaded "
                            f"document appears to be a '{actual_document_name}'. "
                            f"Analysis: {match_reasoning}. "
                            f"Please upload the correct document type or update the document name to match the uploaded file."
                        )
                        print(f"[ERROR] Document content mismatch: Claimed '{ingestion_name}' vs Actual '{actual_document_name}'")
                    else:
                        # LLM validated the match successfully
                        llm_validated = True
                        print(f"[✓] LLM validated document match successfully")
                else:
                    print("[WARN] No OpenAI API key, skipping LLM document matching")
                    
            except Exception as e:
                print(f"[WARN] LLM document matching failed: {e}")
                # Don't fail on LLM errors, just log
    
    # Only perform subtype validation if LLM validation wasn't performed or wasn't confident
    # If LLM already validated with high/medium confidence, trust that result
    perform_subtype_check = True
    if actual_document_analysis and actual_document_analysis.get("confidence") in ["high", "medium"]:
        # LLM already validated the document match, skip redundant subtype check
        perform_subtype_check = False
        print(f"[INFO] Skipping subtype check - LLM already validated document with {actual_document_analysis.get('confidence')} confidence")
    
    if doc_type == "identity" and ingestion_name and not metadata_db_name_mismatch and not actual_document_mismatch and perform_subtype_check:
        # Use LLM-detected document type if available and confident, otherwise fall back to heuristics
        if actual_document_analysis and actual_document_analysis.get("confidence") in ["high", "medium"]:
            # Map LLM document type to subtype
            llm_doc_type = actual_document_analysis.get("document_type", "").lower()
            subtype_detected = _map_display_name_to_identity_subtype(llm_doc_type)
            print(f"[INFO] Using LLM-detected document type: '{llm_doc_type}' -> '{subtype_detected}'")
        else:
            # Fall back to heuristic-based detection
            subtype_detected = _guess_identity_subtype_from_ocr(ocr_json or {})
            print(f"[INFO] Using heuristic OCR detection: -> '{subtype_detected}'")
        
        subtype_expected = _map_display_name_to_identity_subtype(ingestion_name)
        
        # Log detection results for debugging
        print(f"[INFO] Document name mapping: '{ingestion_name}' -> '{subtype_expected}'")
        print(f"[INFO] Final detected subtype: '{subtype_detected}'")
        
        # Enhanced mismatch detection logic
        # Allow some flexibility for related document types
        compatible_groups = [
            {"driving_license", "mobile_drivers_license"},  # Mobile DL is compatible with regular DL
            {"state_id", "real_id", "passport_card"},  # State ID, REAL ID, and Passport Card are often similar in format
            {"passport", "passport_card"},  # Passport types are related
            {"certificate_of_naturalization", "certificate_of_citizenship"},  # Citizenship documents
            {"military_id", "veteran_id"},  # Military-related IDs
            {"utility_bill", "lease_agreement", "bank_statement"},  # Proof of residence documents
            {"employment_authorization_document", "permanent_resident_card"},  # Immigration documents
            {"consular_id", "identity_document"},  # International identity documents
        ]
        
        # Check if documents are in compatible groups
        is_compatible = False
        if subtype_expected == subtype_detected:
            is_compatible = True
        else:
            for group in compatible_groups:
                if subtype_expected in group and subtype_detected in group:
                    is_compatible = True
                    break
        
        # Only flag mismatch if both are specific and not compatible
        if (subtype_expected not in {"other_identity", "identity_document"} and 
            subtype_detected not in {"other_identity", "identity_document"} and 
            not is_compatible):
            name_mismatch = True
            detailed_reason = (
                f"Document type mismatch: You indicated '{ingestion_name}' (expected: {subtype_expected}), "
                f"but the uploaded document appears to be a {subtype_detected.replace('_', ' ').title()}. "
                f"Please upload the correct document type or update the document name."
            )
        elif subtype_detected == "other_identity" and subtype_expected != "other_identity":
            # Special case: couldn't detect specific type but user specified one
            print(f"[WARN] Could not definitively identify document type from OCR. Expected: {subtype_expected}")
            # Don't fail in this case, just log the warning

    passed = category_match and not name_mismatch and not metadata_db_name_mismatch and not actual_document_mismatch
    
    # Generate appropriate message based on results
    if passed:
        message = "pass"
        if subtype_detected and subtype_expected:
            print(f"[✓] Document classification successful: {subtype_detected}")
        if actual_document_name:
            print(f"[✓] Document content validation successful: {actual_document_name}")
    else:
        if actual_document_mismatch:
            message = detailed_reason
        elif metadata_db_name_mismatch:
            message = detailed_reason
        elif not category_match:
            message = (
                f"Document category mismatch: Expected '{doc_category}' document, "
                f"but detected '{doc_type}'. Please upload a valid identity document "
                f"(accepted formats: PNG, JPG, JPEG, PDF, DOCX, etc.)."
            )
        else:
            message = detailed_reason or (
                "Document name and content mismatch. Please ensure the uploaded document "
                "matches the document type you specified, or update the document name to match "
                "the uploaded file."
            )
    
    # Enhanced logging for debugging
    print(f"[INFO] Classification Results:")
    print(f"  - Expected Category: {doc_category}")
    print(f"  - Detected Type: {doc_type}")
    print(f"  - Document Name (DB): {ingestion_name}")
    if state.ingestion and state.ingestion.raw_metadata:
        metadata_doc_name = state.ingestion.raw_metadata.get("document_name", "")
        print(f"  - Document Name (Metadata): {metadata_doc_name}")
    if actual_document_name:
        print(f"  - Actual Document (OCR): {actual_document_name}")
    print(f"  - Expected Subtype: {subtype_expected}")
    print(f"  - Detected Subtype: {subtype_detected}")
    print(f"  - Category Match: {category_match}")
    print(f"  - Name Match: {not name_mismatch}")
    print(f"  - Metadata/DB Name Match: {not metadata_db_name_mismatch}")
    print(f"  - Actual Document Match: {not actual_document_mismatch}")
    print(f"  - Final Result: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(f"  - Reason: {message}")

    state.classification = ClassificationState(
        expected_category=doc_category,
        detected_doc_type=doc_type,
        passed=passed,
        message=message,
    )
    
    # Store additional classification details for debugging
    if hasattr(state.classification, '__dict__'):
        state.classification.__dict__.update({
            'subtype_expected': subtype_expected,
            'subtype_detected': subtype_detected,
            'category_match': category_match,
            'name_mismatch': name_mismatch
        })
    
    log_agent_event(state, "Document Classification", "completed", {
        "passed": passed, 
        "detected": doc_type,
        "subtype_expected": subtype_expected,
        "subtype_detected": subtype_detected,
        "document_name_db": ingestion_name,
        "document_name_metadata": (state.ingestion.raw_metadata.get("document_name", "") if state.ingestion and state.ingestion.raw_metadata else ""),
        "actual_document_name": actual_document_name,
        "metadata_db_name_match": not metadata_db_name_mismatch,
        "actual_document_match": not actual_document_mismatch
    })
    
    # If classification FAILED, store validation result to database immediately
    # (because pipeline won't reach extraction node for failed docs)
    if not passed:
        try:
            import json
            from datetime import datetime, timezone
            from ..tools.db import update_tblaigents_by_keys
            
            # Determine failure type and suggestions based on message
            failure_type = "validation_failed"
            suggestions = []
            message_lower = message.lower()
            
            if "mismatch" in message_lower or "wrong" in message_lower:
                failure_type = "document_mismatch"
                suggestions = [
                    "Upload the correct document type that matches what you selected",
                    "Ensure the document image is clear and readable",
                    "Check that you're uploading the front side of the document"
                ]
            elif "expired" in message_lower:
                failure_type = "expired"
                suggestions = [
                    "Please upload a valid, non-expired document",
                    "If you recently renewed, upload the new document"
                ]
            elif "content" in message_lower:
                failure_type = "content_mismatch"
                suggestions = [
                    "The document content doesn't match what was specified",
                    "Please verify you uploaded the correct document",
                    "Re-upload with a clearer image if text is hard to read"
                ]
            else:
                suggestions = [
                    "Please re-upload a clear photo of your document",
                    "Ensure all text and information is visible",
                    "Contact support if the issue persists"
                ]
            
            # Generate AI-powered user-friendly reason
            ai_reason = _generate_classification_failure_reason(
                message=message,
                doc_type=doc_type,
                ingestion_name=ingestion_name,
                actual_document_name=actual_document_name
            )
            
            # Format ai_reason as list if it's a string
            if isinstance(ai_reason, str):
                # Split by newline or pipe separator to create list
                reason_lines = [line.strip() for line in ai_reason.replace('|', '\n').split('\n') if line.strip()]
                # Remove empty strings
                reason_lines = [line for line in reason_lines if line]
            else:
                reason_lines = ai_reason if isinstance(ai_reason, list) else [str(ai_reason)]
                # Remove empty strings
                reason_lines = [line for line in reason_lines if line]
            
            # Ensure we have at least one reason line
            if not reason_lines:
                reason_lines = ["Document type mismatch detected", "Please upload the correct document type"]
            
            # Generate validation result JSON
            doc_verification_result_json = json.dumps({
                "score": 0,
                "stats": {
                    "score": 0,
                    "matched_fields": 0,
                    "mismatched_fields": 0
                },
                "reason": reason_lines,
                "details": []
            })
            
            # Update database with failure details
            if state.ingestion:
                update_tblaigents_by_keys(
                    FPCID=state.ingestion.FPCID,
                    airecordid=state.ingestion.airecordid,
                    updates={
                        "document_status": "fail",
                        "Validation_status": "fail",  # Set Validation_status to fail as well
                        "doc_verification_result": doc_verification_result_json,
                        "cross_validation": False,
                        "checklistId": state.ingestion.checklistId if hasattr(state.ingestion, 'checklistId') else None,
                        "doc_id": state.ingestion.doc_id if hasattr(state.ingestion, 'doc_id') else None,
                        "document_type": state.ingestion.document_type if hasattr(state.ingestion, 'document_type') else None,
                        "file_s3_location": f"s3://{state.ingestion.s3_bucket}/{state.ingestion.s3_key}" if hasattr(state.ingestion, 's3_bucket') and hasattr(state.ingestion, 's3_key') else None,
                        "metadata_s3_path": state.ingestion.metadata_s3_path if hasattr(state.ingestion, 'metadata_s3_path') else None,
                        "uploadedat": state.ingestion.uploaded_at if hasattr(state.ingestion, 'uploaded_at') else None,
                    },
                    document_name=state.ingestion.document_name,
                    LMRId=state.ingestion.LMRId,
                )
                print(f"[✓] Validation result saved to database: {failure_type}")
                print(f"[✓] Both document_status and Validation_status set to 'fail'")
        except Exception as e:
            print(f"[WARN] Failed to save validation result to database: {e}")
    
    return state
