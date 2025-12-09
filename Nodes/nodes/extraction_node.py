"""
Extraction node for field extraction and data cleaning.
Deterministic GPT extraction + strict post-clean:
- Removes empty/missing fields from final JSON
- Deduplicates common aliases (dob/dateOfBirth, issueDate/dateIssued)
- Dynamic S3 key path (FPCID, LMRId, UTC date)
- INDIAN DOCUMENTS FIRST with multi-language support (22+ languages)
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, List

from ..config.state_models import PipelineState, ExtractionState
from ..tools.bedrock_client import get_bedrock_client, strip_json_code_fences
from ..utils.helpers import log_agent_event
from ..utils.language_utils import (
    detect_languages, get_language_suffix, normalize_multilingual_fields,
    LANGUAGE_CODES, detect_scripts
)
from ..tools.aws_services import get_s3_client
from ..tools.db import update_tblaigents_by_keys


# ----------------------------------------------------------------------
# Document field definitions - INDIAN DOCUMENTS FIRST
# Multi-language fields use suffix convention: fieldName_LanguageCode
# e.g., holderName (English), holderName_Hi (Hindi), holderName_Ta (Tamil)
# ----------------------------------------------------------------------
DOC_FIELDS = {
    # ==========================================
    # INDIAN IDENTITY DOCUMENTS (Primary Focus)
    # ==========================================
    
    # Aadhaar Card (आधार कार्ड) - Most common Indian ID
    "aadhaar": [
        # Core fields (English)
        "aadhaarNumber", "vid", "holderName", "gender",
        "yearOfBirth", "dateOfBirth",
        "fatherName", "motherName", "husbandName", "spouseName",
        "addressLine1", "addressLine2", "city", "district", "state", "pinCode",
        "mobileNumber", "email", "issueDate", "qrData", "photoReference",
        # Hindi variants (most common)
        "holderName_Hi", "fatherName_Hi", "motherName_Hi", "husbandName_Hi",
        "addressLine1_Hi", "addressLine2_Hi", "city_Hi", "district_Hi", "state_Hi",
        # Regional language variants (dynamically added based on detection)
        "detectedLanguages"
    ],
    
    # PAN Card (पैन कार्ड)
    "pan_card": [
        "panNumber", "holderName", "fatherName", "motherName",
        "dateOfBirth", "issueDate", "signature", "photoReference", "qrData",
        # Hindi variants
        "holderName_Hi", "fatherName_Hi",
        "detectedLanguages"
    ],
    
    # Voter ID / EPIC (मतदाता पहचान पत्र)
    "voter_id": [
        "epicNumber", "holderName", "fatherName", "husbandName", "motherName",
        "relationType", "gender", "dateOfBirth", "age",
        "addressLine1", "addressLine2", "assemblyConstituency", "parliamentaryConstituency",
        "partNumber", "serialNumber", "pollingStation",
        "issueDate", "qrData", "photoReference",
        # Hindi variants
        "holderName_Hi", "fatherName_Hi", "husbandName_Hi",
        "addressLine1_Hi", "assemblyConstituency_Hi",
        "detectedLanguages"
    ],
    
    # Indian Passport (भारतीय पासपोर्ट)
    "indian_passport": [
        "passportNumber", "type", "countryCode",
        "surname", "givenNames", "sex", "dateOfBirth", "placeOfBirth",
        "nationality", "fileNumber",
        "issueDate", "expirationDate", "issuingAuthority", "passportOffice",
        "oldPassportNumber", "fatherName", "motherName", "spouseName",
        "addressLine1", "addressLine2", "city", "state", "pinCode",
        "emergencyContact", "photoReference", "mrzLine1", "mrzLine2",
        # Hindi variants
        "surname_Hi", "givenNames_Hi", "placeOfBirth_Hi",
        "fatherName_Hi", "motherName_Hi", "spouseName_Hi",
        "detectedLanguages"
    ],
    
    # Indian Driving License (ड्राइविंग लाइसेंस)
    "indian_driving_license": [
        "licenseNumber", "holderName", "fatherName", "motherName",
        "dateOfBirth", "bloodGroup",
        "addressLine1", "addressLine2", "city", "district", "state", "pinCode",
        "issuingRTO", "issuingAuthority",
        "issueDate", "validTillNonTransport", "validTillTransport",
        "covList", "vehicleClasses", "restrictions", "endorsements",
        "badgeNumber", "transportValidity", "hazardousValidity", "hillValidity",
        "qrData", "photoReference",
        # Regional variants
        "holderName_Hi", "fatherName_Hi", "addressLine1_Hi", "city_Hi", "state_Hi",
        "detectedLanguages"
    ],
    
    # Ration Card (राशन कार्ड)
    "ration_card": [
        "cardNumber", "cardType", "headOfFamily", "familyMembers", "numberOfMembers",
        "addressLine1", "addressLine2", "city", "district", "state", "pinCode",
        "issuingAuthority", "issueDate", "validTill",
        "fpsNumber", "fpsName", "category",
        # Regional variants
        "headOfFamily_Hi", "addressLine1_Hi", "city_Hi", "district_Hi", "state_Hi",
        "detectedLanguages"
    ],
    
    # NREGA Job Card (नरेगा जॉब कार्ड)
    "nrega_job_card": [
        "jobCardNumber", "holderName", "headOfHousehold", "registrationNumber",
        "village", "gramPanchayat", "block", "district", "state",
        "familyMembers", "issueDate", "photoReference",
        # Regional variants
        "holderName_Hi", "headOfHousehold_Hi", "village_Hi", "gramPanchayat_Hi",
        "block_Hi", "district_Hi", "state_Hi",
        "detectedLanguages"
    ],
    
    # GST Certificate
    "gst_certificate": [
        "gstin", "legalName", "tradeName", "constitutionOfBusiness", "registrationType",
        "registrationDate", "validityDate", "liabilityDate",
        "principalPlaceOfBusiness", "additionalPlaceOfBusiness",
        "jurisdictionState", "jurisdictionCentre",
        "taxpayerType", "authorizedSignatory", "qrData",
        # Regional variants
        "tradeName_Hi", "legalName_Hi",
        "detectedLanguages"
    ],
    
    # Indian Birth Certificate (जन्म प्रमाण पत्र)
    "indian_birth_certificate": [
        "certificateNumber", "registrationNumber",
        "childName", "sex", "dateOfBirth", "placeOfBirth",
        "fatherName", "motherName", "fatherNationality", "motherNationality",
        "addressAtBirth", "permanentAddress",
        "registrationDate", "issueDate", "issuingAuthority",
        "registrarName", "registrarSignature",
        # Regional variants
        "childName_Hi", "fatherName_Hi", "motherName_Hi",
        "placeOfBirth_Hi", "addressAtBirth_Hi", "issuingAuthority_Hi",
        "detectedLanguages"
    ],
    
    # Indian Marriage Certificate (विवाह प्रमाण पत्र)
    "indian_marriage_certificate": [
        "certificateNumber", "registrationNumber",
        "husbandName", "husbandDob", "husbandAge", "husbandFatherName",
        "wifeName", "wifeDob", "wifeAge", "wifeFatherName",
        "marriageDate", "marriagePlace",
        "witness1Name", "witness2Name",
        "registrationDate", "issueDate", "issuingAuthority",
        "registrarName", "registrarSignature",
        # Regional variants
        "husbandName_Hi", "wifeName_Hi", "marriagePlace_Hi",
        "witness1Name_Hi", "witness2Name_Hi", "issuingAuthority_Hi",
        "detectedLanguages"
    ],
    
    # Caste Certificate (जाति प्रमाण पत्र)
    "caste_certificate": [
        "certificateNumber", "holderName", "fatherName",
        "caste", "category", "subCaste",
        "addressLine1", "village", "tehsil", "district", "state",
        "issueDate", "validTill", "issuingAuthority", "purpose",
        # Regional variants
        "holderName_Hi", "fatherName_Hi", "caste_Hi", "category_Hi",
        "village_Hi", "tehsil_Hi", "district_Hi", "state_Hi", "issuingAuthority_Hi",
        "detectedLanguages"
    ],
    
    # Income Certificate (आय प्रमाण पत्र)
    "income_certificate": [
        "certificateNumber", "holderName", "fatherName",
        "annualIncome", "incomeSource", "occupation",
        "addressLine1", "village", "tehsil", "district", "state",
        "issueDate", "validTill", "issuingAuthority", "purpose", "financialYear",
        # Regional variants
        "holderName_Hi", "fatherName_Hi", "occupation_Hi",
        "village_Hi", "tehsil_Hi", "district_Hi", "state_Hi", "issuingAuthority_Hi",
        "detectedLanguages"
    ],
    
    # Domicile Certificate (अधिवास प्रमाण पत्र)
    "domicile_certificate": [
        "certificateNumber", "holderName", "fatherName", "motherName",
        "dateOfBirth", "placeOfBirth",
        "permanentAddress", "village", "tehsil", "district", "state",
        "residingSince", "issueDate", "validTill", "issuingAuthority",
        # Regional variants
        "holderName_Hi", "fatherName_Hi", "motherName_Hi",
        "placeOfBirth_Hi", "permanentAddress_Hi",
        "village_Hi", "tehsil_Hi", "district_Hi", "state_Hi", "issuingAuthority_Hi",
        "detectedLanguages"
    ],
    
    # Educational Documents - Marksheet (अंकपत्र)
    "marksheet": [
        "rollNumber", "registrationNumber", "studentName", "fatherName", "motherName",
        "dateOfBirth", "schoolName", "boardName", "examName", "examYear",
        "subjects", "totalMarks", "marksObtained", "percentage", "grade",
        "result", "issueDate", "certificateNumber",
        # Regional variants
        "studentName_Hi", "fatherName_Hi", "motherName_Hi",
        "schoolName_Hi", "boardName_Hi",
        "detectedLanguages"
    ],
    
    # Degree Certificate
    "degree_certificate": [
        "certificateNumber", "registrationNumber", "studentName", "fatherName",
        "dateOfBirth", "universityName", "collegeName",
        "degreeName", "specialization", "examYear", "convocationDate",
        "grade", "cgpa", "division", "issueDate",
        # Regional variants
        "studentName_Hi", "fatherName_Hi", "universityName_Hi", "collegeName_Hi",
        "degreeName_Hi",
        "detectedLanguages"
    ],
    
    # Bank Passbook (as ID proof)
    "bank_passbook": [
        "accountNumber", "accountHolderName", "bankName", "branchName", "ifscCode",
        "addressLine1", "addressLine2", "city", "state", "pinCode",
        "accountType", "openingDate", "nomineeName",
        # Regional variants
        "accountHolderName_Hi", "bankName_Hi", "branchName_Hi",
        "addressLine1_Hi", "city_Hi", "state_Hi",
        "detectedLanguages"
    ],
    
    # Disability Certificate
    "disability_certificate": [
        "certificateNumber", "holderName", "fatherName", "motherName",
        "dateOfBirth", "gender", "disabilityType", "disabilityPercentage",
        "addressLine1", "district", "state",
        "issueDate", "validTill", "issuingAuthority", "issuingHospital",
        # Regional variants
        "holderName_Hi", "fatherName_Hi", "disabilityType_Hi",
        "addressLine1_Hi", "district_Hi", "state_Hi", "issuingAuthority_Hi",
        "detectedLanguages"
    ],
    
    # Pension Card / PPO
    "pension_card": [
        "ppoNumber", "pensionerName", "fatherName", "spouseName",
        "dateOfBirth", "retirementDate", "pensionType",
        "bankName", "accountNumber", "ifscCode",
        "addressLine1", "district", "state", "pinCode",
        "issueDate", "issuingAuthority",
        # Regional variants
        "pensionerName_Hi", "fatherName_Hi", "spouseName_Hi",
        "addressLine1_Hi", "district_Hi", "state_Hi",
        "detectedLanguages"
    ],
    
    # ==========================================
    # U.S. IDENTITY DOCUMENTS (Secondary)
    # ==========================================
    
    # U.S. Driving License
    "driving_license": [
        "firstName", "middleName", "lastName", "suffix", "dob",
        "addressLine1", "addressLine2", "city", "state", "zip",
        "countryName", "expirationDate", "idNumber", "licenseNumber", "issuingState", "issueDate",
        "class", "restrictions", "endorsements"
    ],
    
    "mobile_drivers_license": [
        "firstName", "middleName", "lastName", "suffix", "dob",
        "licenseNumber", "issuingState", "issueDate", "expirationDate",
        "digitalSignature", "qrCode", "mobileAppProvider"
    ],
    
    "state_id": [
        "firstName", "middleName", "lastName", "suffix", "dob",
        "addressLine1", "addressLine2", "city", "state", "zip",
        "countryName", "expirationDate", "idNumber", "issuingState", "issueDate"
    ],
    
    "real_id": [
        "firstName", "middleName", "lastName", "suffix", "dob",
        "addressLine1", "addressLine2", "city", "state", "zip",
        "countryName", "expirationDate", "idNumber", "issuingState", "issueDate", "realIdCompliant"
    ],
    
    # U.S. Passport Documents
    "passport": [
        "passportNumber", "firstName", "middleName", "lastName", "suffix",
        "issuingCountry", "dateOfBirth", "issueDate", "expirationDate",
        "placeOfBirth", "nationality", "sex"
    ],
    
    "passport_card": [
        "passportCardNumber", "firstName", "middleName", "lastName", "suffix",
        "issuingCountry", "dateOfBirth", "issueDate", "expirationDate",
        "placeOfBirth", "nationality", "sex"
    ],
    
    # U.S. Birth and Vital Records
    "birth_certificate": [
        "firstName", "middleName", "lastName", "dateOfBirth", "stateOfBirth", "dateIssued",
        "certificateNumber", "registrarSignature", "sealOfState"
    ],
    
    "marriage_certificate": [
        "spouseName1", "spouseName2", "marriageDate", "marriagePlace",
        "certificateNumber", "issuingOffice", "officiantName", "witnessNames"
    ],
    
    "divorce_decree": [
        "petitionerName", "respondentName", "divorceDate", "courtName",
        "caseNumber", "judgeName", "finalDecreeDate"
    ],
    
    # Social Security
    "social_security_card": [
        "firstName", "middleName", "lastName", "suffix", "socialSecurityNumber", "number"
    ],
    
    # Immigration Documents
    "permanent_resident_card": [
        "firstName", "middleName", "lastName", "suffix", "dateOfBirth",
        "alienNumber", "cardNumber", "categoryCode", "countryOfBirth",
        "issuingCountry", "expirationDate", "residentSince"
    ],
    
    "certificate_of_naturalization": [
        "firstName", "middleName", "lastName", "suffix", "dateOfBirth",
        "certificateNumber", "dateOfNaturalization", "placeOfNaturalization",
        "formerNationality","issuingOffice"
    ],
    "certificate_of_citizenship": [
        "firstName","middleName","lastName","suffix","dateOfBirth",
        "certificateNumber","dateOfCitizenship","placeOfBirth",
        "issuingOffice","parentCitizenship"
    ],
    "employment_authorization_document": [
        "firstName","middleName","lastName","suffix","dateOfBirth",
        "alienNumber","cardNumber","categoryCode","countryOfBirth",
        "expirationDate","employmentAuthorized"
    ],
    "form_i94": [
        "firstName","middleName","lastName","admissionNumber",
        "dateOfArrival","dateOfDeparture","portOfEntry","classOfAdmission"
    ],
    "us_visa": [
        "firstName","middleName","lastName","suffix","dateOfBirth",
        "visaNumber","visaType","issueDate","expirationDate",
        "issuingPost","nationality","passportNumber"
    ],
    "reentry_permit": [
        "firstName","middleName","lastName","suffix","dateOfBirth",
        "permitNumber","issueDate","expirationDate","alienNumber"
    ],
    
    # Military and Government IDs
    "military_id": [
        "firstName","middleName","lastName","suffix","dateOfBirth",
        "serviceNumber","rank","branch","issueDate","expirationDate",
        "bloodType","sponsor"
    ],
    "veteran_id": [
        "firstName","middleName","lastName","suffix","dateOfBirth",
        "veteranIdNumber","issueDate","expirationDate","branch","serviceYears"
    ],
    "tribal_id": [
        "firstName","middleName","lastName","suffix","dateOfBirth",
        "tribalIdNumber","tribeName","issueDate","expirationDate","bloodQuantum"
    ],
    "global_entry_card": [
        "firstName","middleName","lastName","suffix","dateOfBirth",
        "passId","membershipNumber","issueDate","expirationDate"
    ],
    "tsa_precheck_card": [
        "firstName","middleName","lastName","suffix","dateOfBirth",
        "knownTravelerNumber","issueDate","expirationDate"
    ],
    "voter_registration": [
        "firstName","middleName","lastName","suffix","dateOfBirth",
        "voterIdNumber","registrationDate","politicalParty","precinct",
        "addressLine1","addressLine2","city","state","zip"
    ],
    
    # Professional and Educational
    "professional_license": [
        "firstName","middleName","lastName","suffix","licenseNumber",
        "licenseType","profession","issueDate","expirationDate",
        "issuingState","issuingBoard"
    ],
    "student_id": [
        "firstName","middleName","lastName","suffix","studentId",
        "institution","program","issueDate","expirationDate","academicYear"
    ],
    
    # Financial and Proof Documents
    "utility_bill": [
        "utilityBillType","serviceProvider","serviceProviderAddress",
        "accountHolderBillingName","accountHolderBillingAddress",
        "serviceAddress","accountNumber","billDate","dueDate",
        "periodStartDate","periodEndDate","amountDue"
    ],
    "lease_agreement": [
        "tenantName","landlordName","propertyAddress","leaseStartDate",
        "leaseEndDate","monthlyRent","securityDeposit","signatureDate"
    ],
    "bank_statement": [
        "accountHolderName","bankName","accountNumber","routingNumber",
        "statementDate","statementPeriod","accountType","balance"
    ],
    "insurance_card": [
        "policyHolderName","policyNumber","groupNumber","insuranceCompany",
        "effectiveDate","expirationDate","coverageType","dependents"
    ],
    "voided_check": [
        "accountHolderName","bankName","bankAddress","accountNumber",
        "routingNumber","checkNumber","checkDate"
    ],
    "direct_deposit": [
        "employeeName","employeeAddress","employeePhoneNumber","employeeSocialSecurityNumber",
        "bankName","bankAddress","accountNumber","routingNumber","typeOfAccount",
        "partneringInstitutionName","employeeSignature","employeeSignatureDate"
    ],
    
    # International and Digital IDs
    "consular_id": [
        "firstName","middleName","lastName","suffix","dateOfBirth",
        "consularIdNumber","issuingConsulate","nationality","issueDate","expirationDate"
    ],
    "digital_id": [
        "firstName","middleName","lastName","suffix","digitalIdNumber",
        "platform","issueDate","expirationDate","verificationLevel"
    ],
    
    # Visa Types (including H1B)
    "h1b_visa": [
        "firstName","middleName","lastName","suffix",
        "dateOfBirth","issuingCountry","expirationDate","visaNumber","petitionNumber"
    ],
    
    # Generic identity document for unknown types
    "identity_document": [
        "firstName","middleName","lastName","suffix","dob","dateOfBirth",
        "address","addressLine1","addressLine2","city","state","zip","country",
        "idNumber","documentNumber","passportNumber","licenseNumber",
        "issueDate","expirationDate","dateIssued","issuingAuthority",
        "nationality","placeOfBirth","sex","height","weight","eyeColor","hairColor"
    ],
    
    # Generic fallback
    "other": [
        "firstName","middleName","lastName","suffix",
        "dob","dateOfBirth","address","city","state","zip","country",
        "idNumber","passportNumber","accountNumber","ssn",
        "issueDate","expirationDate","dateIssued","residentSince","documentType"
    ],
}


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
_EMPTY_STRINGS = {"", "n/a", "na", "none", "null"}  # case-insensitive

def _is_empty_value(v: Any) -> bool:
    """Treat empty strings/whitespace/None/empty collections as empty."""
    if v is None:
        return True
    if isinstance(v, str):
        s = v.strip()
        return s == "" or s.lower() in _EMPTY_STRINGS
    if isinstance(v, (list, dict, set, tuple)):
        return len(v) == 0
    return False

# Alias groups where values represent the same concept; keep only one.
# Order = priority (first wins if both have values).
_ALIAS_GROUPS = [
    ("dob", "dateOfBirth"),
    ("issueDate", "dateIssued"),
]

def _normalize_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """Lowercase keys for case-insensitive matching (values preserved)."""
    if not isinstance(d, dict):
        return d
    return {k.lower(): _normalize_keys(v) for k, v in d.items()}

def _postprocess_complete_schema(cleaned: Dict[str, Any], fields: list[str]) -> Dict[str, Any]:
    """Ensure all expected fields exist (with empty string if missing)."""
    return {f: cleaned.get(f, "") for f in fields}

def _dedupe_aliases(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collapse alias groups so only the first non-empty key in each group remains.
    Example: if dob and dateOfBirth both have values, keep dob (first), drop dateOfBirth.
    """
    result = dict(data)
    for group in _ALIAS_GROUPS:
        # find first non-empty in group
        keep_key = None
        keep_val = None
        for k in group:
            if k in result and not _is_empty_value(result[k]):
                keep_key = k
                keep_val = result[k]
                break
        # remove others
        for k in group:
            if k in result:
                if k != keep_key:
                    result.pop(k, None)
        # ensure the kept one is present (if any value existed)
        if keep_key is not None:
            result[keep_key] = keep_val
    return result

def _drop_empty_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove fields whose value is empty per _is_empty_value."""
    return {k: v for k, v in data.items() if not _is_empty_value(v)}

def _validate_extracted_fields(extracted: Dict[str, Any], expected_fields: list[str], doc_type: str) -> Dict[str, Any]:
    """
    Validate that extracted fields match expected fields for the document type.
    Remove any unexpected fields and ensure only valid fields are present.
    """
    validated = {}
    unexpected_fields = []
    
    for key, value in extracted.items():
        if key in expected_fields:
            validated[key] = value
        else:
            unexpected_fields.append(key)
    
    if unexpected_fields:
        print(f"[WARN] Removed unexpected fields for {doc_type}: {unexpected_fields}")
    
    return validated


def _validate_and_fix_name_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and fix name field extraction to prevent firstName/middleName/lastName swapping.
    
    Common issues fixed:
    - middleName mistakenly placed in lastName
    - lastName mistakenly placed in middleName
    - firstName and lastName swapped
    
    Heuristics:
    - If lastName looks like a middle name (single letter or very short), it might be swapped
    - If middleName is longer than lastName, they might be swapped
    """
    first = data.get("firstName", "").strip() if data.get("firstName") else ""
    middle = data.get("middleName", "").strip() if data.get("middleName") else ""
    last = data.get("lastName", "").strip() if data.get("lastName") else ""
    
    # If we don't have name fields, nothing to validate
    if not first and not middle and not last:
        return data
    
    # Case 1: middleName is much longer than lastName (likely swapped)
    # Example: middleName="MCKINLEY", lastName="DEAN" → should be middleName="DEAN", lastName="MCKINLEY"
    if middle and last and len(middle) > len(last) + 3:
        print(f"[NAME FIX] Detected potential swap: middleName='{middle}' (len={len(middle)}) > lastName='{last}' (len={len(last)})")
        print(f"[NAME FIX] Swapping middleName ↔ lastName")
        data["middleName"] = last
        data["lastName"] = middle
        return data
    
    # Case 2: lastName is a single letter or very short (likely a middle initial)
    # Example: firstName="DENNIS", lastName="D", middleName="" → should be firstName="DENNIS", middleName="D", lastName=""
    if last and len(last) <= 2 and not middle:
        print(f"[NAME FIX] Detected lastName='{last}' looks like middle initial (len={len(last)})")
        print(f"[NAME FIX] Moving lastName → middleName")
        data["middleName"] = last
        data["lastName"] = ""
        return data
    
    # Case 3: middleName is empty but we have 3+ words in firstName
    # Example: firstName="DENNIS DEAN MCKINLEY", middleName="", lastName="" → split it
    if first and not middle and not last and len(first.split()) >= 3:
        parts = first.split()
        print(f"[NAME FIX] Detected multiple names in firstName: '{first}'")
        print(f"[NAME FIX] Splitting: firstName='{parts[0]}', middleName='{parts[1]}', lastName='{' '.join(parts[2:])}'")
        data["firstName"] = parts[0]
        data["middleName"] = parts[1]
        data["lastName"] = " ".join(parts[2:])
        return data
    
    # Case 4: firstName and lastName both present, but lastName looks like a first name
    # This is harder to detect automatically, so we'll skip for now
    
    return data

def _fallback_extraction(normalized_input: Dict[str, Any], fields: list[str]) -> Dict[str, Any]:
    """
    Fallback extraction method when GPT fails.
    Performs case-insensitive field matching from normalized input.
    """
    flat = {k.lower(): v for k, v in normalized_input.items()}
    result = {}
    
    for field in fields:
        field_lower = field.lower()
        # Try exact match first
        if field_lower in flat:
            result[field] = flat[field_lower]
        else:
            # Try common variations
            variations = [
                field_lower.replace("_", ""),
                field_lower.replace("_", " "),
                field_lower.replace(" ", ""),
                field_lower.replace(" ", "_"),
            ]
            
            found = False
            for variation in variations:
                if variation in flat and not _is_empty_value(flat[variation]):
                    result[field] = flat[variation]
                    found = True
                    break
            
            if not found:
                result[field] = ""
    
    return result


# ----------------------------------------------------------------------
# GPT Extraction with Multi-Language Support
# ----------------------------------------------------------------------
def extract_fields_with_gpt(input_json: Dict[str, Any], doc_type: str) -> Dict[str, Any]:
    """
    Use Claude to strictly extract only the fields defined for the specific document type.
    Ensures proper field extraction according to DOC_FIELDS for verification purposes.
    
    MULTI-LANGUAGE SUPPORT:
    - Extracts text in ALL detected scripts (Devanagari, Tamil, Telugu, etc.)
    - Uses suffix convention: fieldName_LanguageCode (e.g., holderName_Hi, holderName_Ta)
    - Preserves original script text without translation
    """
    client = get_bedrock_client()
    
    # Validate doc_type and get corresponding fields
    doc_type_key = (doc_type or "").lower()
    if doc_type_key not in DOC_FIELDS:
        print(f"[WARN] Unknown document type '{doc_type}', using 'other' fields")
        doc_type_key = "other"
    
    fields = DOC_FIELDS[doc_type_key]
    normalized_input = _normalize_keys(input_json)
    
    # Detect if this is an Indian document type
    indian_doc_types = [
        "aadhaar", "pan_card", "voter_id", "indian_passport", "indian_driving_license",
        "ration_card", "nrega_job_card", "gst_certificate", "indian_birth_certificate",
        "indian_marriage_certificate", "caste_certificate", "income_certificate",
        "domicile_certificate", "marksheet", "degree_certificate", "bank_passbook",
        "disability_certificate", "pension_card"
    ]
    is_indian_doc = doc_type_key in indian_doc_types

    # Build multi-language extraction prompt
    multilang_instructions = ""
    if is_indian_doc:
        multilang_instructions = """
MULTI-LANGUAGE EXTRACTION (CRITICAL FOR INDIAN DOCUMENTS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Indian documents contain text in MULTIPLE languages/scripts. You MUST:

1. DETECT ALL SCRIPTS present: Devanagari (Hindi/Marathi), Tamil, Telugu, Bengali, 
   Gujarati, Kannada, Malayalam, Odia, Gurmukhi (Punjabi), Arabic (Urdu), etc.

2. For EACH text field, extract BOTH English AND regional language versions:
   - English field: holderName, fatherName, addressLine1, etc.
   - Regional field: holderName_Hi (Hindi), holderName_Ta (Tamil), etc.

3. LANGUAGE CODE SUFFIXES:
   | Language   | Code | Example Field        |
   |------------|------|----------------------|
   | Hindi      | _Hi  | holderName_Hi        |
   | Marathi    | _Mr  | holderName_Mr        |
   | Tamil      | _Ta  | holderName_Ta        |
   | Telugu     | _Te  | holderName_Te        |
   | Bengali    | _Bn  | holderName_Bn        |
   | Gujarati   | _Gu  | holderName_Gu        |
   | Kannada    | _Kn  | holderName_Kn        |
   | Malayalam  | _Ml  | holderName_Ml        |
   | Odia       | _Or  | holderName_Or        |
   | Punjabi    | _Pa  | holderName_Pa        |
   | Assamese   | _As  | holderName_As        |
   | Urdu       | _Ur  | holderName_Ur        |

4. PRESERVE EXACT TEXT - do NOT transliterate or translate
5. Include "detectedLanguages" field listing all languages found

EXAMPLE:
Input: "नाम / Name: राजेश कुमार / RAJESH KUMAR"
Output: {
  "holderName": "RAJESH KUMAR",
  "holderName_Hi": "राजेश कुमार",
  "detectedLanguages": ["english", "hindi"]
}
"""

    prompt = f"""
You are a strict document field extractor for {doc_type.upper()} documents.

CRITICAL REQUIREMENTS:
1. Document Type: {doc_type}
2. ONLY extract these exact fields: {fields}
3. Match field names case-insensitively from the input
4. If a required field is missing or empty, set it to ""
5. DO NOT include any fields not in the required list
6. DO NOT add extra keys, comments, or metadata
7. Return ONLY a valid JSON object with the specified fields
{multilang_instructions}
CRITICAL NAME EXTRACTION RULES:
- For US docs: firstName, middleName, lastName convention
- For Indian docs: holderName (full name), fatherName, motherName, husbandName convention
- If input shows "NAME: RAJESH KUMAR" → extract as holderName="RAJESH KUMAR"
- If input shows bilingual "नाम/Name: राजेश कुमार/RAJESH KUMAR":
  → holderName="RAJESH KUMAR", holderName_Hi="राजेश कुमार"

INDIAN DOCUMENT SPECIFIC RULES:
- Aadhaar: 12-digit number (format: XXXX XXXX XXXX), may have VID (16 digits)
- PAN: 10-char alphanumeric (format: AAAAA9999A)
- Voter ID (EPIC): Alphanumeric (e.g., ABC1234567)
- Dates: Handle DD/MM/YYYY (Indian) and YYYY-MM-DD formats
- Addresses: Capture pinCode (6 digits), district, tehsil, village

Required Fields for {doc_type}:
{json.dumps(fields, indent=2)}

Input OCR Data:
{json.dumps(normalized_input, indent=2, ensure_ascii=False)}

Extract ONLY the required fields listed above. Ignore all other data.
For Indian documents, extract ALL language variants present in the document.
"""

    try:
        system_content = f"""You are a strict field extractor specializing in Indian and international identity documents.
Return ONLY a JSON object containing these fields: {fields}.
For Indian documents, extract text in ALL scripts present (Hindi, Tamil, Telugu, etc.).
Use language code suffixes: _Hi (Hindi), _Ta (Tamil), _Te (Telugu), _Bn (Bengali), etc.
No extra fields, no comments, no explanations.
IMPORTANT: Return ONLY valid JSON, no markdown or code fences."""

        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            system=system_content,
            temperature=0
        )
        
        content = strip_json_code_fences(response)
        raw = json.loads(content)
        
        # For Indian documents, allow additional language variant fields
        if is_indian_doc:
            # Don't strictly validate - allow language variant fields
            validated = {}
            for key, value in raw.items():
                # Allow base fields and language variant fields (with _XX suffix)
                base_key = key.split('_')[0] if '_' in key else key
                if key in fields or base_key in fields or key == "detectedLanguages":
                    validated[key] = value
                elif any(key.startswith(f + "_") for f in fields):
                    validated[key] = value
            raw = validated
        else:
            # Validate that GPT only returned expected fields
            raw = _validate_extracted_fields(raw, fields, doc_type)
        
    except Exception as e:
        # Safe local fallback if the model fails
        print(f"[WARN] GPT extraction failed for {doc_type}, using fallback mode: {e}")
        raw = _fallback_extraction(normalized_input, fields)

    # 1) Ensure full schema (all required fields present)
    complete = _postprocess_complete_schema(raw, fields)
    
    # 2) Add back any language variant fields that were extracted
    if is_indian_doc:
        for key, value in raw.items():
            if key not in complete and not _is_empty_value(value):
                complete[key] = value
    
    # 3) Dedupe alias keys (keep priority key only)
    deduped = _dedupe_aliases(complete)
    # 4) Drop empty values for clean output
    final_clean = _drop_empty_fields(deduped)
    
    # 5) Normalize multilingual field names
    final_clean = normalize_multilingual_fields(final_clean)

    return final_clean


# ----------------------------------------------------------------------
# Helper: map document display/name to doc_type
# INDIAN DOCUMENTS CHECKED FIRST
# ----------------------------------------------------------------------
def _map_document_name_to_doc_type(name: str, classification_type: str = None, fallback: str = "identity_document") -> str:
    """
    Map document name to doc_type with comprehensive coverage.
    INDIAN DOCUMENTS ARE CHECKED FIRST for better detection.
    Uses both document name and classification type for better detection.
    """
    n = (name or "").strip().lower()
    c = (classification_type or "").strip().lower()
    
    # Check both name and classification for better accuracy
    combined = f"{n} {c}".strip()
    
    # ==========================================
    # INDIAN IDENTITY DOCUMENTS (Check First)
    # ==========================================
    
    # Aadhaar Card (आधार)
    if "aadhaar" in combined or "aadhar" in combined or "आधार" in combined or "uidai" in combined:
        return "aadhaar"
    
    # PAN Card (पैन)
    if ("pan" in combined and ("card" in combined or "permanent account" in combined or "number" in combined)) or "पैन" in combined:
        return "pan_card"
    
    # Voter ID / EPIC (मतदाता पहचान पत्र)
    if ("voter" in combined and "id" in combined) or "epic" in combined or "मतदाता" in combined or "election commission" in combined:
        if "registration" not in combined:  # US voter registration is different
            return "voter_id"
    
    # Indian Passport (भारतीय पासपोर्ट)
    if "passport" in combined and ("indian" in combined or "india" in combined or "भारतीय" in combined or "republic of india" in combined):
        return "indian_passport"
    
    # Indian Driving License (ड्राइविंग लाइसेंस)
    if (("driver" in combined or "driving" in combined) and ("license" in combined or "licence" in combined)):
        if "indian" in combined or "india" in combined or "rto" in combined or "dto" in combined or "परिवहन" in combined:
            return "indian_driving_license"
    
    # Ration Card (राशन कार्ड)
    if "ration" in combined and "card" in combined:
        return "ration_card"
    
    # NREGA Job Card (नरेगा जॉब कार्ड)
    if "nrega" in combined or "mgnrega" in combined or ("job" in combined and "card" in combined):
        return "nrega_job_card"
    
    # GST Certificate
    if "gst" in combined or "gstin" in combined or ("goods" in combined and "services" in combined and "tax" in combined):
        return "gst_certificate"
    
    # Indian Birth Certificate (जन्म प्रमाण पत्र)
    if ("birth" in combined and "certificate" in combined) and ("india" in combined or "जन्म" in combined or "registrar" in combined):
        return "indian_birth_certificate"
    
    # Indian Marriage Certificate (विवाह प्रमाण पत्र)
    if ("marriage" in combined and "certificate" in combined) and ("india" in combined or "विवाह" in combined):
        return "indian_marriage_certificate"
    
    # Caste Certificate (जाति प्रमाण पत्र)
    if "caste" in combined and "certificate" in combined:
        return "caste_certificate"
    
    # Income Certificate (आय प्रमाण पत्र)
    if "income" in combined and "certificate" in combined:
        return "income_certificate"
    
    # Domicile Certificate (अधिवास प्रमाण पत्र)
    if "domicile" in combined and "certificate" in combined:
        return "domicile_certificate"
    
    # Marksheet / Educational (अंकपत्र)
    if "marksheet" in combined or "mark sheet" in combined or "अंकपत्र" in combined:
        return "marksheet"
    
    # Degree Certificate
    if "degree" in combined and "certificate" in combined:
        return "degree_certificate"
    
    # Bank Passbook
    if "passbook" in combined or ("bank" in combined and "pass" in combined):
        return "bank_passbook"
    
    # Disability Certificate
    if "disability" in combined and "certificate" in combined:
        return "disability_certificate"
    
    # Pension Card / PPO
    if "pension" in combined or "ppo" in combined:
        return "pension_card"
    
    # ==========================================
    # U.S. IDENTITY DOCUMENTS (Secondary)
    # ==========================================
    
    # U.S. Driving License
    if ("driver" in combined and "license" in combined) or "dl" in combined or "driving_license" in combined:
        if "mobile" in combined or "mdl" in combined:
            return "mobile_drivers_license"
        return "driving_license"
    
    if ("state" in combined and ("id" in combined or "identification" in combined)) and "driver" not in combined:
        if "real" in combined:
            return "real_id"
        return "state_id"
    
    # U.S. Passport Documents
    if "passport" in combined and "card" in combined:
        return "passport_card"
    if "passport" in combined and "card" not in combined:
        return "passport"
    
    # U.S. Birth and Vital Records
    if ("birth" in combined and "certificate" in combined) or "birth_cert" in combined:
        return "birth_certificate"
    if ("marriage" in combined and "certificate" in combined) or "marriage_cert" in combined:
        return "marriage_certificate"
    if ("divorce" in combined and ("decree" in combined or "certificate" in combined)):
        return "divorce_decree"
    
    # Social Security
    if ("social" in combined and "security" in combined) or "ssn" in combined or "ss_card" in combined:
        return "social_security_card"
    
    # Immigration Documents
    if ("permanent" in combined and "resident" in combined) or "green_card" in combined or "prc" in combined or "i-551" in combined:
        return "permanent_resident_card"
    if ("naturalization" in combined and "certificate" in combined) or "n-550" in combined or "n-570" in combined:
        return "certificate_of_naturalization"
    if ("citizenship" in combined and "certificate" in combined) or "n-560" in combined or "n-561" in combined:
        return "certificate_of_citizenship"
    if ("employment" in combined and "authorization" in combined) or "ead" in combined or "i-766" in combined:
        return "employment_authorization_document"
    if "i-94" in combined or ("arrival" in combined and "departure" in combined):
        return "form_i94"
    if ("visa" in combined and ("us" in combined or "american" in combined)) or "h1b" in combined or "h-1b" in combined:
        if "h1b" in combined or "h-1b" in combined:
            return "h1b_visa"
        return "us_visa"
    if ("reentry" in combined and "permit" in combined) or "i-327" in combined:
        return "reentry_permit"
    
    # Military and Government IDs
    if ("military" in combined and "id" in combined) or "cac" in combined or "common_access" in combined:
        return "military_id"
    if ("veteran" in combined and "id" in combined) or "vic" in combined:
        return "veteran_id"
    if ("tribal" in combined and "id" in combined) or "tribal_card" in combined:
        return "tribal_id"
    if ("global" in combined and "entry" in combined) or "nexus" in combined:
        return "global_entry_card"
    if ("tsa" in combined and "precheck" in combined) or "precheck" in combined:
        return "tsa_precheck_card"
    if ("voter" in combined and ("registration" in combined or "card" in combined)):
        return "voter_registration"
    
    # Professional and Educational
    if ("professional" in combined and "license" in combined) or ("license" in combined and any(prof in combined for prof in ["medical", "legal", "contractor", "nursing", "teaching"])):
        return "professional_license"
    if ("student" in combined and "id" in combined) or "student_card" in combined:
        return "student_id"
    
    # Financial and Proof Documents
    if ("utility" in combined and "bill" in combined) or any(util in combined for util in ["electric", "gas", "water", "internet", "cable"]):
        return "utility_bill"
    if ("lease" in combined and "agreement" in combined) or "rental_agreement" in combined:
        return "lease_agreement"
    if ("bank" in combined and "statement" in combined) or "account_statement" in combined:
        return "bank_statement"
    if ("insurance" in combined and "card" in combined) or any(ins in combined for ins in ["health_insurance", "auto_insurance"]):
        return "insurance_card"
    if ("voided" in combined and "check" in combined) or "void_check" in combined:
        return "voided_check"
    if ("direct" in combined and "deposit" in combined) or "dd_form" in combined:
        return "direct_deposit"
    
    # Consular and International
    if ("consular" in combined and "id" in combined) or "matricula" in combined:
        return "consular_id"
    
    # Digital IDs
    if ("digital" in combined and "id" in combined) or any(platform in combined for platform in ["id.me", "login.gov"]):
        return "digital_id"
    
    # If it's any kind of identity document but we can't determine the specific type
    if any(identity_term in combined for identity_term in ["id", "identification", "license", "card", "certificate", "document"]):
        return "identity_document"
    
    # Final fallback
    return fallback


# ----------------------------------------------------------------------
# Main pipeline node
# ----------------------------------------------------------------------
def Extract(state: PipelineState) -> PipelineState:
    if state.classification is None or state.ocr is None:
        raise ValueError("OCR or Classification missing; run previous nodes first.")

    log_agent_event(state, "Document Data Extraction", "start")
    message = state.classification.message

    document_name = state.ocr.document_name or ""
    classification_type = state.classification.detected_doc_type or ""
    mapped_doc_type = _map_document_name_to_doc_type(
        document_name, 
        classification_type=classification_type, 
        fallback="identity_document"
    )
    
    # Log document type detection for debugging
    print(f"[INFO] Document detection - Name: '{document_name}', Classification: '{classification_type}', Final Type: '{mapped_doc_type}'")

    ocr_struct = state.ocr.ocr_json or {}
    page1 = ocr_struct.get("1") if isinstance(ocr_struct, dict) else None
    input_for_cleaner = page1 if isinstance(page1, dict) else ocr_struct

    # --- field extraction (strict clean) ---
    try:
        cleaned = extract_fields_with_gpt(input_for_cleaner, mapped_doc_type)
        
        # Validate and fix name fields (firstName, middleName, lastName)
        cleaned = _validate_and_fix_name_fields(cleaned)
        
        # Validate that we have the expected fields for this document type
        expected_fields = DOC_FIELDS.get(mapped_doc_type.lower(), DOC_FIELDS["other"])
        extracted_fields = list(cleaned.keys())
        
        print(f"[INFO] Expected fields for {mapped_doc_type}: {len(expected_fields)}")
        print(f"[INFO] Successfully extracted fields: {len(extracted_fields)}")
        
        if not cleaned or len(cleaned) == 0:
            print(f"[WARN] No fields extracted for {mapped_doc_type}")
            
    except Exception as e:
        print(f"[ERROR] Field extraction failed for {mapped_doc_type}: {str(e)}")
        cleaned = {"error": str(e), "document_type": mapped_doc_type}

    print("PASS" if state.classification.passed else "FAIL")
    if message:
        print(message)
    # Print the already-cleaned JSON (has no empty fields, no dup aliases)
    try:
        print(json.dumps(cleaned, indent=2, ensure_ascii=False))
    except Exception:
        print(str(cleaned))

    state.extraction = ExtractionState(
        passed=state.classification.passed,
        message=message,
        extracted=cleaned,
    )
    log_agent_event(state, "Document Data Extraction", "completed")

    # --- Upload cleaned data to S3 (dynamic path) ---
    key = None
    try:
        s3 = get_s3_client()
        from ..config.settings import S3_BUCKET
        bucket = os.getenv("S3_BUCKET", S3_BUCKET)
        doc_name = (
            (state.ingestion.document_name if getattr(state, "ingestion", None) else None)
            or (state.ocr.document_name if state.ocr else None)
            or "document"
        )
        # Get date from ingestion metadata (same as used during upload)
        fpcid = state.ingestion.FPCID if getattr(state, "ingestion", None) and hasattr(state.ingestion, 'FPCID') else "3363"
        lmrid = state.ingestion.LMRId if getattr(state, "ingestion", None) and hasattr(state.ingestion, 'LMRId') else "1"
        
        # Extract date from ingestion metadata or prefix_parts
        year = month = day = None
        if state.ingestion and hasattr(state.ingestion, '_raw_metadata') and state.ingestion._raw_metadata:
            meta = state.ingestion._raw_metadata
            year = meta.get("year")
            month = meta.get("month") 
            day = meta.get("day")
        
        # If not found in metadata, try prefix_parts
        if not all([year, month, day]) and state.ingestion and hasattr(state.ingestion, 'prefix_parts') and state.ingestion.prefix_parts:
            parts = state.ingestion.prefix_parts
            year = parts.get("year")
            month = parts.get("month")
            day = parts.get("day")
        
        # Fallback to today's date if still not found
        if not all([year, month, day]):
            today = datetime.now(timezone.utc)
            year = year or today.year
            month = month or today.month
            day = day or today.day
            print(f"[WARN] Using fallback date: {year}-{month:02d}-{day:02d}")
        
        # Ensure proper formatting
        year = str(year)
        month = f"{int(month):02d}" if month else "01"
        day = f"{int(day):02d}" if day else "01"
        
        print(f"[DEBUG] Using date from ingestion: {year}-{month}-{day}")
        
        key = (
            f"LMRFileDocNew/{fpcid}/{year}/{month}/{day}/"
            f"{lmrid}/upload/result/result_{doc_name}.json"
        )

        body = json.dumps(cleaned, ensure_ascii=False).encode("utf-8")
        s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json; charset=utf-8")
        print(f"[✓] Extraction result uploaded to s3://{bucket}/{key}")
    except Exception as e:
        print(f"[WARN] Failed to upload extraction result to S3: {e}")

    # --- DO NOT UPDATE DB HERE ---
    # The validation_check_node will handle the final DB update after validation
    # This prevents duplicate rows and ensures only the final status is written
    print("[INFO] Extraction complete - DB will be updated by validation_check_node")

    return state
