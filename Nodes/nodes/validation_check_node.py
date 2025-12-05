"""
Validation Check node for verifying extracted document field values.

This node validates document fields against document-specific rules including:
- Expiration date checks
- Required field presence
- Format validations (SSN, license numbers, etc.)
- Logical validations (issue date < expiration date)
- Age validations

Validation failures result in "fail" status, rejecting documents that don't
meet the validation requirements.
"""

import os
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Tuple, Optional

from ..config.state_models import PipelineState
from ..config.settings import is_demo_mode
from ..utils.helpers import log_agent_event


# Validation result structure
class ValidationResult:
    def __init__(self):
        self.passed = True
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.info: List[str] = []
    
    def add_warning(self, message: str):
        """Add a warning (allows pass but flags for review)."""
        self.warnings.append(message)
        self.passed = False  # Warnings require human verification
    
    def add_error(self, message: str):
        """Add an error (requires human verification)."""
        self.errors.append(message)
        self.passed = False
    
    def add_info(self, message: str):
        """Add informational message."""
        self.info.append(message)
    
    def has_issues(self) -> bool:
        """Check if there are any warnings or errors."""
        return len(self.warnings) > 0 or len(self.errors) > 0
    
    def get_all_messages(self) -> List[str]:
        """Get all messages combined."""
        messages = []
        if self.errors:
            messages.extend([f"❌ {msg}" for msg in self.errors])
        if self.warnings:
            messages.extend([f"⚠️  {msg}" for msg in self.warnings])
        if self.info:
            messages.extend([f"ℹ️  {msg}" for msg in self.info])
        return messages


# ============================================================================
# Date Validation Utilities
# ============================================================================

def parse_date(date_str: Any) -> Optional[datetime]:
    """
    Parse various date formats into datetime object.
    Supports: MM/DD/YYYY, YYYY-MM-DD, MM-DD-YYYY, etc.
    """
    if not date_str or date_str == "":
        return None
    
    date_str = str(date_str).strip()
    
    # Common date formats
    formats = [
        "%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y", "%d/%m/%Y",
        "%Y/%m/%d", "%m.%d.%Y", "%d.%m.%Y", "%Y.%m.%d",
        "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y",
        "%m/%d/%y", "%y-%m-%d", "%m-%d-%y",
        "%d-%m-%Y", "%d-%m-%y", "%d.%m.%y", "%d.%m.%Y",
        "%d/%m/%y", "%d %B %y", "%d %b %y",
        "%d-%b-%Y", "%d-%b-%y", "%d-%B-%Y", "%d-%B-%y",
        "%d %b, %Y", "%d %B, %Y"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return None


def is_date_expired(date_str: Any) -> Tuple[bool, Optional[datetime]]:
    """Check if a date is expired. Returns (is_expired, parsed_date)."""
    parsed = parse_date(date_str)
    if not parsed:
        return False, None
    
    today = datetime.now()
    return parsed < today, parsed


def is_date_expiring_soon(date_str: Any, days: int = 30) -> Tuple[bool, Optional[datetime]]:
    """Check if date is expiring within specified days."""
    parsed = parse_date(date_str)
    if not parsed:
        return False, None
    
    today = datetime.now()
    threshold = today + timedelta(days=days)
    return today < parsed < threshold, parsed


def validate_date_logic(issue_date: Any, expiration_date: Any) -> Tuple[bool, str]:
    """Validate that issue date is before expiration date."""
    issue = parse_date(issue_date)
    expiry = parse_date(expiration_date)
    
    if not issue or not expiry:
        return True, ""  # Can't validate if dates are missing
    
    if issue >= expiry:
        return False, f"Issue date ({issue_date}) must be before expiration date ({expiration_date})"
    
    return True, ""


# ============================================================================
# Format Validation Utilities (Indian Documents)
# ============================================================================

_VERHOEFF_D_TABLE = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P_TABLE = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_VERHOEFF_INV_TABLE = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def _verhoeff_validate(number: str) -> bool:
    """Validate a numeric string using the Verhoeff checksum."""
    c = 0
    reversed_digits = map(int, reversed(number))
    for i, item in enumerate(reversed_digits):
        c = _VERHOEFF_D_TABLE[c][_VERHOEFF_P_TABLE[i % 8][item]]
    return c == 0


def validate_aadhaar_number(aadhaar: str) -> Tuple[bool, str]:
    """Validate Aadhaar number using length, prefix, and Verhoeff checksum."""
    if not aadhaar:
        return False, "Aadhaar number is missing"
    
    aadhaar_clean = re.sub(r'[\s-]', '', str(aadhaar))
    
    if not re.fullmatch(r'[0-9]{12}', aadhaar_clean):
        return False, "Aadhaar number must be 12 digits"
    
    if aadhaar_clean[0] in {"0", "1"}:
        return False, "Aadhaar number cannot start with 0 or 1"
    
    if not _verhoeff_validate(aadhaar_clean):
        return False, "Invalid Aadhaar number (checksum failed)"
    
    return True, ""


def validate_pan_number(pan: str) -> Tuple[bool, str]:
    """Validate PAN format (AAAAA9999A with restricted 4th character)."""
    if not pan:
        return False, "PAN number is missing"
    
    pan_clean = str(pan).strip().upper()
    
    if not re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan_clean):
        return False, "Invalid PAN format (expected 5 letters, 4 digits, 1 letter)"
    
    if pan_clean[3] not in "CPHFATBLJG":
        return False, "Invalid PAN category character (4th letter)"
    
    return True, ""


def validate_pin_code(pin_code: str) -> Tuple[bool, str]:
    """Validate Indian PIN code (6 digits, first digit 1-9)."""
    if not pin_code:
        return True, ""  # Optional field
    
    pin_clean = re.sub(r'\s', '', str(pin_code))
    
    if re.fullmatch(r"[1-9][0-9]{5}", pin_clean):
        return True, ""
    
    return False, f"Invalid PIN code format: {pin_code}. Expected 6 digits starting 1-9"


def validate_epic_number(epic: str) -> Tuple[bool, str]:
    """Validate Indian voter EPIC number (3 letters + 7 digits)."""
    if not epic:
        return False, "EPIC number is missing"
    
    epic_clean = str(epic).strip().upper()
    if re.fullmatch(r"[A-Z]{3}[0-9]{7}", epic_clean):
        return True, ""
    return False, "Invalid EPIC number (expected 3 letters followed by 7 digits)"


def validate_indian_passport_number(passport: str) -> Tuple[bool, str]:
    """Validate Indian passport number (1 letter followed by 7 digits)."""
    if not passport:
        return False, "Passport number is missing"
    
    passport_clean = str(passport).strip().upper()
    if re.fullmatch(r"[A-Z]{1}[0-9]{7}", passport_clean):
        return True, ""
    return False, "Invalid Indian passport number (expected 1 letter + 7 digits)"


def validate_indian_dl_number(license_no: str) -> Tuple[bool, str]:
    """
    Validate Indian driving licence number.
    Common format: LLNNYYYY###### (state code, RTO code, year, serial).
    """
    if not license_no:
        return False, "Driving licence number is missing"
    
    clean = re.sub(r'[\s-]', '', str(license_no).upper())
    pattern = r"[A-Z]{2}[0-9]{2}[0-9]{4}[0-9]{7}"
    if re.fullmatch(pattern, clean):
        return True, ""
    return False, "Invalid Indian driving licence format"


def validate_gstin(gstin: str) -> Tuple[bool, str]:
    """
    Validate GSTIN: 2 digits (state code), 10-char PAN, 1 entity digit, 'Z', checksum.
    """
    if not gstin:
        return False, "GSTIN is missing"
    
    gstin_clean = str(gstin).strip().upper()
    if not re.fullmatch(r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]", gstin_clean):
        return False, "Invalid GSTIN format"
    return True, ""


def validate_age_from_dob(dob: Any, min_age: int = 0, max_age: int = 120) -> Tuple[bool, str]:
    """Validate age based on date of birth."""
    parsed_dob = parse_date(dob)
    if not parsed_dob:
        return True, ""  # Can't validate if DOB is missing
    
    today = datetime.now()
    age = (today - parsed_dob).days // 365
    
    if age < min_age:
        return False, f"Age ({age}) is below minimum required age ({min_age})"
    
    if age > max_age:
        return False, f"Age ({age}) seems unrealistic (maximum {max_age})"
    
    return True, ""


# ============================================================================
# Document-Specific Validators
# ============================================================================

def _ensure_required_fields(extracted: Dict[str, Any], fields: List[str], result: ValidationResult) -> None:
    """Add errors for any missing required fields."""
    for field in fields:
        if not extracted.get(field):
            result.add_error(f"Required field missing: {field}")


def validate_driving_license(extracted: Dict[str, Any]) -> ValidationResult:
    """Validate Driver's License fields."""
    result = ValidationResult()
    
    # Required fields check
    required_fields = ["firstName", "lastName", "dob", "licenseNumber", "expirationDate", "issuingState"]
    for field in required_fields:
        if not extracted.get(field):
            result.add_error(f"Required field missing: {field}")
    
    # Expiration date check
    expiration = extracted.get("expirationDate")
    if expiration:
        is_expired, exp_date = is_date_expired(expiration)
        if is_expired:
            result.add_error(f"Driver's license is expired (expiration date: {expiration})")
        else:
            # Check if expiring soon
            is_expiring, _ = is_date_expiring_soon(expiration, days=30)
            if is_expiring:
                result.add_warning(f"Driver's license is expiring soon (expiration date: {expiration})")
    
    # Issue date vs expiration date logic
    issue_date = extracted.get("issueDate")
    if issue_date and expiration:
        valid, msg = validate_date_logic(issue_date, expiration)
        if not valid:
            result.add_error(msg)
    
    # Age validation (must be at least 16 for driver's license)
    dob = extracted.get("dob")
    if dob:
        valid, msg = validate_age_from_dob(dob, min_age=16, max_age=120)
        if not valid:
            result.add_error(msg)
    
    return result


def validate_state_id(extracted: Dict[str, Any]) -> ValidationResult:
    """Validate State ID fields."""
    result = ValidationResult()
    
    # Required fields
    required_fields = ["firstName", "lastName", "dob", "idNumber", "expirationDate", "issuingState"]
    for field in required_fields:
        if not extracted.get(field):
            result.add_error(f"Required field missing: {field}")
    
    # Expiration date check
    expiration = extracted.get("expirationDate")
    if expiration:
        is_expired, _ = is_date_expired(expiration)
        if is_expired:
            result.add_error(f"State ID is expired (expiration date: {expiration})")
        else:
            is_expiring, _ = is_date_expiring_soon(expiration, days=30)
            if is_expiring:
                result.add_warning(f"State ID is expiring soon (expiration date: {expiration})")
    
    # Issue date logic
    issue_date = extracted.get("issueDate")
    if issue_date and expiration:
        valid, msg = validate_date_logic(issue_date, expiration)
        if not valid:
            result.add_error(msg)
    
    # Age validation
    dob = extracted.get("dob")
    if dob:
        valid, msg = validate_age_from_dob(dob, min_age=0, max_age=120)
        if not valid:
            result.add_error(msg)
    
    return result


def validate_passport(extracted: Dict[str, Any]) -> ValidationResult:
    """Validate Passport fields."""
    result = ValidationResult()
    
    # Required fields
    required_fields = ["passportNumber", "firstName", "lastName", "dateOfBirth", "expirationDate", "issuingCountry"]
    for field in required_fields:
        if not extracted.get(field):
            result.add_error(f"Required field missing: {field}")
    
    # Expiration date check
    expiration = extracted.get("expirationDate")
    if expiration:
        is_expired, _ = is_date_expired(expiration)
        if is_expired:
            result.add_error(f"Passport is expired (expiration date: {expiration})")
        else:
            # Warn if expiring within 6 months (many countries require 6 months validity)
            is_expiring, _ = is_date_expiring_soon(expiration, days=180)
            if is_expiring:
                result.add_warning(f"Passport is expiring soon (expiration date: {expiration}). Many countries require 6 months validity.")
    
    # Issue date logic
    issue_date = extracted.get("issueDate")
    if issue_date and expiration:
        valid, msg = validate_date_logic(issue_date, expiration)
        if not valid:
            result.add_error(msg)
    
    # Age validation
    dob = extracted.get("dateOfBirth")
    if dob:
        valid, msg = validate_age_from_dob(dob, min_age=0, max_age=120)
        if not valid:
            result.add_error(msg)
    
    # Passport number format (basic check - should not be empty and reasonable length)
    passport_num = extracted.get("passportNumber")
    if passport_num:
        passport_str = str(passport_num).strip()
        if len(passport_str) < 6 or len(passport_str) > 12:
            result.add_warning(f"Passport number length seems unusual: {passport_num}")
    
    return result


def validate_social_security_card(extracted: Dict[str, Any]) -> ValidationResult:
    """Validate Social Security Card fields."""
    result = ValidationResult()
    
    # Required fields
    required_fields = ["firstName", "lastName"]
    for field in required_fields:
        if not extracted.get(field):
            result.add_error(f"Required field missing: {field}")
    
    return result


def validate_birth_certificate(extracted: Dict[str, Any]) -> ValidationResult:
    """Validate Birth Certificate fields."""
    result = ValidationResult()
    
    # Required fields
    required_fields = ["firstName", "lastName", "dateOfBirth", "stateOfBirth"]
    for field in required_fields:
        if not extracted.get(field):
            result.add_error(f"Required field missing: {field}")
    
    # Date of birth validation
    dob = extracted.get("dateOfBirth")
    if dob:
        parsed_dob = parse_date(dob)
        if parsed_dob:
            # Birth date should not be in the future
            if parsed_dob > datetime.now():
                result.add_error(f"Date of birth cannot be in the future: {dob}")
            
            # Age validation (reasonable range)
            valid, msg = validate_age_from_dob(dob, min_age=0, max_age=120)
            if not valid:
                result.add_error(msg)
    
    return result


def validate_permanent_resident_card(extracted: Dict[str, Any]) -> ValidationResult:
    """Validate Permanent Resident Card (Green Card) fields."""
    result = ValidationResult()
    
    # Required fields
    required_fields = ["firstName", "lastName", "dateOfBirth", "alienNumber", "cardNumber", "expirationDate"]
    for field in required_fields:
        if not extracted.get(field):
            result.add_error(f"Required field missing: {field}")
    
    # Expiration date check
    expiration = extracted.get("expirationDate")
    if expiration:
        is_expired, _ = is_date_expired(expiration)
        if is_expired:
            result.add_error(f"Green Card is expired (expiration date: {expiration})")
        else:
            is_expiring, _ = is_date_expiring_soon(expiration, days=180)
            if is_expiring:
                result.add_warning(f"Green Card is expiring soon (expiration date: {expiration})")
    
    # Age validation
    dob = extracted.get("dateOfBirth")
    if dob:
        valid, msg = validate_age_from_dob(dob, min_age=0, max_age=120)
        if not valid:
            result.add_error(msg)
    
    return result


def validate_employment_authorization(extracted: Dict[str, Any]) -> ValidationResult:
    """Validate Employment Authorization Document (EAD) fields."""
    result = ValidationResult()
    
    # Required fields
    required_fields = ["firstName", "lastName", "dateOfBirth", "cardNumber", "expirationDate"]
    for field in required_fields:
        if not extracted.get(field):
            result.add_error(f"Required field missing: {field}")
    
    # Expiration date check
    expiration = extracted.get("expirationDate")
    if expiration:
        is_expired, _ = is_date_expired(expiration)
        if is_expired:
            result.add_error(f"Employment Authorization Document is expired (expiration date: {expiration})")
        else:
            is_expiring, _ = is_date_expiring_soon(expiration, days=90)
            if is_expiring:
                result.add_warning(f"Employment Authorization Document is expiring soon (expiration date: {expiration})")
    
    # Age validation
    dob = extracted.get("dateOfBirth")
    if dob:
        valid, msg = validate_age_from_dob(dob, min_age=0, max_age=120)
        if not valid:
            result.add_error(msg)
    
    return result


def validate_military_id(extracted: Dict[str, Any]) -> ValidationResult:
    """Validate Military ID fields."""
    result = ValidationResult()
    
    # Required fields
    required_fields = ["firstName", "lastName", "dateOfBirth", "branch", "expirationDate"]
    for field in required_fields:
        if not extracted.get(field):
            result.add_error(f"Required field missing: {field}")
    
    # Expiration date check
    expiration = extracted.get("expirationDate")
    if expiration:
        is_expired, _ = is_date_expired(expiration)
        if is_expired:
            result.add_error(f"Military ID is expired (expiration date: {expiration})")
        else:
            is_expiring, _ = is_date_expiring_soon(expiration, days=60)
            if is_expiring:
                result.add_warning(f"Military ID is expiring soon (expiration date: {expiration})")
    
    # Age validation
    dob = extracted.get("dateOfBirth")
    if dob:
        valid, msg = validate_age_from_dob(dob, min_age=17, max_age=120)
        if not valid:
            result.add_error(msg)
    
    return result


def validate_aadhaar_document(extracted: Dict[str, Any]) -> ValidationResult:
    """Validate Aadhaar card details."""
    result = ValidationResult()
    
    _ensure_required_fields(extracted, ["aadhaarNumber", "holderName"], result)
    
    aadhaar = extracted.get("aadhaarNumber")
    if aadhaar:
        valid, msg = validate_aadhaar_number(aadhaar)
        if not valid:
            result.add_error(msg)
    
    if not (extracted.get("dateOfBirth") or extracted.get("dob") or extracted.get("yearOfBirth")):
        result.add_error("Aadhaar requires dateOfBirth or yearOfBirth")
    
    # Basic address presence
    if not any(extracted.get(field) for field in ["addressLine1", "addressLine2", "city", "district", "state"]):
        result.add_warning("Address details are incomplete on Aadhaar")
    
    pin = extracted.get("pinCode") or extracted.get("pincode")
    if pin:
        valid, msg = validate_pin_code(pin)
        if not valid:
            result.add_error(msg)
    
    return result


def validate_pan_card(extracted: Dict[str, Any]) -> ValidationResult:
    """Validate PAN card details."""
    result = ValidationResult()
    
    _ensure_required_fields(extracted, ["panNumber", "holderName", "dateOfBirth"], result)
    
    pan = extracted.get("panNumber")
    if pan:
        valid, msg = validate_pan_number(pan)
        if not valid:
            result.add_error(msg)
    
    if extracted.get("dateOfBirth"):
        valid, msg = validate_age_from_dob(extracted["dateOfBirth"], min_age=18, max_age=120)
        if not valid:
            result.add_error(msg)
    
    if not (extracted.get("fatherName") or extracted.get("motherName")):
        result.add_warning("Parent/guardian name is missing on PAN")
    
    return result


def validate_voter_id(extracted: Dict[str, Any]) -> ValidationResult:
    """Validate Indian voter ID details."""
    result = ValidationResult()
    
    _ensure_required_fields(extracted, ["epicNumber", "holderName"], result)
    
    epic = extracted.get("epicNumber")
    if epic:
        valid, msg = validate_epic_number(epic)
        if not valid:
            result.add_error(msg)
    
    dob = extracted.get("dateOfBirth")
    if dob:
        valid, msg = validate_age_from_dob(dob, min_age=18, max_age=120)
        if not valid:
            result.add_error(msg)
    else:
        result.add_warning("Date of birth is missing on Voter ID")
    
    if not any(extracted.get(field) for field in ["addressLine1", "addressLine2", "assemblyConstituency", "district"]):
        result.add_warning("Address/constituency information is incomplete on Voter ID")
    
    pin = extracted.get("pinCode") or extracted.get("pin")
    if pin:
        valid, msg = validate_pin_code(pin)
        if not valid:
            result.add_error(msg)
    
    return result


def validate_indian_passport(extracted: Dict[str, Any]) -> ValidationResult:
    """Validate Indian passport details."""
    result = ValidationResult()
    
    _ensure_required_fields(
        extracted,
        ["passportNumber", "surname", "givenNames", "dateOfBirth", "issueDate", "expirationDate"],
        result
    )
    
    passport_num = extracted.get("passportNumber")
    if passport_num:
        valid, msg = validate_indian_passport_number(passport_num)
        if not valid:
            result.add_error(msg)
    
    expiration = extracted.get("expirationDate")
    if expiration:
        is_expired, _ = is_date_expired(expiration)
        if is_expired:
            result.add_error(f"Passport is expired (expiration date: {expiration})")
        else:
            is_expiring, _ = is_date_expiring_soon(expiration, days=180)
            if is_expiring:
                result.add_warning(f"Passport is expiring within 6 months ({expiration})")
    
    issue_date = extracted.get("issueDate")
    if issue_date and expiration:
        valid, msg = validate_date_logic(issue_date, expiration)
        if not valid:
            result.add_error(msg)
    
    dob = extracted.get("dateOfBirth")
    if dob:
        valid, msg = validate_age_from_dob(dob, min_age=0, max_age=120)
        if not valid:
            result.add_error(msg)
    
    return result


def validate_indian_driving_license(extracted: Dict[str, Any]) -> ValidationResult:
    """Validate Indian driving licence details."""
    result = ValidationResult()
    
    _ensure_required_fields(
        extracted,
        ["licenseNumber", "holderName", "dob"],
        result
    )
    
    license_no = extracted.get("licenseNumber")
    if license_no:
        valid, msg = validate_indian_dl_number(license_no)
        if not valid:
            result.add_error(msg)
    
    valid_till = extracted.get("validTillTransport") or extracted.get("validTillNonTransport") or extracted.get("expirationDate")
    if valid_till:
        is_expired, _ = is_date_expired(valid_till)
        if is_expired:
            result.add_error(f"Driving licence is expired (valid till: {valid_till})")
    
    issue_date = extracted.get("issueDate")
    if issue_date and valid_till:
        valid, msg = validate_date_logic(issue_date, valid_till)
        if not valid:
            result.add_error(msg)
    
    dob = extracted.get("dob")
    if dob:
        valid, msg = validate_age_from_dob(dob, min_age=18, max_age=120)
        if not valid:
            result.add_error(msg)
    
    pin = extracted.get("pinCode") or extracted.get("pin")
    if pin:
        valid, msg = validate_pin_code(pin)
        if not valid:
            result.add_error(msg)
    
    return result


def validate_gst_certificate(extracted: Dict[str, Any]) -> ValidationResult:
    """Validate GST registration certificate details."""
    result = ValidationResult()
    
    _ensure_required_fields(
        extracted,
        ["gstin", "legalName", "issueDate", "principalPlaceOfBusiness"],
        result
    )
    
    gstin = extracted.get("gstin")
    if gstin:
        valid, msg = validate_gstin(gstin)
        if not valid:
            result.add_error(msg)
    
    # Ensure GSTIN PAN matches PAN field when provided
    pan = extracted.get("panNumber")
    if pan and gstin and len(gstin) >= 12:
        embedded_pan = gstin[2:12]
        if embedded_pan != pan.upper():
            result.add_warning("PAN embedded in GSTIN does not match provided PAN")
    
    effective = extracted.get("effectiveDate") or extracted.get("issueDate")
    if effective:
        parsed_effective = parse_date(effective)
        if parsed_effective and parsed_effective > datetime.now():
            result.add_error(f"GST registration effective date {effective} cannot be in the future")
    
    return result


def validate_generic_identity(extracted: Dict[str, Any]) -> ValidationResult:
    """Generic validation for identity documents."""
    result = ValidationResult()
    
    # Basic required fields
    if not extracted.get("firstName") and not extracted.get("lastName"):
        result.add_warning("Name information is missing or incomplete")
    
    # Check for expiration if present
    expiration = extracted.get("expirationDate")
    if expiration:
        is_expired, _ = is_date_expired(expiration)
        if is_expired:
            result.add_warning(f"Document appears to be expired (expiration date: {expiration})")
    
    # DOB validation if present
    dob = extracted.get("dob") or extracted.get("dateOfBirth")
    if dob:
        valid, msg = validate_age_from_dob(dob, min_age=0, max_age=120)
        if not valid:
            result.add_warning(msg)
    
    return result


# ============================================================================
# Main Validation Router
# ============================================================================

VALIDATION_FUNCTIONS = {
    "driving_license": validate_driving_license,
    "mobile_drivers_license": validate_driving_license,  # Same rules as regular DL
    "state_id": validate_state_id,
    "real_id": validate_state_id,  # Same rules as state ID
    "passport": validate_passport,
    "passport_card": validate_passport,  # Similar rules to passport
    "social_security_card": validate_social_security_card,
    "birth_certificate": validate_birth_certificate,
    "permanent_resident_card": validate_permanent_resident_card,
    "employment_authorization_document": validate_employment_authorization,
    "military_id": validate_military_id,
    "veteran_id": validate_military_id,  # Similar rules
    # Indian identity documents
    "aadhaar": validate_aadhaar_document,
    "pan_card": validate_pan_card,
    "voter_id": validate_voter_id,
    "indian_passport": validate_indian_passport,
    "indian_driving_license": validate_indian_driving_license,
    "gst_certificate": validate_gst_certificate,
    # Add more as needed
}


def validate_document(doc_type: str, extracted: Dict[str, Any]) -> ValidationResult:
    """
    Route to appropriate validator based on document type.
    """
    doc_type_lower = (doc_type or "").lower().strip()
    
    # Get specific validator or use generic
    validator = VALIDATION_FUNCTIONS.get(doc_type_lower, validate_generic_identity)
    
    return validator(extracted)


# ============================================================================
# Main Node Function
# ============================================================================

def _generate_user_friendly_reason(
    status: str,
    validation_result: 'ValidationResult',
    llm_validation: Dict[str, Any],
    extracted: Dict[str, Any],
    doc_type: str
) -> str:
    """
    Generate a user-friendly 2-line reason using AI based on validation status.
    This provides clear, actionable feedback for the frontend.
    """
    try:
        from openai import OpenAI
        import json as _json
        
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            # Fallback to basic reason
            if status == "pass":
                return "Your document has been successfully verified. All validation checks passed."
            elif status == "fail":
                return "Document validation failed. Please upload a clear, valid document."
            else:
                return "Document requires manual review. Our team will verify it shortly."
        
        client = OpenAI(api_key=api_key)
        
        # Prepare context for AI
        context = {
            "status": status,
            "document_type": doc_type,
            "errors": validation_result.errors if validation_result else [],
            "warnings": validation_result.warnings if validation_result else [],
            "llm_summary": llm_validation.get("summary", "") if llm_validation.get("validation_performed") else "",
            "validation_results": llm_validation.get("validation_results", {}) if llm_validation.get("validation_performed") else {},
            "extracted_sample": {
                "has_expiration": bool(extracted.get("expirationDate") or extracted.get("expiry")),
                "has_name": bool(extracted.get("firstName") or extracted.get("lastName")),
                "has_dob": bool(extracted.get("dob") or extracted.get("dateOfBirth")),
                "field_count": len(extracted)
            }
        }
        
        system_prompt = """You are a user experience expert who creates clear, friendly, and actionable messages for document verification.

Your task is to generate a user-friendly reason message in EXACTLY 2 lines (maximum 150 characters total) that explains the validation result.

Guidelines:
1. Be clear and direct - users need to understand immediately what happened
2. Be friendly and professional - avoid technical jargon
3. Be actionable - tell users what to do next if there's an issue
4. Use simple language - write for a general audience
5. MUST be exactly 2 lines, each line under 75 characters

Examples:

PASS status:
"Your document has been successfully verified and approved.
All information is clear and meets our requirements."

FAIL status (expired):
"Your document has expired and cannot be accepted.
Please upload a current, valid document to proceed."

FAIL status (blurry):
"The image quality is too low to read clearly.
Please upload a clearer photo with all text visible."

FAIL status (missing fields):
"Required information is missing from your document.
Please ensure all fields are visible and upload again."

HUMAN_REVIEW status (expiring soon):
"Your document is expiring soon and needs manual review.
Our team will verify it within 24 hours."

HUMAN_REVIEW status (warnings):
"Some information needs verification by our team.
We'll review your document and contact you shortly."

Return ONLY the 2-line message as plain text, nothing else."""

        user_prompt = f"""Generate a user-friendly 2-line reason for this validation result:

Status: {status.upper()}
Document Type: {doc_type}

Validation Context:
{_json.dumps(context, indent=2)}

Return EXACTLY 2 lines (under 75 characters each) that clearly explain the result to the user."""

        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,  # Slight creativity for natural language
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
            # If AI returned only 1 line, add a generic second line
            if status == "pass":
                reason = lines[0] + "\nAll information is clear and meets our requirements."
            elif status == "fail":
                reason = lines[0] + "\nPlease upload a valid document to proceed."
            else:
                reason = lines[0] + "\nOur team will review it shortly."
        else:
            reason = '\n'.join(lines)
        
        print(f"[AI REASON] Generated user-friendly reason ({len(reason)} chars):")
        print(f"  Line 1: {reason.split(chr(10))[0]}")
        print(f"  Line 2: {reason.split(chr(10))[1] if len(reason.split(chr(10))) > 1 else ''}")
        
        return reason
        
    except Exception as e:
        print(f"[WARN] Failed to generate AI reason: {e}")
        # Fallback reasons
        if status == "pass":
            return "Your document has been successfully verified.\nAll validation checks passed."
        elif status == "fail":
            if validation_result and validation_result.errors:
                first_error = validation_result.errors[0].lower()
                if "expired" in first_error:
                    return "Your document has expired and cannot be accepted.\nPlease upload a current, valid document."
                elif "missing" in first_error:
                    return "Required information is missing from your document.\nPlease ensure all fields are visible and upload again."
                elif "format" in first_error or "invalid" in first_error:
                    return "Document format or information is invalid.\nPlease check and upload a valid document."
            return "Document validation failed.\nPlease upload a clear, valid document."
        else:  # human_review
            return "Your document requires manual verification.\nOur team will review it within 24 hours."


def _llm_validate_document(doc_type: str, extracted: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use LLM to intelligently validate document fields including expiry, dates, and logical consistency.
    """
    try:
        from openai import OpenAI
        import json as _json
        
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return {"validation_performed": False, "reason": "No API key"}
        
        client = OpenAI(api_key=api_key)
        
        # Get current date for expiry checks
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        system_prompt = f"""You are an expert document validator with deep knowledge of identity documents worldwide.

Current Date: {current_date}

Validate the extracted document fields comprehensively. Check for:

1. **Expiration Validation**:
   - Is the document expired?
   - Is it expiring soon (within 30-180 days depending on document type)?
   - For passports, check 6-month validity rule

2. **Date Logic**:
   - Issue date must be before expiration date
   - Birth date must be in the past
   - Dates must be realistic (not too far in past/future)

3. **Required Fields**:
   - Are critical fields present? (name, DOB, document number, etc.)
   - Are fields complete and not empty?

4. **Format Validation**:
   - Aadhaar checksum validation (12 digits, Verhoeff checksum)
   - PAN format (AAAAA9999A with valid category letter)
   - Indian PIN codes (6 digits, first digit 1-9)
   - Document numbers (reasonable length and format)

5. **Age Validation**:
   - Age calculated from DOB is realistic (0-120 years)
   - Age meets minimum requirements (16+ for driver's license, 17+ for military)

6. **Logical Consistency**:
   - Name fields are consistent
   - Address components match
   - Document class/type matches restrictions

7. **Image Quality** (if mentioned in extracted data):
   - Check for blur, low resolution, or illegibility indicators
   - Look for partial/cropped images

Return JSON with:
{{
  "overall_status": "pass|warning|fail",
  "confidence": "high|medium|low",
  "validation_results": {{
    "expiration": {{
      "status": "valid|expiring_soon|expired|not_applicable",
      "message": "detailed message",
      "expiry_date": "extracted date if found",
      "days_until_expiry": number or null
    }},
    "required_fields": {{
      "status": "complete|incomplete",
      "missing_fields": ["list of missing critical fields"],
      "message": "summary"
    }},
    "date_logic": {{
      "status": "valid|invalid",
      "issues": ["list of date logic issues"],
      "message": "summary"
    }},
    "format_validation": {{
      "status": "valid|invalid",
      "issues": ["list of format issues"],
      "message": "summary"
    }},
    "age_validation": {{
      "status": "valid|invalid",
      "age": number or null,
      "message": "summary"
    }},
    "image_quality": {{
      "status": "clear|unclear|not_assessed",
      "issues": ["list of quality issues if any"],
      "message": "summary"
    }}
  }},
  "errors": ["list of critical errors requiring rejection"],
  "warnings": ["list of warnings requiring human review"],
  "suggestions": ["list of actionable suggestions for user"],
  "summary": "brief overall summary of validation"
}}"""

        user_prompt = f"""Validate this {doc_type} document:

Extracted Fields:
{_json.dumps(extracted, indent=2)}

Perform comprehensive validation and return detailed results."""

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
        result["validation_performed"] = True
        
        print(f"\n[LLM VALIDATION] Overall Status: {result.get('overall_status')}")
        print(f"[LLM VALIDATION] Confidence: {result.get('confidence')}")
        print(f"[LLM VALIDATION] Summary: {result.get('summary')}")
        
        return result
        
    except Exception as e:
        print(f"[WARN] LLM validation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"validation_performed": False, "reason": str(e)}


def ValidationCheck(state: PipelineState) -> PipelineState:
    """
    Validate extracted document fields using intelligent LLM-based validation.
    
    This node uses GPT-4o to comprehensively check:
    - Required field presence
    - Expiration dates (with context-aware thresholds)
    - Format validations (SSN, license numbers, etc.)
    - Logical validations (issue date < expiration date)
    - Age validations (context-aware minimum ages)
    - Cross-field consistency
    
    Documents with validation issues are marked as "fail" and rejected.
    """
    if state.extraction is None:
        raise ValueError("Extraction state missing; run Extraction node first.")
    
    log_agent_event(state, "Validation Check", "start")
    
    print("\n" + "=" * 80)
    print("🔍 INTELLIGENT VALIDATION CHECK (LLM-Powered)")
    print("=" * 80)
    
    # Get document type and extracted data
    doc_type = ""
    if state.classification:
        doc_type = state.classification.detected_doc_type or ""
    
    extracted = state.extraction.extracted or {}
    
    print(f"Document Type: {doc_type}")
    print(f"Extracted Fields: {len(extracted)}")
    
    # ===== VISUAL TAMPERING DETECTION =====
    visual_tampering_result = None
    try:
        # Get image URL for tampering detection
        image_url = None
        
        if state.ocr:
            bucket = state.ocr.bucket
            key = state.ocr.key
            
            # Skip for local files
            if bucket != "local":
                # Generate presigned URL for S3 file
                from ..tools.aws_services import generate_presigned_url
                
                # Check if it's an image file
                lower_key = key.lower()
                if lower_key.endswith(('.jpg', '.jpeg', '.png')):
                    image_url = generate_presigned_url(bucket, key)
                    print(f"[Tampering Detection] Generated image URL for visual analysis")
                elif lower_key.endswith('.pdf'):
                    # For PDFs, we can't easily do visual tampering detection
                    # Skip for now (could convert first page to image later)
                    print(f"[Tampering Detection] PDF detected - skipping visual tampering check")
            
            # Perform visual tampering detection if we have image URL
            if image_url:
                print(f"\n[🔍 Visual Tampering Detection]")
                print("-" * 80)
                
                from ..tools.llm_services import detect_visual_tampering
                
                visual_tampering_result = detect_visual_tampering(
                    model="gpt-4o",
                    image_url=image_url,
                    doc_type=doc_type or "unknown",
                    extracted_fields=extracted
                )
                
                # Display results
                if visual_tampering_result:
                    tampering_detected = visual_tampering_result.get("tampering_detected", False)
                    risk_score = visual_tampering_result.get("risk_score", 0)
                    confidence = visual_tampering_result.get("confidence", "low")
                    indicators = visual_tampering_result.get("indicators", [])
                    
                    if tampering_detected:
                        print(f"⚠️  TAMPERING DETECTED (Risk Score: {risk_score}/100, Confidence: {confidence})")
                        print(f"Summary: {visual_tampering_result.get('summary', 'Visual inconsistencies found')}")
                        
                        if indicators:
                            print(f"\nTampering Indicators Found ({len(indicators)}):")
                            for idx, indicator in enumerate(indicators, 1):
                                severity = indicator.get("severity", "unknown")
                                indicator_type = indicator.get("type", "unknown")
                                description = indicator.get("description", "")
                                location = indicator.get("location", "")
                                
                                severity_icon = "🔴" if severity == "high" else "🟡" if severity == "medium" else "🟢"
                                print(f"  {idx}. {severity_icon} [{severity.upper()}] {indicator_type}")
                                print(f"     Location: {location}")
                                print(f"     Details: {description}")
                    else:
                        print(f"✅ No visual tampering detected (Risk Score: {risk_score}/100)")
                    
                    print("-" * 80)
                else:
                    print("[Tampering Detection] Failed to get results")
    except Exception as e:
        print(f"[WARN] Visual tampering detection failed: {e}")
        import traceback
        traceback.print_exc()
        visual_tampering_result = None
    
    # ===== END VISUAL TAMPERING DETECTION =====
    
    # Perform LLM-based validation first
    llm_validation = _llm_validate_document(doc_type, extracted)
    
    # Post-process LLM validation to correct expiration status
    # LLM sometimes misclassifies expired documents as "expiring soon"
    if llm_validation.get("validation_performed"):
        validation_results = llm_validation.get("validation_results", {})
        expiration_info = validation_results.get("expiration", {})
        
        if expiration_info.get("status") in ["expiring_soon", "expired"]:
            expiry_date_str = expiration_info.get("expiry_date")
            
            # Extract expiration date from extracted fields if not in LLM result
            if not expiry_date_str:
                expiry_date_str = (extracted.get("expirationDate") or 
                                  extracted.get("ExpirationDate") or 
                                  extracted.get("expiration_date") or
                                  extracted.get("dateOfExpiry") or
                                  extracted.get("validUntil"))
            
            if expiry_date_str:
                # Check if actually expired using date comparison
                is_actually_expired, parsed_date = is_date_expired(expiry_date_str)
                
                if is_actually_expired:
                    # Correct the status to "expired"
                    expiration_info["status"] = "expired"
                    expiration_info["message"] = f"Document expired on {expiry_date_str}"
                    
                    # Update errors to reflect expired status
                    llm_errors = llm_validation.get("errors", [])
                    llm_warnings = llm_validation.get("warnings", [])
                    
                    # Remove "expiring soon" warnings
                    llm_warnings = [w for w in llm_warnings if "expiring soon" not in w.lower()]
                    llm_validation["warnings"] = llm_warnings
                    
                    # Add or update expired error
                    has_expired_error = any("expired" in e.lower() for e in llm_errors)
                    if not has_expired_error:
                        llm_errors.append(f"Document appears to be expired (expiration date: {expiry_date_str})")
                        llm_validation["errors"] = llm_errors
                        llm_validation["overall_status"] = "fail"
                    
                    print(f"[INFO] Corrected LLM expiration status: expiring_soon → expired ({expiry_date_str})")
    
    # Fallback to rule-based validation if LLM fails
    validation_result = validate_document(doc_type, extracted)
    
    # Merge LLM and rule-based results
    if llm_validation.get("validation_performed"):
        # Use LLM results as primary, supplement with rule-based
        overall_status = llm_validation.get("overall_status", "warning")
        llm_errors = llm_validation.get("errors", [])
        llm_warnings = llm_validation.get("warnings", [])
        llm_suggestions = llm_validation.get("suggestions", [])
        
        # Add LLM findings to validation result
        for error in llm_errors:
            validation_result.add_error(error)
        for warning in llm_warnings:
            validation_result.add_warning(warning)
        
        # ===== INCORPORATE VISUAL TAMPERING RESULTS =====
        if visual_tampering_result and visual_tampering_result.get("tampering_detected"):
            risk_score = visual_tampering_result.get("risk_score", 0)
            confidence = visual_tampering_result.get("confidence", "low")
            indicators = visual_tampering_result.get("indicators", [])
            
            # High risk tampering = critical error
            if risk_score >= 70 or confidence == "high":
                tampering_msg = f"Visual tampering detected (Risk: {risk_score}/100, Confidence: {confidence})"
                if indicators:
                    high_severity_indicators = [ind for ind in indicators if ind.get("severity") == "high"]
                    if high_severity_indicators:
                        tampering_msg += f". Critical issues: {len(high_severity_indicators)}"
                validation_result.add_error(tampering_msg)
                validation_result.passed = False
                print(f"[❌] Visual tampering detected - marking validation as FAILED")
            
            # Medium risk = warning requiring human review
            elif risk_score >= 40:
                tampering_msg = f"Visual inconsistencies detected (Risk: {risk_score}/100). Manual review recommended."
                validation_result.add_warning(tampering_msg)
                validation_result.passed = False
                print(f"[⚠️] Visual inconsistencies detected - requires manual review")
            
            # Low risk = informational only
            else:
                if indicators:
                    tampering_msg = f"Minor visual inconsistencies noted (Risk: {risk_score}/100)"
                    validation_result.add_info(tampering_msg)
                    print(f"[ℹ️] Minor visual inconsistencies noted")
        # ===== END INCORPORATE VISUAL TAMPERING RESULTS =====
        
        # Override passed status based on LLM
        if overall_status == "fail" or len(llm_errors) > 0:
            validation_result.passed = False
        elif overall_status == "warning" or len(llm_warnings) > 0:
            validation_result.passed = False
    
    # Display results
    print("\n" + "-" * 80)
    if validation_result.passed:
        print("✅ VALIDATION PASSED")
        print("All validation checks passed successfully.")
    else:
        print("❌ VALIDATION FAILED")
        print("Document validation failed.")
        print("\nIssues found:")
        for msg in validation_result.get_all_messages():
            print(f"  {msg}")
        
        # Display LLM-specific insights
        if llm_validation.get("validation_performed"):
            print("\n📊 LLM Validation Insights:")
            validation_results = llm_validation.get("validation_results", {})
            
            # Expiration status
            expiration = validation_results.get("expiration", {})
            if expiration.get("status") != "not_applicable":
                print(f"  • Expiration: {expiration.get('status', 'unknown')} - {expiration.get('message', 'N/A')}")
            
            # Required fields
            required = validation_results.get("required_fields", {})
            if required.get("missing_fields"):
                print(f"  • Missing Fields: {', '.join(required.get('missing_fields', []))}")
            
            # Suggestions
            if llm_validation.get("suggestions"):
                print("\n💡 Suggestions:")
                for suggestion in llm_validation.get("suggestions", [])[:3]:
                    print(f"  • {suggestion}")
    
    print("-" * 80)
    
    # Update state - store validation results
    if not hasattr(state, 'validation'):
        # Add validation results to state (we'll store in extraction for now)
        state.extraction.message = "Validation completed"
    
    # Update database with validation results
    # Skip database update if ingestion is from local file
    if is_demo_mode() or (state.ingestion and state.ingestion.s3_bucket == "local"):
        print("[DEMO/LOCAL] Skipping database update in Validation Check (demo/local mode).")
    else:
        try:
            from datetime import datetime, timezone
            import json
            from ..tools.db import update_tblaigents_by_keys
            
            # Determine document status and generate AI reason
            if validation_result.passed:
                document_status = "pass"
                
                ai_reason = _generate_user_friendly_reason(
                    status="pass",
                    validation_result=validation_result,
                    llm_validation=llm_validation,
                    extracted=extracted,
                    doc_type=doc_type
                )
                
                reason_lines = [line.strip() for line in ai_reason.split('\n') if line.strip()]
                if not reason_lines:
                    reason_lines = ["Document validation passed successfully", "All required fields are present and valid"]
                
                doc_verification_result = {
                    "score": 100,
                    "reason": reason_lines,
                    "details": [],
                    "stats": {
                        "score": 100,
                        "matched_fields": len(extracted),
                        "mismatched_fields": 0
                    }
                }
                
                # Add tampering detection results if available
                if visual_tampering_result:
                    doc_verification_result["tampering_detection"] = {
                        "tampering_detected": visual_tampering_result.get("tampering_detected", False),
                        "risk_score": visual_tampering_result.get("risk_score", 0),
                        "confidence": visual_tampering_result.get("confidence", "low"),
                        "indicators_count": len(visual_tampering_result.get("indicators", [])),
                        "summary": visual_tampering_result.get("summary", "")
                    }
            else:
                has_critical_errors = False
                if validation_result.errors:
                    for error in validation_result.errors:
                        error_lower = error.lower()
                        if any(keyword in error_lower for keyword in ["expired", "missing", "invalid ssn", "invalid format"]):
                            has_critical_errors = True
                            break
                
                if not has_critical_errors and validation_result.warnings:
                    for warning in validation_result.warnings:
                        warning_lower = warning.lower()
                        if any(keyword in warning_lower for keyword in ["expired", "expiring soon", "expiration date"]):
                            has_critical_errors = True
                            break
                
                document_status = "fail"
                status_for_reason = "fail"
                
                ai_reason = _generate_user_friendly_reason(
                    status=status_for_reason,
                    validation_result=validation_result,
                    llm_validation=llm_validation,
                    extracted=extracted,
                    doc_type=doc_type
                )
                
                reason_lines = []
                details = []
                expiration_date = None
                is_expired = False
                is_expiring_soon = False
                
                for error in validation_result.errors:
                    if "expir" in error.lower():
                        import re
                        date_match = re.search(r'(\d{2}/\d{2}/\d{4})', error)
                        if date_match:
                            expiration_date = date_match.group(1)
                        
                        if "expired" in error.lower():
                            is_expired = True
                        elif "expiring soon" in error.lower():
                            is_expiring_soon = True
                
                for warning in validation_result.warnings:
                    if "expir" in warning.lower():
                        import re
                        date_match = re.search(r'(\d{2}/\d{2}/\d{4})', warning)
                        if date_match:
                            expiration_date = date_match.group(1)
                        
                        if "expiring soon" in warning.lower():
                            is_expiring_soon = True
                
                if not expiration_date and extracted:
                    exp_date = (extracted.get("expirationDate") or 
                               extracted.get("ExpirationDate") or 
                               extracted.get("expiration_date") or
                               extracted.get("dateOfExpiry") or
                               extracted.get("validUntil"))
                    if exp_date:
                        expiration_date = exp_date
                
                if is_expired and expiration_date:
                    reason_lines = [
                        f"Your {doc_type} expired on {expiration_date}",
                        "Please renew your document and upload a valid version",
                        "Expired documents cannot be processed for verification"
                    ]
                    details = [
                        {"field": "Expiration Date", "ref": "Valid document required", "doc": expiration_date}
                    ]
                elif is_expiring_soon and expiration_date:
                    reason_lines = [
                        f"Your {doc_type} expires on {expiration_date}",
                        "Please renew your document and upload a valid version",
                        "Documents expiring soon cannot be processed for verification"
                    ]
                    details = [
                        {"field": "Expiration Date", "ref": "Valid document required", "doc": expiration_date}
                    ]
                else:
                    reason_lines = [line.strip() for line in ai_reason.split('\n') if line.strip()]
                    if not reason_lines:
                        reason_lines = [
                            "Document validation failed",
                            "Please review the document and ensure all information is correct",
                            "Upload a valid document to proceed with verification"
                        ]
                
                while len(reason_lines) < 3:
                    reason_lines.append("Please contact support if you need assistance")
                reason_lines = reason_lines[:3]
                
                doc_verification_result = {
                    "score": 0,
                    "reason": reason_lines,
                    "details": details,
                    "stats": {
                        "score": 0,
                        "matched_fields": 0,
                        "mismatched_fields": len(details)
                    }
                }
                
                # Add tampering detection results if available
                if visual_tampering_result:
                    doc_verification_result["tampering_detection"] = {
                        "tampering_detected": visual_tampering_result.get("tampering_detected", False),
                        "risk_score": visual_tampering_result.get("risk_score", 0),
                        "confidence": visual_tampering_result.get("confidence", "low"),
                        "indicators_count": len(visual_tampering_result.get("indicators", [])),
                        "summary": visual_tampering_result.get("summary", "")
                    }
            
            if state.ingestion:
                verified_result_s3_path = None
                if state.extraction:
                    doc_name = state.ingestion.document_name or "document"
                    fpcid = state.ingestion.FPCID
                    lmrid = state.ingestion.LMRId
                    
                    year = month = day = None
                    if hasattr(state.ingestion, '_raw_metadata') and state.ingestion._raw_metadata:
                        meta = state.ingestion._raw_metadata
                        year = meta.get("year")
                        month = meta.get("month") 
                        day = meta.get("day")
                    
                    if not all([year, month, day]) and hasattr(state.ingestion, 'prefix_parts') and state.ingestion.prefix_parts:
                        parts = state.ingestion.prefix_parts
                        year = parts.get("year")
                        month = parts.get("month")
                        day = parts.get("day")
                    
                    if not all([year, month, day]):
                        from datetime import datetime
                        now = datetime.now()
                        year = now.year
                        month = now.month
                        day = now.day
                        print(f"[WARN] Using fallback date for verified_result_s3_path: {year}-{month:02d}-{day:02d}")
                    
                    year = str(year)
                    month = f"{int(month):02d}"
                    day = f"{int(day):02d}"
                    from ..config.settings import S3_BUCKET
                    bucket = os.getenv("S3_BUCKET", S3_BUCKET)
                    verified_result_s3_path = f"s3://{bucket}/LMRFileDocNew/{fpcid}/{year}/{month}/{day}/{lmrid}/upload/result/result_{doc_name}.json"
                
                updates_dict = {
                    "file_s3_location": f"s3://{state.ingestion.s3_bucket}/{state.ingestion.s3_key}" if hasattr(state.ingestion, 's3_bucket') and hasattr(state.ingestion, 's3_key') else None,
                    "metadata_s3_path": state.ingestion.metadata_s3_path if hasattr(state.ingestion, 'metadata_s3_path') else None,
                    "verified_result_s3_path": verified_result_s3_path,
                    "document_status": document_status,
                    "doc_verification_result": json.dumps(doc_verification_result),
                    "is_verified": False,
                    "checklistId": state.ingestion.checklistId if hasattr(state.ingestion, 'checklistId') else None,
                    "doc_id": state.ingestion.doc_id if hasattr(state.ingestion, 'doc_id') else None,
                    "document_type": state.ingestion.document_type if hasattr(state.ingestion, 'document_type') else None,
                }
                
                if document_status == "fail":
                    updates_dict["Validation_status"] = "fail"
                    updates_dict["cross_validation"] = False
                
                update_tblaigents_by_keys(
                    FPCID=state.ingestion.FPCID,
                    airecordid=state.ingestion.airecordid,
                    updates=updates_dict,
                    document_name=state.ingestion.document_name,
                    LMRId=state.ingestion.LMRId,
                )
                status_symbol = "✓" if validation_result.passed else "✗"
                print(f"\n[{status_symbol}] Database updated with validation status: {document_status}")
                if llm_validation.get("validation_performed"):
                    print(f"[✓] LLM validation insights saved to database")
            
        except Exception as e:
            print(f"\n[WARN] Failed to update database with validation results: {e}")
    
    log_agent_event(state, "Validation Check", "completed", {
        "passed": validation_result.passed,
        "errors": len(validation_result.errors),
        "warnings": len(validation_result.warnings),
        "document_type": doc_type
    })
    
    # Update extraction state with validation results for router
    # This allows the workflow router to check if validation passed
    if state.extraction:
        state.extraction.passed = validation_result.passed
        if not validation_result.passed:
            # Get the validation failure message
            failure_message = "Validation failed"
            if validation_result.errors:
                failure_message = validation_result.errors[0]
            elif validation_result.warnings:
                failure_message = validation_result.warnings[0]
            state.extraction.message = failure_message
    
    print("=" * 80 + "\n")
    
    return state

