"""
Multi-Language Support Utilities for Indian Identity Documents.

Supports all 22 official languages of India (8th Schedule) plus regional languages.
Provides script detection, language identification, and field normalization.
"""

import re
from typing import List, Dict, Tuple, Optional, Any


# ==========================================
# UNICODE RANGES FOR INDIAN SCRIPTS
# ==========================================
INDIAN_SCRIPTS = {
    "devanagari": {
        "range": (0x0900, 0x097F),
        "languages": ["hindi", "marathi", "sanskrit", "nepali", "konkani", "dogri", "bodo", "maithili"],
        "pattern": r'[\u0900-\u097F]+',
        "name_native": "देवनागरी"
    },
    "bengali": {
        "range": (0x0980, 0x09FF),
        "languages": ["bengali", "assamese"],
        "pattern": r'[\u0980-\u09FF]+',
        "name_native": "বাংলা"
    },
    "tamil": {
        "range": (0x0B80, 0x0BFF),
        "languages": ["tamil"],
        "pattern": r'[\u0B80-\u0BFF]+',
        "name_native": "தமிழ்"
    },
    "telugu": {
        "range": (0x0C00, 0x0C7F),
        "languages": ["telugu"],
        "pattern": r'[\u0C00-\u0C7F]+',
        "name_native": "తెలుగు"
    },
    "gujarati": {
        "range": (0x0A80, 0x0AFF),
        "languages": ["gujarati"],
        "pattern": r'[\u0A80-\u0AFF]+',
        "name_native": "ગુજરાતી"
    },
    "kannada": {
        "range": (0x0C80, 0x0CFF),
        "languages": ["kannada"],
        "pattern": r'[\u0C80-\u0CFF]+',
        "name_native": "ಕನ್ನಡ"
    },
    "malayalam": {
        "range": (0x0D00, 0x0D7F),
        "languages": ["malayalam"],
        "pattern": r'[\u0D00-\u0D7F]+',
        "name_native": "മലയാളം"
    },
    "odia": {
        "range": (0x0B00, 0x0B7F),
        "languages": ["odia"],
        "pattern": r'[\u0B00-\u0B7F]+',
        "name_native": "ଓଡ଼ିଆ"
    },
    "gurmukhi": {
        "range": (0x0A00, 0x0A7F),
        "languages": ["punjabi"],
        "pattern": r'[\u0A00-\u0A7F]+',
        "name_native": "ਗੁਰਮੁਖੀ"
    },
    "arabic": {
        "range": (0x0600, 0x06FF),
        "languages": ["urdu", "kashmiri_arabic", "sindhi_arabic"],
        "pattern": r'[\u0600-\u06FF]+',
        "name_native": "اردو"
    },
    "ol_chiki": {
        "range": (0x1C50, 0x1C7F),
        "languages": ["santali"],
        "pattern": r'[\u1C50-\u1C7F]+',
        "name_native": "ᱚᱞ ᱪᱤᱠᱤ"
    },
    "meitei": {
        "range": (0xABC0, 0xABFF),
        "languages": ["manipuri"],
        "pattern": r'[\uABC0-\uABFF]+',
        "name_native": "ꯃꯩꯇꯩ"
    }
}


# ==========================================
# LANGUAGE CODE MAPPING (ISO 639-1 based)
# ==========================================
LANGUAGE_CODES = {
    # Primary Indian Languages (8th Schedule)
    "hindi": "Hi",
    "marathi": "Mr",
    "tamil": "Ta",
    "telugu": "Te",
    "bengali": "Bn",
    "gujarati": "Gu",
    "kannada": "Kn",
    "malayalam": "Ml",
    "odia": "Or",
    "punjabi": "Pa",
    "assamese": "As",
    "urdu": "Ur",
    "nepali": "Ne",
    "konkani": "Kok",
    "manipuri": "Mni",
    "kashmiri": "Ks",
    "sanskrit": "Sa",
    "sindhi": "Sd",
    "dogri": "Doi",
    "bodo": "Brx",
    "maithili": "Mai",
    "santali": "Sat",
    
    # English (common in all documents)
    "english": "En",
    
    # Additional regional languages
    "mizo": "Lus",
    "khasi": "Kha",
    "garo": "Grt",
    "kokborok": "Trp",
}


# ==========================================
# STATE TO PRIMARY LANGUAGE MAPPING
# ==========================================
STATE_LANGUAGES = {
    # States
    "andhra pradesh": ["telugu", "english"],
    "arunachal pradesh": ["english", "hindi"],
    "assam": ["assamese", "bengali", "bodo", "english"],
    "bihar": ["hindi", "maithili", "english"],
    "chhattisgarh": ["hindi", "english"],
    "goa": ["konkani", "marathi", "english"],
    "gujarat": ["gujarati", "english"],
    "haryana": ["hindi", "english"],
    "himachal pradesh": ["hindi", "english"],
    "jharkhand": ["hindi", "santali", "english"],
    "karnataka": ["kannada", "english"],
    "kerala": ["malayalam", "english"],
    "madhya pradesh": ["hindi", "english"],
    "maharashtra": ["marathi", "english"],
    "manipur": ["manipuri", "english"],
    "meghalaya": ["english", "khasi", "garo"],
    "mizoram": ["mizo", "english"],
    "nagaland": ["english"],
    "odisha": ["odia", "english"],
    "punjab": ["punjabi", "english"],
    "rajasthan": ["hindi", "english"],
    "sikkim": ["nepali", "english"],
    "tamil nadu": ["tamil", "english"],
    "telangana": ["telugu", "english"],
    "tripura": ["bengali", "kokborok", "english"],
    "uttar pradesh": ["hindi", "english"],
    "uttarakhand": ["hindi", "english"],
    "west bengal": ["bengali", "english"],
    
    # Union Territories
    "delhi": ["hindi", "english"],
    "jammu and kashmir": ["urdu", "kashmiri", "dogri", "english"],
    "ladakh": ["urdu", "english"],
    "puducherry": ["tamil", "telugu", "malayalam", "english"],
    "chandigarh": ["hindi", "punjabi", "english"],
    "andaman and nicobar islands": ["hindi", "english"],
    "dadra and nagar haveli and daman and diu": ["gujarati", "hindi", "english"],
    "lakshadweep": ["malayalam", "english"],
    
    # Common aliases
    "ap": ["telugu", "english"],
    "ts": ["telugu", "english"],
    "tn": ["tamil", "english"],
    "ka": ["kannada", "english"],
    "kl": ["malayalam", "english"],
    "mh": ["marathi", "english"],
    "gj": ["gujarati", "english"],
    "wb": ["bengali", "english"],
    "up": ["hindi", "english"],
    "mp": ["hindi", "english"],
    "rj": ["hindi", "english"],
    "pb": ["punjabi", "english"],
    "hr": ["hindi", "english"],
    "or": ["odia", "english"],
    "jh": ["hindi", "santali", "english"],
    "br": ["hindi", "maithili", "english"],
}


# ==========================================
# MULTI-LANGUAGE KEYWORDS FOR DOCUMENT DETECTION
# ==========================================
MULTILINGUAL_KEYWORDS = {
    "aadhaar": {
        "english": ["aadhaar", "aadhar", "unique identification", "uidai"],
        "hindi": ["आधार", "भारत सरकार", "विशिष्ट पहचान प्राधिकरण", "यूआईडीएआई"],
        "tamil": ["ஆதார்", "இந்திய அரசு", "தனிப்பட்ட அடையாள ஆணையம்"],
        "telugu": ["ఆధార్", "భారత ప్రభుత్వం", "ప్రత్యేక గుర్తింపు ప్రాధికార సంస్థ"],
        "bengali": ["আধার", "ভারত সরকার", "স্বতন্ত্র পরিচয় কর্তৃপক্ষ"],
        "marathi": ["आधार", "भारत सरकार", "विशिष्ट ओळख प्राधिकरण"],
        "gujarati": ["આધાર", "ભારત સરકાર", "યુનિક આઇડેન્ટિફિકેશન ઓથોરિટી"],
        "kannada": ["ಆಧಾರ್", "ಭಾರತ ಸರ್ಕಾರ", "ವಿಶಿಷ್ಟ ಗುರುತಿನ ಪ್ರಾಧಿಕಾರ"],
        "malayalam": ["ആധാർ", "ഇന്ത്യൻ സർക്കാർ", "യുണീക്ക് ഐഡന്റിഫിക്കേഷൻ അതോറിറ്റി"],
        "odia": ["ଆଧାର", "ଭାରତ ସରକାର", "ୟୁନିକ୍ ଆଇଡେଣ୍ଟିଫିକେସନ୍ ଅଥରିଟି"],
        "punjabi": ["ਆਧਾਰ", "ਭਾਰਤ ਸਰਕਾਰ"],
        "assamese": ["আধাৰ", "ভাৰত চৰকাৰ"],
    },
    "pan_card": {
        "english": ["income tax", "permanent account number", "pan"],
        "hindi": ["आयकर विभाग", "स्थायी खाता संख्या", "पैन", "भारत सरकार"],
    },
    "voter_id": {
        "english": ["election commission", "voter", "epic", "elector photo identity"],
        "hindi": ["निर्वाचन आयोग", "मतदाता", "ईपीआईसी", "भारत निर्वाचन आयोग"],
        "tamil": ["தேர்தல் ஆணையம்", "வாக்காளர்", "எபிக்"],
        "telugu": ["ఎన్నికల సంఘం", "ఓటరు", "ఎపిక్"],
        "bengali": ["নির্বাচন কমিশন", "ভোটার", "এপিক"],
        "marathi": ["निवडणूक आयोग", "मतदार", "एपिक"],
        "gujarati": ["ચૂંટણી પંચ", "મતદાર"],
        "kannada": ["ಚುನಾವಣಾ ಆಯೋಗ", "ಮತದಾರ"],
        "malayalam": ["തിരഞ്ഞെടുപ്പ് കമ്മീഷൻ", "വോട്ടർ"],
        "odia": ["ନିର୍ବାଚନ ଆୟୋଗ", "ଭୋଟର"],
        "punjabi": ["ਚੋਣ ਕਮਿਸ਼ਨ", "ਵੋਟਰ"],
    },
    "indian_passport": {
        "english": ["passport", "republic of india", "ministry of external affairs"],
        "hindi": ["पासपोर्ट", "भारत गणराज्य", "विदेश मंत्रालय"],
    },
    "indian_driving_license": {
        "english": ["driving licence", "driving license", "transport department", "rto", "motor vehicle"],
        "hindi": ["ड्राइविंग लाइसेंस", "परिवहन विभाग", "आरटीओ", "मोटर वाहन"],
        "tamil": ["ஓட்டுநர் உரிமம்", "போக்குவரத்து துறை"],
        "telugu": ["డ్రైవింగ్ లైసెన్స్", "రవాణా శాఖ"],
        "bengali": ["ড্রাইভিং লাইসেন্স", "পরিবহন বিভাগ"],
        "marathi": ["ड्रायव्हिंग लायसन्स", "परिवहन विभाग"],
        "gujarati": ["ડ્રાઇવિંગ લાઇસન્સ", "પરિવહન વિભાગ"],
        "kannada": ["ಚಾಲನಾ ಪರವಾನಗಿ", "ಸಾರಿಗೆ ಇಲಾಖೆ"],
        "malayalam": ["ഡ്രൈവിംഗ് ലൈസൻസ്", "ഗതാഗത വകുപ്പ്"],
        "odia": ["ଡ୍ରାଇଭିଂ ଲାଇସେନ୍ସ", "ପରିବହନ ବିଭାଗ"],
        "punjabi": ["ਡਰਾਈਵਿੰਗ ਲਾਇਸੈਂਸ", "ਟ੍ਰਾਂਸਪੋਰਟ ਵਿਭਾਗ"],
    },
    "ration_card": {
        "english": ["ration card", "public distribution", "food supply"],
        "hindi": ["राशन कार्ड", "सार्वजनिक वितरण", "खाद्य आपूर्ति"],
        "tamil": ["ரேஷன் கார்டு", "பொது விநியோக அமைப்பு"],
        "telugu": ["రేషన్ కార్డు", "ప్రజా పంపిణీ వ్యవస్థ"],
        "bengali": ["রেশন কার্ড", "জন বিতরণ ব্যবস্থা"],
        "marathi": ["रेशन कार्ड", "सार्वजनिक वितरण"],
    },
    "gst_certificate": {
        "english": ["goods and services tax", "gstin", "gst certificate", "gst registration"],
        "hindi": ["वस्तु एवं सेवा कर", "जीएसटीआईएन", "जीएसटी प्रमाणपत्र"],
    },
    "birth_certificate": {
        "english": ["birth certificate", "birth registration", "registrar of births"],
        "hindi": ["जन्म प्रमाण पत्र", "जन्म पंजीकरण", "जन्म पंजीयक"],
        "tamil": ["பிறப்பு சான்றிதழ்", "பிறப்பு பதிவு"],
        "telugu": ["జన్మ ధృవీకరణ పత్రం", "జన్మ నమోదు"],
        "bengali": ["জন্ম সনদ", "জন্ম নিবন্ধন"],
        "marathi": ["जन्म प्रमाणपत्र", "जन्म नोंदणी"],
    },
    "marriage_certificate": {
        "english": ["marriage certificate", "marriage registration"],
        "hindi": ["विवाह प्रमाण पत्र", "विवाह पंजीकरण"],
        "tamil": ["திருமண சான்றிதழ்", "திருமண பதிவு"],
        "telugu": ["వివాహ ధృవీకరణ పత్రం"],
        "bengali": ["বিবাহ সনদ", "বিবাহ নিবন্ধন"],
    },
    "caste_certificate": {
        "english": ["caste certificate", "community certificate", "sc", "st", "obc"],
        "hindi": ["जाति प्रमाण पत्र", "समुदाय प्रमाण पत्र", "अनुसूचित जाति", "अनुसूचित जनजाति"],
        "tamil": ["சாதி சான்றிதழ்", "சமூக சான்றிதழ்"],
        "telugu": ["కులం ధృవీకరణ పత్రం"],
        "bengali": ["জাতি সনদ"],
        "marathi": ["जात प्रमाणपत्र"],
    },
    "income_certificate": {
        "english": ["income certificate", "annual income"],
        "hindi": ["आय प्रमाण पत्र", "वार्षिक आय"],
        "tamil": ["வருமான சான்றிதழ்"],
        "telugu": ["ఆదాయ ధృవీకరణ పత్రం"],
        "bengali": ["আয় সনদ"],
    },
    "domicile_certificate": {
        "english": ["domicile certificate", "residence certificate", "permanent resident"],
        "hindi": ["अधिवास प्रमाण पत्र", "निवास प्रमाण पत्र", "स्थायी निवासी"],
        "tamil": ["வசிப்பிட சான்றிதழ்"],
        "telugu": ["నివాస ధృవీకరణ పత్రం"],
        "bengali": ["বাসস্থান সনদ"],
    },
}


# ==========================================
# SCRIPT DETECTION FUNCTIONS
# ==========================================
def detect_scripts(text: str) -> List[str]:
    """
    Detect all scripts present in the text.
    
    Args:
        text: The text to analyze
        
    Returns:
        List of detected script names
    """
    if not text:
        return []
    
    detected = []
    for script_name, script_info in INDIAN_SCRIPTS.items():
        if re.search(script_info["pattern"], text):
            detected.append(script_name)
    
    # Check for Latin/English
    if re.search(r'[a-zA-Z]+', text):
        detected.append("latin")
    
    return detected


def detect_languages(text: str, state_hint: Optional[str] = None) -> List[str]:
    """
    Detect languages in text based on script and context.
    
    Args:
        text: The text to analyze
        state_hint: Optional state name to help disambiguate languages
    
    Returns:
        List of detected language names
    """
    if not text:
        return []
    
    scripts = detect_scripts(text)
    languages = set()
    
    # Add English if Latin script found
    if "latin" in scripts:
        languages.add("english")
    
    for script in scripts:
        if script == "latin":
            continue
        
        script_info = INDIAN_SCRIPTS.get(script, {})
        possible_languages = script_info.get("languages", [])
        
        if len(possible_languages) == 1:
            # Unambiguous script-to-language mapping
            languages.add(possible_languages[0])
        elif state_hint:
            # Use state hint to disambiguate
            state_lower = state_hint.lower().strip()
            state_langs = STATE_LANGUAGES.get(state_lower, [])
            
            matched = False
            for lang in possible_languages:
                if lang in state_langs:
                    languages.add(lang)
                    matched = True
                    break
            
            if not matched and possible_languages:
                # Default to first possible language
                languages.add(possible_languages[0])
        else:
            # Default to first possible language
            if possible_languages:
                languages.add(possible_languages[0])
    
    return list(languages)


def get_language_suffix(language: str) -> str:
    """
    Get the field suffix for a language.
    
    Args:
        language: Language name (e.g., "hindi", "tamil")
        
    Returns:
        Language code suffix (e.g., "Hi", "Ta")
    """
    return LANGUAGE_CODES.get(language.lower(), language[:2].capitalize())


def extract_script_text(text: str, script: str) -> str:
    """
    Extract only the text in a specific script.
    
    Args:
        text: The full text
        script: Script name (e.g., "devanagari", "tamil")
        
    Returns:
        Text containing only characters from the specified script
    """
    pattern = INDIAN_SCRIPTS.get(script, {}).get("pattern", "")
    if not pattern:
        return ""
    
    matches = re.findall(pattern, text)
    return " ".join(matches)


def split_bilingual_field(value: str) -> Dict[str, str]:
    """
    Split a bilingual field value into separate language components.
    
    Example: 
        "RAJESH KUMAR राजेश कुमार" -> {"english": "RAJESH KUMAR", "hindi": "राजेश कुमार"}
    
    Args:
        value: The bilingual text value
        
    Returns:
        Dictionary mapping language to extracted text
    """
    if not value:
        return {}
    
    result = {}
    
    # Extract English (Latin script)
    english_parts = re.findall(r'[a-zA-Z][a-zA-Z\s\.\-\']+', value)
    if english_parts:
        result["english"] = " ".join(english_parts).strip()
    
    # Extract each Indian script
    for script_name, script_info in INDIAN_SCRIPTS.items():
        script_text = extract_script_text(value, script_name)
        if script_text:
            # Determine language from script
            languages = script_info.get("languages", [])
            if languages:
                result[languages[0]] = script_text.strip()
    
    return result


def normalize_multilingual_fields(extracted: Dict[str, Any], detected_languages: List[str] = None) -> Dict[str, Any]:
    """
    Normalize extracted fields to use consistent language suffixes.
    
    Converts:
    - holderNameHindi -> holderName_Hi
    - holderName_hindi -> holderName_Hi
    - etc.
    
    Args:
        extracted: Dictionary of extracted fields
        detected_languages: Optional list of detected languages
        
    Returns:
        Normalized dictionary with consistent field naming
    """
    if not extracted:
        return {}
    
    normalized = {}
    
    for key, value in extracted.items():
        new_key = key
        
        # Handle various suffix formats
        for lang, code in LANGUAGE_CODES.items():
            patterns = [
                (f"_{lang}$", f"_{code}"),           # _hindi -> _Hi
                (f"_{lang.title()}$", f"_{code}"),   # _Hindi -> _Hi
                (f"{lang.title()}$", f"_{code}"),    # Hindi -> _Hi (no underscore)
                (f"_{code.lower()}$", f"_{code}"),   # _hi -> _Hi
            ]
            
            for pattern, replacement in patterns:
                if re.search(pattern, key, re.IGNORECASE):
                    base_key = re.sub(pattern, "", key, flags=re.IGNORECASE)
                    new_key = f"{base_key}{replacement}"
                    break
        
        normalized[new_key] = value
    
    # Add detected languages field if provided
    if detected_languages:
        normalized["detectedLanguages"] = detected_languages
    
    return normalized


def detect_document_by_keywords(text: str) -> Optional[str]:
    """
    Detect document type using multi-language keywords.
    
    Args:
        text: The OCR text to analyze
        
    Returns:
        Detected document type or None
    """
    if not text:
        return None
    
    text_lower = text.lower()
    
    for doc_type, lang_keywords in MULTILINGUAL_KEYWORDS.items():
        for lang, keywords in lang_keywords.items():
            for keyword in keywords:
                # Check both lowercase and original (for non-Latin scripts)
                if keyword.lower() in text_lower or keyword in text:
                    return doc_type
    
    return None


def get_state_from_text(text: str) -> Optional[str]:
    """
    Try to detect the state from text content.
    
    Args:
        text: The text to analyze
        
    Returns:
        Detected state name or None
    """
    if not text:
        return None
    
    text_lower = text.lower()
    
    # Check for state names
    for state in STATE_LANGUAGES.keys():
        if state in text_lower:
            return state
    
    return None


def add_language_variants(fields: Dict[str, Any], detected_languages: List[str]) -> Dict[str, Any]:
    """
    For fields that might have regional language variants,
    add empty placeholders for detected languages.
    
    Args:
        fields: Dictionary of extracted fields
        detected_languages: List of detected languages
        
    Returns:
        Fields with language variant placeholders
    """
    # Fields that commonly have regional language variants
    multilingual_fields = [
        "holderName", "fatherName", "motherName", "husbandName", "spouseName",
        "addressLine1", "addressLine2", "city", "district", "state", "village",
        "placeOfBirth", "issuingAuthority", "caste", "occupation", "tradeName"
    ]
    
    result = dict(fields)
    
    for lang in detected_languages:
        if lang == "english":
            continue
        
        suffix = get_language_suffix(lang)
        
        for field in multilingual_fields:
            variant_key = f"{field}_{suffix}"
            if field in result and variant_key not in result:
                # Add placeholder (will be filled by OCR)
                pass  # Don't add empty placeholders, let OCR fill them
    
    return result


# ==========================================
# VALIDATION HELPERS
# ==========================================
def validate_aadhaar_number(number: str) -> bool:
    """Validate Aadhaar number format (12 digits)."""
    if not number:
        return False
    # Remove spaces and check
    clean = re.sub(r'\s+', '', str(number))
    return bool(re.match(r'^[2-9]\d{11}$', clean))


def validate_pan_number(number: str) -> bool:
    """Validate PAN number format (AAAAA9999A)."""
    if not number:
        return False
    clean = str(number).upper().strip()
    return bool(re.match(r'^[A-Z]{3}[ABCFGHLJPTF][A-Z]\d{4}[A-Z]$', clean))


def validate_epic_number(number: str) -> bool:
    """Validate Voter ID (EPIC) number format."""
    if not number:
        return False
    clean = str(number).upper().strip()
    # EPIC format: 3 letters + 7 digits (e.g., ABC1234567)
    return bool(re.match(r'^[A-Z]{3}\d{7}$', clean))


def validate_gstin(number: str) -> bool:
    """Validate GSTIN format (15 characters)."""
    if not number:
        return False
    clean = str(number).upper().strip()
    # GSTIN format: 2 digits (state) + 10 char PAN + 1 digit + Z + 1 alphanumeric
    return bool(re.match(r'^\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z\d]$', clean))


def format_aadhaar_number(number: str) -> str:
    """Format Aadhaar number with spaces (XXXX XXXX XXXX)."""
    clean = re.sub(r'\s+', '', str(number))
    if len(clean) == 12:
        return f"{clean[:4]} {clean[4:8]} {clean[8:]}"
    return number


# ==========================================
# EXPORTS
# ==========================================
__all__ = [
    # Constants
    "INDIAN_SCRIPTS",
    "LANGUAGE_CODES",
    "STATE_LANGUAGES",
    "MULTILINGUAL_KEYWORDS",
    
    # Functions
    "detect_scripts",
    "detect_languages",
    "get_language_suffix",
    "extract_script_text",
    "split_bilingual_field",
    "normalize_multilingual_fields",
    "detect_document_by_keywords",
    "get_state_from_text",
    "add_language_variants",
    
    # Validators
    "validate_aadhaar_number",
    "validate_pan_number",
    "validate_epic_number",
    "validate_gstin",
    "format_aadhaar_number",
]

