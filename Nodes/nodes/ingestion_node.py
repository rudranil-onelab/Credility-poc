"""
Ingestion node for processing SQS messages and extracting document metadata.

In DEMO_MODE (set DEMO_MODE=true), this node supports direct file processing
without SQS dependency:
- Local file paths: process_demo_file(file_path="/path/to/file.pdf")
- File bytes: process_demo_file(file_bytes=b"...", file_name="doc.pdf")
- S3 keys: process_demo_file(s3_key="path/to/file.pdf")

Set DEMO_MODE environment variable to enable demo mode.
"""

import os
import json
import time
import re
import datetime
from typing import Optional
from urllib.parse import unquote_plus

from ..config.state_models import PipelineState, IngestionState
from ..utils.helpers import log_agent_event, diagnose_network_issue, check_network_connectivity, check_dns_resolution
from botocore.exceptions import ClientError, EndpointConnectionError
from botocore.exceptions import BotoCoreError
import boto3
from ..tools.aws_services import get_s3_client
from ..tools.db import fetch_agent_context, update_doc_id_if_not_set
from ..config.settings import S3_BUCKET, SQS_QUEUE_URL, DEMO_MODE, is_demo_mode
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def process_demo_file(
    file_path: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    file_name: Optional[str] = None,
    s3_key: Optional[str] = None,
    s3_bucket: Optional[str] = None,
    document_name: Optional[str] = None,
    **kwargs
) -> PipelineState:
    """
    Convenience function to process a file directly in demo mode.
    
    This function creates a PipelineState and processes the file without SQS.
    Use this when DEMO_MODE is enabled to bypass SQS polling.
    
    Args:
        file_path: Local file system path to the document
        file_bytes: Raw file bytes (requires file_name)
        file_name: Filename when using file_bytes
        s3_key: S3 key if file is already in S3
        s3_bucket: S3 bucket name (defaults to configured bucket)
        document_name: Document type name (e.g., "Aadhaar", "PAN Card")
        **kwargs: Additional parameters (FPCID, LMRId, checklistId, etc.)
    
    Returns:
        PipelineState with ingestion state populated
    
    Example:
        from Nodes.nodes.ingestion_node import process_demo_file
        from Nodes.config.settings import is_demo_mode
        
        if is_demo_mode():
            state = process_demo_file(
                file_path="/path/to/aadhaar.pdf",
                document_name="Aadhaar"
            )
    """
    state = PipelineState()
    
    # Store file bytes in state if provided
    if file_bytes:
        state._demo_file_bytes = file_bytes
    
    # Process the file
    ingestion_state = _process_demo_file(
        file_path=file_path,
        file_bytes=file_bytes,
        file_name=file_name,
        s3_key=s3_key,
        s3_bucket=s3_bucket,
        document_name=document_name,
        FPCID=kwargs.get("FPCID"),
        LMRId=kwargs.get("LMRId"),
        checklistId=kwargs.get("checklistId"),
        airecordid=kwargs.get("airecordid"),
        doc_id=kwargs.get("doc_id"),
        user_id=kwargs.get("user_id"),
    )
    
    state.ingestion = ingestion_state
    return state


# ---------- S3 Upload Utilities (from s3_uploader.py) ----------
BUCKET = os.getenv("S3_BUCKET", S3_BUCKET)
ROOT_PREFIX = os.getenv("ROOT_PREFIX", "LMRFileDocNew")


def sanitize_name(name: str) -> str:
    """Sanitize filename by removing special characters."""
    name = name.strip().replace("\\", "/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._ ()\-]", "_", name) or "document"


def split_base_ext(filename: str):
    """Split filename into base and extension."""
    m = re.match(r"^(.*?)(\.[^.]+)?$", filename)
    return (m.group(1) or filename), (m.group(2) or "")


def key_exists(s3_client, bucket: str, key: str) -> bool:
    """Check if a key exists in S3."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def dedup_key(s3_client, bucket: str, key: str) -> str:
    """Generate a unique key by appending (n) if key exists."""
    if not key_exists(s3_client, bucket, key):
        return key
    folder, name = (key.rsplit("/", 1) + [""])[:2]
    base, ext = split_base_ext(name)
    n = 1
    while True:
        cand = f"{folder}/{base}({n}){ext}" if folder else f"{base}({n}){ext}"
        if not key_exists(s3_client, bucket, cand):
            return cand
        n += 1


def build_prefix(FPCID: str, year: str, month: str, day: str, LMRId: str) -> str:
    """Build S3 prefix path."""
    return f"{ROOT_PREFIX}/{FPCID}/{year}/{month}/{day}/{LMRId}/upload"


def upload_bytes(s3_client, bucket: str, key: str, body: bytes, content_type="application/octet-stream"):
    """Upload bytes to S3."""
    return s3_client.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)


def read_efs_file(path: str) -> tuple:
    """Read file from EFS and return (bytes, content_type, size_bytes)."""
    # Check if file exists first
    if not os.path.exists(path):
        # Try to provide helpful debugging info
        dir_path = os.path.dirname(path)
        filename = os.path.basename(path)
        print(f"[DEBUG] File not found at: {path}")
        print(f"[DEBUG] Directory: {dir_path}")
        print(f"[DEBUG] Filename: {filename}")
        
        # Check if directory exists
        if os.path.exists(dir_path):
            print(f"[DEBUG] Directory exists. Contents:")
            try:
                files = os.listdir(dir_path)
                for f in files[:10]:  # Show first 10 files
                    print(f"[DEBUG]   - {f}")
                if len(files) > 10:
                    print(f"[DEBUG]   ... and {len(files) - 10} more files")
            except Exception as e:
                print(f"[DEBUG] Could not list directory: {e}")
        else:
            print(f"[DEBUG] Directory does not exist!")
        
        raise FileNotFoundError(f"EFS file not found: {path}")
    
    with open(path, "rb") as f:
        b = f.read()
    ext = os.path.splitext(path)[1].lower()
    ctype = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif"
    }.get(ext, "application/octet-stream")
    return b, ctype, len(b)


def upload_from_efs_to_s3(
    s3_client,
    efs_path: str,
    FPCID: str,
    LMRId: str,
    year: str,
    month: str,
    day: str,
    document_name: str = None,
    bucket: str = BUCKET
) -> dict:
    """
    Upload file from EFS to S3 with metadata.
    Returns dict with document_key, metadata_key, s3_key, metadata.
    """
    # Ensure year, month, day are zero-padded strings
    year = str(year)
    month = str(month).zfill(2)
    day = str(day).zfill(2)
    
    # Build S3 prefix
    prefix = build_prefix(str(FPCID), year, month, day, str(LMRId))
    
    # Sanitize filename
    filename = sanitize_name(os.path.basename(efs_path))
    doc_key = dedup_key(s3_client, bucket, f"{prefix}/document/{filename}")
    
    # Read file from EFS
    print(f"[INFO] Reading file from EFS: {efs_path}")
    content, content_type, size_bytes = read_efs_file(efs_path)
    
    # Upload document to S3
    print(f"[INFO] Uploading to S3: s3://{bucket}/{doc_key}")
    put_resp = upload_bytes(s3_client, bucket, doc_key, content, content_type)
    
    # Create metadata
    meta_dir = f"{prefix}/metadata/"
    meta_name = sanitize_name(os.path.basename(doc_key)) + ".json"
    meta_key = f"{meta_dir}{meta_name}"
    
    metadata = {
        "FPCID": str(FPCID),
        "LMRId": str(LMRId),
        "file_name": filename,
        "s3_bucket": bucket,
        "s3_key": doc_key,
        "uploaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "content_type": content_type,
        "size_bytes": size_bytes,
        "etag": put_resp.get("ETag", "").strip('"'),
        "prefix_parts": {"year": year, "month": month, "day": day},
        "_source": "efs_upload"
    }
    
    # Add document_name if provided
    if document_name:
        metadata["document_name"] = str(document_name)
    
    # Upload metadata to S3
    print(f"[INFO] Uploading metadata to S3: s3://{bucket}/{meta_key}")
    upload_bytes(s3_client, bucket, meta_key, json.dumps(metadata, indent=2).encode("utf-8"), "application/json")
    
    print(f"[✓] Successfully uploaded file and metadata to S3")
    
    return {
        "document_key": doc_key,
        "metadata_key": meta_key,
        "s3_key": doc_key,
        "metadata": metadata,
        "bucket": bucket
    }
# ---------- End S3 Upload Utilities ----------


# Global cache to track recently processed files (prevents duplicate processing)
_PROCESSED_FILES_CACHE = {}
_CACHE_EXPIRY_SECONDS = 300  # 5 minutes

def _is_recently_processed(s3_key: str) -> bool:
    """Check if file was recently processed (within last 5 minutes)."""
    import time
    current_time = time.time()
    
    # Clean up expired entries
    expired_keys = [k for k, v in _PROCESSED_FILES_CACHE.items() if current_time - v > _CACHE_EXPIRY_SECONDS]
    for k in expired_keys:
        del _PROCESSED_FILES_CACHE[k]
    
    # Check if this file was recently processed
    if s3_key in _PROCESSED_FILES_CACHE:
        time_since_processed = current_time - _PROCESSED_FILES_CACHE[s3_key]
        return time_since_processed < _CACHE_EXPIRY_SECONDS
    
    return False

def _mark_as_processed(s3_key: str):
    """Mark file as processed."""
    import time
    _PROCESSED_FILES_CACHE[s3_key] = time.time()


def _process_demo_file(
    file_path: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    file_name: Optional[str] = None,
    s3_key: Optional[str] = None,
    s3_bucket: Optional[str] = None,
    document_name: Optional[str] = None,
    FPCID: Optional[str] = None,
    LMRId: Optional[str] = None,
    checklistId: Optional[str] = None,
    airecordid: Optional[str] = None,
    doc_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> IngestionState:
    """
    Process a file directly in demo mode (no SQS required).
    
    Supports:
    - Local file path: file_path="/path/to/file.pdf"
    - File bytes: file_bytes=b"...", file_name="document.pdf"
    - S3 key: s3_key="path/to/file.pdf", s3_bucket="bucket-name"
    
    Args:
        file_path: Local file system path to the document
        file_bytes: Raw file bytes (requires file_name)
        file_name: Filename when using file_bytes
        s3_key: S3 key if file is already in S3
        s3_bucket: S3 bucket name (defaults to configured bucket)
        document_name: Document type name (e.g., "Aadhaar", "PAN Card")
        FPCID: Optional FPCID for demo
        LMRId: Optional LMRId for demo
        checklistId: Optional checklistId for demo
        airecordid: Optional airecordid for demo
        doc_id: Optional doc_id for demo
        user_id: Optional user_id for demo
    
    Returns:
        IngestionState populated with file information
    """
    print("\n" + "=" * 80)
    print("🔧 DEMO MODE: Direct File Processing (SQS Bypassed)")
    print("=" * 80)
    
    s3 = get_s3_client()
    bucket = s3_bucket or BUCKET
    
    # Determine file source and get file info
    if file_path:
        # Local file path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Local file not found: {file_path}")
        
        print(f"[DEMO] Processing local file: {file_path}")
        filename = file_name or os.path.basename(file_path)
        
        # Read file
        with open(file_path, "rb") as f:
            file_bytes_data = f.read()
        
        # Get file extension for content type
        ext = os.path.splitext(filename)[1].lower()
        content_type_map = {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif"
        }
        content_type = content_type_map.get(ext, "application/octet-stream")
        size_bytes = len(file_bytes_data)
        
        # Upload to S3 for processing (optional - can process locally too)
        # For now, we'll create a demo S3 key
        now = datetime.datetime.now(datetime.timezone.utc)
        year = str(now.year)
        month = str(now.month).zfill(2)
        day = str(now.day).zfill(2)
        
        demo_prefix = f"demo/{year}/{month}/{day}"
        sanitized_name = sanitize_name(filename)
        demo_key = f"{demo_prefix}/{sanitized_name}"
        
        # Upload to S3 (optional - comment out if you want pure local processing)
        try:
            print(f"[DEMO] Uploading to S3 (optional): s3://{bucket}/{demo_key}")
            s3.put_object(
                Bucket=bucket,
                Key=demo_key,
                Body=file_bytes_data,
                ContentType=content_type
            )
            s3_key = demo_key
            print(f"[DEMO] ✓ File uploaded to S3")
        except Exception as e:
            print(f"[DEMO WARN] Could not upload to S3 (continuing with local processing): {e}")
            # Use a virtual key for local-only processing
            s3_key = f"local://{file_path}"
    
    elif file_bytes and file_name:
        # Direct file bytes upload
        print(f"[DEMO] Processing file bytes: {file_name}")
        filename = file_name
        file_bytes_data = file_bytes
        
        # Get content type from filename
        ext = os.path.splitext(filename)[1].lower()
        content_type_map = {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif"
        }
        content_type = content_type_map.get(ext, "application/octet-stream")
        size_bytes = len(file_bytes_data)
        
        # Upload to S3
        now = datetime.datetime.now(datetime.timezone.utc)
        year = str(now.year)
        month = str(now.month).zfill(2)
        day = str(now.day).zfill(2)
        
        demo_prefix = f"demo/{year}/{month}/{day}"
        sanitized_name = sanitize_name(filename)
        demo_key = f"{demo_prefix}/{sanitized_name}"
        
        try:
            print(f"[DEMO] Uploading to S3: s3://{bucket}/{demo_key}")
            s3.put_object(
                Bucket=bucket,
                Key=demo_key,
                Body=file_bytes_data,
                ContentType=content_type
            )
            s3_key = demo_key
            print(f"[DEMO] ✓ File uploaded to S3")
        except Exception as e:
            raise RuntimeError(f"Failed to upload file bytes to S3: {e}")
    
    elif s3_key:
        # File already in S3
        print(f"[DEMO] Processing S3 file: s3://{bucket}/{s3_key}")
        filename = os.path.basename(s3_key)
        
        # Get file info from S3
        try:
            obj = s3.head_object(Bucket=bucket, Key=s3_key)
            content_type = obj.get("ContentType", "application/octet-stream")
            size_bytes = obj.get("ContentLength", 0)
        except ClientError as e:
            raise FileNotFoundError(f"S3 file not found: s3://{bucket}/{s3_key} - {e}")
    
    else:
        raise ValueError("Must provide one of: file_path, (file_bytes + file_name), or s3_key")
    
    # Validate file format
    valid_extensions = ['.jpg', '.jpeg', '.png', '.pdf']
    file_extension = os.path.splitext(filename)[1].lower()
    
    if not file_extension or file_extension not in valid_extensions:
        raise ValueError(f"Invalid file format: {file_extension}. Accepted: jpg, jpeg, png, pdf")
    
    # Create demo metadata
    now = datetime.datetime.now(datetime.timezone.utc)
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    
    metadata = {
        "FPCID": FPCID or "DEMO",
        "LMRId": LMRId or "DEMO",
        "checklistId": str(checklistId) if checklistId else None,
        "airecordid": str(airecordid) if airecordid else None,
        "doc_id": str(doc_id) if doc_id else None,
        "user_id": str(user_id) if user_id else None,
        "document_name": document_name or filename,
        "file_name": filename,
        "s3_bucket": bucket,
        "s3_key": s3_key,
        "uploaded_at": now.isoformat(),
        "content_type": content_type,
        "size_bytes": size_bytes,
        "prefix_parts": {"year": year, "month": month, "day": day},
        "_source": "demo_mode_direct_upload"
    }
    
    print(f"[DEMO] File: {filename}")
    print(f"[DEMO] Document Name: {document_name or 'Not specified'}")
    print(f"[DEMO] Size: {size_bytes} bytes")
    print(f"[DEMO] Content Type: {content_type}")
    print(f"[DEMO] S3 Location: s3://{bucket}/{s3_key}")
    print("=" * 80 + "\n")
    
    # Create and return IngestionState
    return IngestionState(
        s3_bucket=bucket,
        s3_key=s3_key,
        metadata_s3_path=None,  # No metadata file in demo mode
        FPCID=FPCID or "DEMO",
        LMRId=LMRId or "DEMO",
        checklistId=str(checklistId) if checklistId else None,
        airecordid=str(airecordid) if airecordid else None,
        doc_id=str(doc_id) if doc_id else None,
        user_id=str(user_id) if user_id else None,
        document_name=document_name or filename,
        sqs_receipt_handle=None,
        sqs_message_id="DEMO_MODE",
        document_type=None,
        agent_name="Demo Identity Verification Agent",
        agent_type=None,
        tool="ocr+llm",
        source_url=None,
        content_type=content_type,
        uploaded_at=now.isoformat(),
        size_bytes=size_bytes,
        etag=None,
        prefix_parts=metadata.get("prefix_parts"),
        raw_metadata=metadata,
    )


def Ingestion(state: PipelineState) -> PipelineState:
    """
    Poll SQS until a message arrives and populate ingestion state.
    
    In DEMO_MODE, supports direct file processing without SQS:
    - Local file paths
    - File bytes (direct upload)
    - S3 keys (if file already uploaded)
    
    Demo mode parameters can be passed via state.ingestion or environment variables.
    """
    log_agent_event(state, "Ingestion", "start")
    
    # Check if demo mode is enabled
    if is_demo_mode():
        print("\n" + "=" * 80)
        print("🔧 DEMO MODE ENABLED - Bypassing SQS")
        print("=" * 80)
        
        # Check if ingestion state already has data (from direct call)
        if state.ingestion and state.ingestion.s3_key:
            print("[DEMO] Using existing ingestion state")
            return state
        
        # Check for demo mode parameters in environment or state
        # These can be set by calling code before Ingestion
        demo_file_path = os.getenv("DEMO_FILE_PATH")
        demo_file_name = os.getenv("DEMO_FILE_NAME")
        demo_s3_key = os.getenv("DEMO_S3_KEY")
        demo_s3_bucket = os.getenv("DEMO_S3_BUCKET")
        demo_document_name = os.getenv("DEMO_DOCUMENT_NAME")
        
        # Try to get file bytes from state if available (for direct uploads)
        demo_file_bytes = getattr(state, '_demo_file_bytes', None)
        
        # If we have any demo parameters, process directly
        if demo_file_path or (demo_file_bytes and demo_file_name) or demo_s3_key:
            try:
                ingestion_state = _process_demo_file(
                    file_path=demo_file_path,
                    file_bytes=demo_file_bytes,
                    file_name=demo_file_name,
                    s3_key=demo_s3_key,
                    s3_bucket=demo_s3_bucket or BUCKET,
                    document_name=demo_document_name,
                    FPCID=os.getenv("DEMO_FPCID"),
                    LMRId=os.getenv("DEMO_LMRId"),
                    checklistId=os.getenv("DEMO_CHECKLIST_ID"),
                    airecordid=os.getenv("DEMO_AI_RECORD_ID"),
                    doc_id=os.getenv("DEMO_DOC_ID"),
                    user_id=os.getenv("DEMO_USER_ID"),
                )
                state.ingestion = ingestion_state
                log_agent_event(state, "Ingestion", "completed", {"mode": "demo", "file": ingestion_state.s3_key})
                return state
            except Exception as e:
                print(f"[DEMO ERROR] Failed to process demo file: {e}")
                import traceback
                traceback.print_exc()
                log_agent_event(state, "Ingestion", "error", {"error": str(e), "mode": "demo"})
                raise
        
        # If no demo parameters, show helpful message
        print("[DEMO] No demo file parameters provided.")
        print("[DEMO] Set one of the following environment variables:")
        print("  - DEMO_FILE_PATH=/path/to/local/file.pdf")
        print("  - DEMO_S3_KEY=path/to/file.pdf (with DEMO_S3_BUCKET)")
        print("[DEMO] Or pass file_bytes via state._demo_file_bytes with DEMO_FILE_NAME")
        print("[DEMO] Falling back to SQS polling (will work but not recommended in demo mode)...")
        print("=" * 80 + "\n")
        # Continue to SQS polling as fallback
    
    # Production mode: SQS polling (original behavior)
    # Setup clients
    region = os.getenv("AWS_REGION", "us-east-2")
    queue_url = os.getenv("SQS_QUEUE_URL", SQS_QUEUE_URL)
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    kwargs = {"region_name": region}
    if aws_access_key_id and aws_secret_access_key:
        kwargs["aws_access_key_id"] = aws_access_key_id
        kwargs["aws_secret_access_key"] = aws_secret_access_key
    
    sqs = boto3.client("sqs", **kwargs)
    s3 = get_s3_client()

    # Helper: fetch sidecar metadata JSON alongside the uploaded object
    def fetch_metadata(bucket: str, key: str) -> dict:
        base_dir = os.path.dirname(key)
        filename = os.path.basename(key)
        # If the object is under /document/, normalize to metadata path parallel to upload/
        # Example: .../upload/document/<file> -> .../upload/metadata/<file>.json
        if base_dir.endswith("/document"):
            meta_base = base_dir.rsplit("/", 1)[0]
            meta_key = f"{meta_base}/metadata/{filename}.json"
        else:
            meta_key = f"{base_dir}/metadata/{filename}.json"
        print(f"[DEBUG] Metadata key being used: {meta_key}")
        for i in range(1, 4):
            try:
                print(f"[INFO] Attempt {i} to fetch metadata: {meta_key}")
                obj = s3.get_object(Bucket=bucket, Key=meta_key)
                content = obj["Body"].read().decode("utf-8")
                data = json.loads(content)
                # Also return the s3 path we used to fetch the metadata
                data.setdefault("_metadata_s3_path", f"s3://{bucket}/{meta_key}")
                return data
            except ClientError as e:
                print(f"[WARN] Failed to fetch metadata {meta_key}: {e}")
                if i < 3:
                    print("[INFO] Retrying in 2 seconds...")
                    time.sleep(2)
        print("[ERROR] Metadata not found after retries.")
        return {}

    # Long-poll for messages
    consecutive_network_errors = 0
    max_network_errors = 3
    
    while True:
        try:
            # Check network connectivity before attempting AWS calls
            if consecutive_network_errors > 0:
                if not check_network_connectivity():
                    if consecutive_network_errors >= max_network_errors:
                        diagnosis = diagnose_network_issue()
                        print(f"[NETWORK ERROR] {diagnosis}")
                        print("[INFO] Waiting 30 seconds before retrying...")
                        time.sleep(30)
                        consecutive_network_errors = 0  # Reset after long wait
                    else:
                        print(f"[NETWORK WARNING] Network connectivity issue detected. Retrying in 10 seconds... ({consecutive_network_errors}/{max_network_errors})")
                        time.sleep(10)
                        consecutive_network_errors += 1
                        continue
                elif not check_dns_resolution():
                    if consecutive_network_errors >= max_network_errors:
                        diagnosis = diagnose_network_issue()
                        print(f"[DNS ERROR] {diagnosis}")
                        print("[INFO] Waiting 30 seconds before retrying...")
                        time.sleep(30)
                        consecutive_network_errors = 0
                    else:
                        print(f"[DNS WARNING] DNS resolution issue detected. Retrying in 10 seconds... ({consecutive_network_errors}/{max_network_errors})")
                        time.sleep(10)
                        consecutive_network_errors += 1
                        continue
                else:
                    # Network is back up, reset counter
                    consecutive_network_errors = 0
            
            resp = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
            )
            messages = resp.get("Messages", [])
            # Reset network error counter on successful connection
            consecutive_network_errors = 0
            if not messages:
                print("[INFO] No messages in queue. Waiting...")
                time.sleep(5)
                continue

            msg = messages[0]
            receipt = msg.get("ReceiptHandle")
            msg_id = msg.get("MessageId")
            print(f"\n[SQS] 📨 Received message: {msg_id}")

            body = json.loads(msg.get("Body", "{}"))
            
            # Log message type for debugging
            is_direct_message = "FPCID" in body and "LMRId" in body and "file" in body
            is_s3_event = "Records" in body
            
            if is_direct_message:
                print(f"[DEBUG] Message type: DIRECT SQS MESSAGE (from batch upload)")
                print(f"[DEBUG] Direct message will be processed (preferred over S3 events)")
            elif is_s3_event:
                print(f"[DEBUG] Message type: S3 EVENT NOTIFICATION")
                # Extract S3 key from S3 event to check if it's already being processed via direct message
                records = body.get("Records", [])
                if records:
                    s3_key_from_event = unquote_plus(records[0].get("s3", {}).get("object", {}).get("key", ""))
                    # Check if this file was recently processed (might have been processed via direct message)
                    if _is_recently_processed(s3_key_from_event):
                        print(f"[INFO] Skipping S3 event - file was recently processed (likely via direct SQS message): {s3_key_from_event}")
                        if receipt:
                            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                            print(f"[✓] SQS message deleted (duplicate S3 event): {msg_id}")
                        continue
            else:
                print(f"[DEBUG] Message type: UNKNOWN - Body keys: {list(body.keys())}")
            
            # Check if this is the new direct message format
            if is_direct_message:
                # New direct message format
                FPCID = body.get("FPCID")
                LMRId = body.get("LMRId")
                checklistId = str(body.get("checklistId")) if body.get("checklistId") else None  # Convert to string
                airecordid = str(body.get("airecordid")) if body.get("airecordid") else None  # Convert to string
                doc_id = str(body.get("doc_id")) if body.get("doc_id") else None  # Convert to string
                user_id = str(body.get("user_id")) if body.get("user_id") else None  # Convert to string
                file_path = body.get("file")
                document_name = body.get("document-name")
                year = body.get("year")
                month = body.get("month")
                day = body.get("day")
                # entity_type = body.get("entity_type")  # Ignore for now as requested
                
                print(f"[INFO] Processing direct SQS message format")
                print(f"[INFO] FPCID: {FPCID}, LMRId: {LMRId}, checklistId: {checklistId}, airecordid: {airecordid}, doc_id: {doc_id}, user_id: {user_id}, File: {file_path}")
                
                # Decode URL-encoded file path (handles spaces and special characters)
                if file_path:
                    file_path = unquote_plus(file_path)
                
                # Check if file_path is an EFS path (starts with /mnt/efs)
                if file_path and file_path.startswith("/mnt/efs"):
                    print(f"[INFO] Detected EFS path, uploading to S3...")
                    
                    # Upload from EFS to S3
                    try:
                        upload_result = upload_from_efs_to_s3(
                            s3_client=s3,
                            efs_path=file_path,
                            FPCID=FPCID,
                            LMRId=LMRId,
                            year=year,
                            month=month,
                            day=day,
                            document_name=document_name,
                            bucket=BUCKET
                        )
                        
                        # Update variables with S3 locations
                        bucket = upload_result["bucket"]
                        key = upload_result["s3_key"]
                        
                        print(f"[✓] File uploaded from EFS to S3: s3://{bucket}/{key}")
                    except FileNotFoundError:
                        print(f"[ERROR] EFS file not found: {file_path}")
                        if receipt:
                            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                        continue
                    except Exception as e:
                        print(f"[ERROR] Failed to upload from EFS to S3: {e}")
                        if receipt:
                            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                        continue
                else:
                    # File path is already an S3 key
                    # Decode URL-encoded characters (e.g., %20 for spaces)
                    bucket = os.getenv("S3_BUCKET", BUCKET)  # Use env var or default
                    key = unquote_plus(file_path) if file_path else ""
                
            else:
                # Legacy S3 event format
                records = body.get("Records") or []
                if not records:
                    print(f"[WARN] Skipping unrecognized message format: {body}")
                    if receipt:
                        sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                    continue

                # Use only the first S3 record
                record = records[0]
                bucket = record.get("s3", {}).get("bucket", {}).get("name")
                key = unquote_plus(record.get("s3", {}).get("object", {}).get("key", ""))
                
                # For legacy format, we'll extract FPCID/LMRId/checklistId/airecordid from metadata later
                FPCID = None
                LMRId = None
                checklistId = None
                airecordid = None
                doc_id = None
                user_id = None
                document_name = None
                year = None
                month = None
                day = None

            # Skip metadata and result files themselves
            if key.endswith(".json") or "/metadata/" in key or "/result/" in key:
                print(f"[INFO] Skipping JSON/metadata/result file: {key}")
                if receipt:
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                    print(f"[✓] SQS message deleted (metadata/result file): {msg_id}")
                continue
            
            # Validate file format - only accept jpg, jpeg, png, and pdf
            # Ensure key is properly decoded (handle URL-encoded spaces and special characters)
            if key:
                # Decode any remaining URL-encoded characters
                key = unquote_plus(key)
            
            valid_extensions = ['.jpg', '.jpeg', '.png', '.pdf']
            # Extract extension from the decoded key
            file_extension = os.path.splitext(key)[1].lower() if key else ""
            
            if not file_extension or file_extension not in valid_extensions:
                error_message = f"Document is of invalid format. Accepted formats: jpg, jpeg, png, pdf. Received: {file_extension or 'unknown format'}"
                print(f"[ERROR] {error_message}")
                print(f"[ERROR] File: {key}")
                
                # Log the error event
                log_agent_event(state, "Ingestion", "error", {
                    "error": error_message,
                    "file_key": key,
                    "file_extension": file_extension
                })
                
                # Delete the SQS message since we can't process this file
                if receipt:
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                    print(f"[✓] SQS message deleted (invalid file format): {msg_id}")
                
                # Continue to next message
                continue
            
            # Check if this file was recently processed (prevents duplicate processing from multiple message sources)
            if _is_recently_processed(key):
                print(f"[INFO] Skipping recently processed file: {key}")
                print(f"[INFO] This file was processed within the last {_CACHE_EXPIRY_SECONDS} seconds")
                print(f"[INFO] This is likely a duplicate S3 event notification - the file was already processed via direct SQS message")
                if receipt:
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                    print(f"[✓] SQS message deleted (duplicate): {msg_id}")
                continue

            # Handle metadata based on message format
            if "FPCID" in body and "LMRId" in body and "file" in body:
                # For new direct format, check if we uploaded from EFS
                if file_path.startswith("/mnt/efs") and 'upload_result' in locals():
                    # Use metadata from the S3 upload
                    meta = upload_result["metadata"]
                    # Add metadata S3 path
                    meta["_metadata_s3_path"] = f"s3://{bucket}/{upload_result['metadata_key']}"
                    print("\n====================== 📄 EFS UPLOAD METADATA ======================")
                    try:
                        print(json.dumps(meta, indent=2))
                    except Exception:
                        print(str(meta))
                    print("====================================================================\n")
                else:
                    # For direct S3 path, create metadata from SQS message
                    meta = {
                        "FPCID": str(FPCID),
                        "LMRId": str(LMRId),
                        "checklistId": str(checklistId) if checklistId else None,
                        "airecordid": str(airecordid) if airecordid else None,
                        "doc_id": str(doc_id) if doc_id else None,
                        "user_id": str(user_id) if user_id else None,
                        "document_name": document_name,
                        "file_name": os.path.basename(key),
                        "s3_bucket": bucket,
                        "s3_key": key,
                        "year": year,
                        "month": month,
                        "day": day,
                        "_source": "direct_sqs_message"
                    }
                    print("\n====================== 📄 DIRECT SQS MESSAGE CONTENT ======================")
                    try:
                        print(json.dumps(meta, indent=2))
                    except Exception:
                        print(str(meta))
                    print("=====================================================================\n")
            else:
                # Legacy format - fetch metadata from S3
                meta = fetch_metadata(bucket, key)
                print("\n====================== 📄 METADATA FILE CONTENT ======================")
                try:
                    print(json.dumps(meta, indent=2))
                except Exception:
                    print(str(meta))
                print("=====================================================================\n")
                
                # Extract FPCID/LMRId/checklistId/airecordid from metadata for legacy format
                FPCID = meta.get("FPCID")
                LMRId = meta.get("LMRId")
                checklistId = str(meta.get("checklistId")) if meta.get("checklistId") else None  # Convert to string
                airecordid = str(meta.get("airecordid")) if meta.get("airecordid") else None  # Convert to string
            
            # Try to determine document name from various sources
            potential_doc_name = (
                document_name or  # From direct SQS message
                meta.get("document_name") or 
                meta.get("file_name") or 
                os.path.splitext(os.path.basename(key))[0]  # Extract filename without extension
            )
            
            db_ctx = {}
            if FPCID and checklistId:
                try:
                    # Pass the potential document name to get more specific context
                    # Using checklistId instead of airecordid as differentiator
                    db_ctx = fetch_agent_context(str(FPCID), str(checklistId), potential_doc_name) or {}
                    print("\n====================== 🗄️ DB CONTEXT (BY FPCID + checklistId + document_name) ======================")
                    print(f"Keys -> FPCID={FPCID}, checklistId={checklistId}, document_name={potential_doc_name}")
                    try:
                        print(json.dumps(db_ctx, indent=2))
                    except Exception:
                        print(str(db_ctx))
                    if not db_ctx:
                        print("[WARNING] No DB config record found for provided keys!")
                        print("[WARNING] Processing will continue but results cannot be saved to database.")
                        print("[ACTION REQUIRED] Please create a config record first:")
                        print(f"  POST /create-agent-record")
                        print(f"  {{")
                        print(f"    \"FPCID\": \"{FPCID}\",")
                        print(f"    \"checklistId\": \"{checklistId}\",")
                        print(f"    \"document_name\": \"{potential_doc_name}\",")
                        print(f"    \"agent_name\": \"Identity Verification Agent\",")
                        print(f"    \"tool\": \"ocr+llm\",")
                        print(f"    \"date\": \"{datetime.datetime.now().strftime('%Y-%m-%d')}\"")
                        print(f"  }}")
                    else:
                        # Extract airecordid from DB context (we need it for other operations)
                        airecordid_from_db = db_ctx.get("id")
                        if airecordid_from_db:
                            airecordid = str(airecordid_from_db)  # Convert to string for Pydantic
                            print(f"[INFO] Extracted airecordid={airecordid} from DB context")
                    print("==============================================================================\n")
                except Exception as e:
                    print(f"[INGESTION] DB context fetch error: {e}")
            
            # Update doc_id in database if present in SQS message (only if not already set)
            # Extract doc_id and user_id from metadata or direct variables
            doc_id_from_msg = doc_id or meta.get("doc_id")
            user_id_from_msg = user_id or meta.get("user_id")
            
            if doc_id_from_msg and FPCID and checklistId and potential_doc_name:
                try:
                    print("\n====================== 🔐 DOC_ID UPDATE PROTECTION ======================")
                    print(f"Attempting to update doc_id='{doc_id_from_msg}' for document '{potential_doc_name}'")
                    # Using checklistId instead of airecordid as differentiator
                    was_updated = update_doc_id_if_not_set(
                        FPCID=str(FPCID),
                        checklistId=str(checklistId),
                        document_name=potential_doc_name,
                        doc_id=str(doc_id_from_msg),
                        user_id=str(user_id_from_msg) if user_id_from_msg else None
                    )
                    if was_updated:
                        print(f"✓ doc_id set to '{doc_id_from_msg}' (first time)")
                    else:
                        print(f"⚠️  doc_id update skipped (already set or no matching record)")
                    print("=========================================================================\n")
                except Exception as e:
                    print(f"[INGESTION] doc_id update error: {e}")
                    print("=========================================================================\n")

            # Build ingestion item
            item = {
                "s3_bucket": bucket,
                "s3_key": key,
                "metadata_s3_path": meta.get("_metadata_s3_path"),
                "FPCID": FPCID,
                "LMRId": LMRId,
                "checklistId": checklistId,
                "airecordid": airecordid,
                "doc_id": doc_id_from_msg,
                "user_id": user_id_from_msg,
                # Prefer DB values when available, fallback to metadata
                "document_name": (
                    db_ctx.get("document_name")
                    or meta.get("document_name")
                    or meta.get("file_name")
                ),
                "document_type": meta.get("document_type"),
                "agent_name": (db_ctx.get("agent_name") or meta.get("agent_name")),
                "agent_type": meta.get("agent_type"),
                "tool": (db_ctx.get("tool") or meta.get("tool")),
                "source_url": meta.get("source_url"),
                "uploaded_at": meta.get("uploaded_at"),
                "content_type": meta.get("content_type"),
                "size_bytes": meta.get("size_bytes"),
                "etag": meta.get("etag"),
                "prefix_parts": meta.get("prefix_parts"),
                "_raw_metadata": meta,
            }

            # Validate file type - only allow jpeg, png, pdf
            content_type = (item.get("content_type") or "").lower()
            file_name = item.get("document_name") or ""
            file_extension = os.path.splitext(file_name)[1].lower() if file_name else ""
            
            # Allowed file types
            allowed_content_types = [
                "image/jpeg",
                "image/jpg", 
                "image/png",
                "application/pdf"
            ]
            allowed_extensions = [".jpg", ".jpeg", ".png", ".pdf"]
            
            # Check if file type is allowed
            is_valid_content_type = content_type in allowed_content_types
            is_valid_extension = file_extension in allowed_extensions
            
            if not is_valid_content_type and not is_valid_extension:
                # Invalid file type - reject and save error to database
                print(f"[ERROR] Invalid file type detected: {content_type} (extension: {file_extension})")
                print(f"[ERROR] Allowed types: JPEG, PNG, PDF only")
                
                # Generate user-friendly error message
                from datetime import datetime, timezone
                from ..tools.db import update_tblaigents_by_keys
                
                file_type_display = content_type if content_type else f"file with extension {file_extension}"
                
                doc_verification_result_json = json.dumps({
                    "score": 0,
                    "stats": {
                        "score": 0,
                        "matched_fields": 0,
                        "mismatched_fields": 0
                    },
                    "reason": [
                        f"Invalid file type: {file_type_display}",
                        "Only JPEG, PNG, and PDF files are accepted",
                        "Please upload a valid document format"
                    ],
                    "details": []
                })
                
                # Save error to database
                try:
                    update_tblaigents_by_keys(
                        FPCID=item.get("FPCID"),
                        airecordid=item.get("airecordid"),
                        updates={
                            "document_status": "fail",
                            "Validation_status": "fail",
                            "doc_verification_result": doc_verification_result_json,
                            "cross_validation": False,
                            "checklistId": item.get("checklistId"),
                            "doc_id": item.get("doc_id"),
                            "document_type": item.get("document_type"),
                            "file_s3_location": f"s3://{bucket}/{key}",
                            "metadata_s3_path": item.get("metadata_s3_path"),
                        },
                        document_name=file_name,
                        LMRId=item.get("LMRId"),
                    )
                    print(f"[✓] Invalid file type error saved to database")
                except Exception as e:
                    print(f"[WARN] Failed to save invalid file type error to database: {e}")
                
                # Delete the SQS message
                if receipt:
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                    print(f"[✓] SQS message deleted: {msg_id}")
                
                # Clean up the invalid file from S3
                try:
                    s3.delete_object(Bucket=bucket, Key=key)
                    print(f"[Cleanup] Deleted invalid file from S3: s3://{bucket}/{key}")
                    
                    # Also delete metadata if exists
                    if item.get("metadata_s3_path"):
                        metadata_key = item.get("metadata_s3_path").replace(f"s3://{bucket}/", "")
                        s3.delete_object(Bucket=bucket, Key=metadata_key)
                        print(f"[Cleanup] Deleted metadata: {item.get('metadata_s3_path')}")
                except Exception as e:
                    print(f"[WARN] Failed to cleanup invalid file from S3: {e}")
                
                print(f"[Worker] Invalid file type rejected. Please upload JPEG, PNG, or PDF files only.")
                print(f"[Worker] Continuing to process other messages...\n")
                
                # Continue to next message instead of processing this invalid file
                continue

            # Print merged summary for quick debugging
            try:
                merged_preview = {
                    "FPCID": item.get("FPCID"),
                    "LMRId": item.get("LMRId"),
                    "checklistId": item.get("checklistId"),
                    "airecordid": item.get("airecordid"),
                    "doc_id": item.get("doc_id"),
                    "user_id": item.get("user_id"),
                    "document_name": item.get("document_name"),
                    "agent_name": item.get("agent_name"),
                    "tool": item.get("tool"),
                }
                print("\n====================== 🔗 MERGED CONTEXT (DB + Metadata) ======================")
                print(json.dumps(merged_preview, indent=2))
                print("==============================================================================\n")
            except Exception:
                pass

            # Delete the message now that we have the payload (like old code)
            # This prevents S3 event notifications from creating duplicate processing
            # IMPORTANT: We delete early to prevent the same message from being processed multiple times
            if receipt:
                sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                print(f"[✓] SQS message deleted from queue: {msg_id}")
            
            # Mark this file as processed to prevent duplicate processing from other message sources
            _mark_as_processed(key)
            print(f"[✓] File marked as processed (cache expires in {_CACHE_EXPIRY_SECONDS}s): {key}")
            
            # Populate ingestion state and return
            state.ingestion = IngestionState(
                s3_bucket=item.get("s3_bucket"),
                s3_key=item.get("s3_key"),
                metadata_s3_path=item.get("metadata_s3_path"),
                FPCID=item.get("FPCID"),
                LMRId=item.get("LMRId"),
                checklistId=item.get("checklistId"),
                airecordid=item.get("airecordid"),
                doc_id=item.get("doc_id"),
                user_id=item.get("user_id"),
                document_name=item.get("document_name"),
                sqs_receipt_handle=None,  # Already deleted (early deletion to prevent duplicates)
                sqs_message_id=msg_id,  # For logging
                document_type=item.get("document_type"),
                agent_name=item.get("agent_name"),
                agent_type=item.get("agent_type"),
                tool=item.get("tool"),
                source_url=item.get("source_url"),
                content_type=item.get("content_type"),
                uploaded_at=item.get("uploaded_at"),
                size_bytes=item.get("size_bytes"),
                etag=item.get("etag"),
                prefix_parts=item.get("prefix_parts"),
                raw_metadata=item.get("_raw_metadata"),
            )
            log_agent_event(state, "Ingestion", "completed", {"messageId": msg.get("MessageId")})
            return state

        except (EndpointConnectionError, BotoCoreError) as e:
            consecutive_network_errors += 1
            error_msg = str(e)
            
            # Check if it's a DNS/network error
            if "getaddrinfo failed" in error_msg or "Failed to resolve" in error_msg or "Could not connect" in error_msg:
                diagnosis = diagnose_network_issue()
                if diagnosis:
                    print(f"[NETWORK ERROR] {diagnosis}")
                else:
                    print(f"[NETWORK ERROR] Failed to connect to AWS endpoint: {error_msg}")
                
                if consecutive_network_errors >= max_network_errors:
                    print(f"[ERROR] Too many consecutive network errors ({consecutive_network_errors}). Waiting 30 seconds before retrying...")
                    time.sleep(30)
                    consecutive_network_errors = 0  # Reset after long wait
                else:
                    print(f"[WARNING] Network error ({consecutive_network_errors}/{max_network_errors}). Retrying in 10 seconds...")
                    time.sleep(10)
            else:
                print(f"[AWS ERROR] {error_msg}")
                log_agent_event(state, "Ingestion", "error", {"error": str(e)})
                time.sleep(5)
        except ClientError as e:
            consecutive_network_errors = 0  # Reset on non-network errors
            print(f"[SQS ERROR] {e}")
            log_agent_event(state, "Ingestion", "error", {"error": str(e)})
            # If we received a message but failed to process it, it will become visible again
            # after visibility timeout, so we continue to next iteration
            time.sleep(5)
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a network-related error
            if "getaddrinfo failed" in error_msg or "Failed to resolve" in error_msg or "Could not connect" in error_msg:
                consecutive_network_errors += 1
                diagnosis = diagnose_network_issue()
                if diagnosis:
                    print(f"[NETWORK ERROR] {diagnosis}")
                else:
                    print(f"[NETWORK ERROR] {error_msg}")
                
                if consecutive_network_errors >= max_network_errors:
                    print(f"[ERROR] Too many consecutive network errors ({consecutive_network_errors}). Waiting 30 seconds before retrying...")
                    time.sleep(30)
                    consecutive_network_errors = 0
                else:
                    print(f"[WARNING] Network error ({consecutive_network_errors}/{max_network_errors}). Retrying in 10 seconds...")
                    time.sleep(10)
            else:
                # Reset on non-network errors
                consecutive_network_errors = 0
                print(f"[INGESTION ERROR] {e}")
                import traceback
                traceback.print_exc()
                log_agent_event(state, "Ingestion", "error", {"error": str(e)})
                # If we received a message but failed to process it, it will become visible again
                # after visibility timeout, so we continue to next iteration
                # This ensures the worker doesn't stop and continues processing other messages
                time.sleep(5)
