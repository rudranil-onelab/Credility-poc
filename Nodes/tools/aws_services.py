"""
AWS service utilities for S3 and Textract operations.
"""

import boto3
from botocore.exceptions import ClientError, BotoCoreError
from botocore.config import Config
from typing import Dict, Any, List, Optional
import time

from ..config.settings import (
    AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
    TEXTRACT_MAX_WAIT_SECONDS, TEXTRACT_FEATURE_TYPES
)

# Default presigned URL expiry (15 minutes)
S3_PRESIGNED_URL_EXPIRY = 900


_BOTO_CONFIG = Config(
    retries={"max_attempts": 5, "mode": "standard"},
    connect_timeout=10,
    read_timeout=60,
)


def get_textract_client():
    """Get or create Textract client (always fresh to ensure correct region)."""
    print(f"[AWS] Creating Textract client for region: {AWS_REGION}")
    kwargs = {"region_name": AWS_REGION, "config": _BOTO_CONFIG}
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
    try:
        return boto3.client("textract", **kwargs)
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError(f"Failed to initialize Textract client: {e}")


def get_s3_client():
    """Get or create S3 client (always fresh to ensure correct region)."""
    print(f"[AWS] Creating S3 client for region: {AWS_REGION}")
    kwargs = {"region_name": AWS_REGION, "config": _BOTO_CONFIG}
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
    try:
        return boto3.client("s3", **kwargs)
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError(f"Failed to initialize S3 client: {e}")


def download_file_from_s3(bucket: str, key: str, local_path: str) -> str:
    """
    Download file from S3 to local path.
    Returns the local file path.
    """
    try:
        s3_client = get_s3_client()
        s3_client.download_file(bucket, key, local_path)
        return local_path
    except Exception as e:
        raise RuntimeError(f"Failed to download file from S3: {e}")


def generate_presigned_url(bucket: str, key: str) -> Optional[str]:
    """Generate presigned URL for S3 object."""
    try:
        s3_client = get_s3_client()
        return s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=S3_PRESIGNED_URL_EXPIRY,
        )
    except Exception as e:
        print(f"Warning: Could not create presigned URL: {e}")
        return None


def run_textract_async_s3(bucket: str, key: str, max_wait_seconds: int = TEXTRACT_MAX_WAIT_SECONDS) -> Dict[str, Any]:
    """Run Textract document analysis on S3 file."""
    client = get_textract_client()
    try:
        response = client.start_document_analysis(
            DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}},
            FeatureTypes=TEXTRACT_FEATURE_TYPES,
        )
        job_id = response["JobId"]
        print(f"[Textract] Started job {job_id}")
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError(f"Textract start failed: {e}")

    start_time = time.time()
    while True:
        try:
            status = client.get_document_analysis(JobId=job_id)
        except (BotoCoreError, ClientError) as e:
            raise RuntimeError(f"Textract polling failed: {e}")
        job_status = status.get("JobStatus")
        if job_status in ["SUCCEEDED", "FAILED"]:
            break
        if time.time() - start_time > max_wait_seconds:
            raise TimeoutError(f"Textract job {job_id} timed out after {max_wait_seconds}s")
        print("[Textract] Job running...")
        time.sleep(5)

    if job_status == "FAILED":
        raise RuntimeError("Textract job failed.")

    blocks: List[Dict[str, Any]] = []
    next_token = None
    pages_total = status["DocumentMetadata"].get("Pages", None)
    while True:
        try:
            if next_token:
                status = client.get_document_analysis(JobId=job_id, NextToken=next_token)
            else:
                status = client.get_document_analysis(JobId=job_id)
        except (BotoCoreError, ClientError) as e:
            raise RuntimeError(f"Textract pagination failed: {e}")
        blocks.extend(status.get("Blocks", []))
        next_token = status.get("NextToken")
        if not next_token:
            break

    return {
        "engine_meta": {
            "mode": "textract:start_document_analysis",
            "pages": pages_total,
            "job_id": job_id,
        },
        "blocks": blocks,
    }


def run_analyze_id_s3(bucket: str, key: str) -> Dict[str, Any]:
    """
    Use Textract AnalyzeID (best for driver licenses, passports).
    Returns dict. We will convert it into KVs later for merging.
    """
    client = get_textract_client()
    try:
        resp = client.analyze_id(DocumentPages=[{"S3Object": {"Bucket": bucket, "Name": key}}])
        return resp
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError(f"AnalyzeID failed: {e}")


def run_textract_sync_bytes(file_bytes: bytes, feature_types: List[str] = None) -> Dict[str, Any]:
    """
    Run Textract synchronously on file bytes (no S3 required).
    
    This is faster for small documents (< 5MB) as it doesn't require S3 upload.
    Supports PDF and images.
    
    Args:
        file_bytes: Raw bytes of the document
        feature_types: List of features to extract (default: TABLES, FORMS)
    
    Returns:
        Dict with blocks from Textract
    """
    client = get_textract_client()
    
    if feature_types is None:
        feature_types = TEXTRACT_FEATURE_TYPES
    
    try:
        print(f"[Textract] Running synchronous analysis ({len(file_bytes)} bytes)...")
        
        response = client.analyze_document(
            Document={"Bytes": file_bytes},
            FeatureTypes=feature_types
        )
        
        blocks = response.get("Blocks", [])
        pages = response.get("DocumentMetadata", {}).get("Pages", 1)
        
        print(f"[Textract] Extracted {len(blocks)} blocks from {pages} page(s)")
        
        return {
            "engine_meta": {
                "mode": "textract:analyze_document",
                "pages": pages,
            },
            "blocks": blocks,
        }
        
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        
        # If document is too large, suggest using S3
        if error_code == "DocumentTooLargeException":
            raise RuntimeError(
                "Document too large for synchronous processing. "
                "Please use S3-based processing for documents > 5MB."
            )
        
        raise RuntimeError(f"Textract analyze_document failed: {e}")
    except BotoCoreError as e:
        raise RuntimeError(f"Textract analyze_document failed: {e}")


def run_textract_local_file(file_path: str, feature_types: List[str] = None) -> Dict[str, Any]:
    """
    Run Textract on a local file (PDF or image).
    
    For single-page images (< 5MB): Uses synchronous API (faster)
    For PDFs or large files: Uploads to S3 temp, runs async, then deletes
    
    Note: Textract's synchronous analyze_document API does NOT support multi-page PDFs.
    PDFs must use the async start_document_analysis API via S3.
    
    Args:
        file_path: Path to local file
        feature_types: List of features to extract
    
    Returns:
        Dict with blocks from Textract
    """
    import os
    import uuid
    from ..config.settings import S3_BUCKET
    
    # Read file
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    
    file_size_mb = len(file_bytes) / (1024 * 1024)
    file_ext = os.path.splitext(file_path)[1].lower()
    is_pdf = file_ext == '.pdf'
    
    print(f"[Textract] File size: {file_size_mb:.2f} MB, Type: {file_ext}")
    
    # PDFs MUST use async API (sync doesn't support multi-page)
    # Also use async for large files (>5MB)
    if is_pdf or file_size_mb >= 5:
        if is_pdf:
            print(f"[Textract] PDF detected - using S3 + async API (required for multi-page support)")
        else:
            print(f"[Textract] Large file - using S3 + async API")
        
        # Upload to S3 for Textract processing
        s3_key = f"documents/{uuid.uuid4()}/{os.path.basename(file_path)}"
        s3_client = get_s3_client()
        
        print(f"[Textract] Uploading to s3://{S3_BUCKET}/{s3_key} (region: {AWS_REGION})")
        s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=file_bytes)
        print(f"[Textract] Upload complete")
        
        # Wait a moment for S3 consistency
        time.sleep(1)
        
        # Verify file exists before calling Textract
        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
            print(f"[Textract] Verified file exists in S3")
        except Exception as e:
            print(f"[Textract] WARNING: Could not verify file in S3: {e}")
        
        # Run async Textract
        result = run_textract_async_s3(S3_BUCKET, s3_key)
        
        return result
    
    # For small images, use synchronous API (faster, no S3 needed)
    print(f"[Textract] Using synchronous API for small image")
    return run_textract_sync_bytes(file_bytes, feature_types)