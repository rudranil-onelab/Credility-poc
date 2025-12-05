#!/usr/bin/env python3
"""
Simple S3 uploader for document validation API.

Uploads documents to S3 for AWS Textract processing.
"""

import os
import datetime
import re
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

# S3 Configuration - New POC bucket
BUCKET = os.getenv("S3_BUCKET", "lendingwise-poc")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


def get_s3_client():
    """Create S3 client with credentials from environment."""
    kwargs = {"region_name": AWS_REGION}
    
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    if aws_access_key_id and aws_secret_access_key:
        kwargs["aws_access_key_id"] = aws_access_key_id
        kwargs["aws_secret_access_key"] = aws_secret_access_key
    
    return boto3.client("s3", **kwargs)


S3 = get_s3_client()


def sanitize_filename(name: str) -> str:
    """Sanitize filename for S3."""
    name = name.strip().replace("\\", "/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._()-]", "_", name) or "document"


def generate_unique_key(prefix: str, filename: str) -> str:
    """Generate unique S3 key with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = sanitize_filename(filename)
    base, ext = os.path.splitext(safe_name)
    return f"{prefix}/{base}_{timestamp}{ext}"


def upload_file_to_s3(
    file_path: str,
    prefix: str = "documents",
    bucket: str = None
) -> dict:
    """
    Upload a local file to S3.
    
    Args:
        file_path: Path to local file
        prefix: S3 key prefix (default: "documents")
        bucket: S3 bucket name (default: from env or lendingwise-poc)
    
    Returns:
        dict with bucket, key, and s3_uri
    """
    bucket = bucket or BUCKET
    filename = os.path.basename(file_path)
    s3_key = generate_unique_key(prefix, filename)
    
    # Determine content type
    ext = os.path.splitext(file_path)[1].lower()
    content_types = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    content_type = content_types.get(ext, "application/octet-stream")
    
    # Upload file
    with open(file_path, "rb") as f:
        S3.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=f.read(),
            ContentType=content_type
        )
    
    print(f"[S3] Uploaded: s3://{bucket}/{s3_key}")
    
    return {
        "bucket": bucket,
        "key": s3_key,
        "s3_uri": f"s3://{bucket}/{s3_key}"
    }


def upload_bytes_to_s3(
    content: bytes,
    filename: str,
    prefix: str = "documents",
    bucket: str = None
) -> dict:
    """
    Upload bytes directly to S3.
    
    Args:
        content: File content as bytes
        filename: Original filename (for extension detection)
        prefix: S3 key prefix (default: "documents")
        bucket: S3 bucket name (default: from env or lendingwise-poc)
    
    Returns:
        dict with bucket, key, and s3_uri
    """
    bucket = bucket or BUCKET
    s3_key = generate_unique_key(prefix, filename)
    
    # Determine content type
    ext = os.path.splitext(filename)[1].lower()
    content_types = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    content_type = content_types.get(ext, "application/octet-stream")
    
    # Upload
    S3.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=content,
        ContentType=content_type
    )
    
    print(f"[S3] Uploaded: s3://{bucket}/{s3_key}")
    
    return {
        "bucket": bucket,
        "key": s3_key,
        "s3_uri": f"s3://{bucket}/{s3_key}"
    }


def delete_from_s3(bucket: str, key: str) -> bool:
    """
    Delete a file from S3.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
    
    Returns:
        True if deleted successfully
    """
    try:
        S3.delete_object(Bucket=bucket, Key=key)
        print(f"[S3] Deleted: s3://{bucket}/{key}")
        return True
    except ClientError as e:
        print(f"[S3] Delete failed: {e}")
        return False
