from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import mysql.connector
from mysql.connector import Error
import uuid
import os
import sys
import json
import traceback
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

load_dotenv()

# Import test pipeline processor
try:
    from S3_Sqs.test_pipeline_processor import (
        TestProcessRequest,
        TestProcessResponse,
        process_document_test
    )
except ImportError:
    from test_pipeline_processor import (
        TestProcessRequest,
        TestProcessResponse,
        process_document_test
    )

# Import custom agent API router
try:
    from S3_Sqs.custom_agent_api import router as custom_agent_router
except ImportError:
    from custom_agent_api import router as custom_agent_router

app = FastAPI(
    title="AI Agents Database API",
    description="API for creating AI agent records, testing document processing pipeline, and custom dynamic agents",
    version="2.0.0"
)

# Include custom agent routes
app.include_router(custom_agent_router)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - restrict this in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)


# Pydantic model for request
class AgentRecordCreate(BaseModel):
    """Model for creating a new agent record"""
    FPCID: str = Field(..., description="FPC ID", example="3580")
    document_name: str = Field(..., description="Name of the document", example="Driving license")
    agent_name: str = Field(..., description="Name of the agent", example="Identity Verification Agent")
    tool: str = Field(..., description="Tool used (e.g., ocr+llm)", example="ocr+llm")
    date: str = Field(..., description="Date in YYYY-MM-DD format", example="2025-10-20")
    checklistId: str = Field(..., description="Checklist ID", example="163")
    user_id: str = Field(..., description="User ID", example="12")
    enable_ai_agent: bool = Field(default=True, description="Enable AI agent (true/false)", example=True)
    run_agent_on_doc: bool = Field(default=True, description="Run agent on document (true/false)", example=True)
    to_cross_validate: bool = Field(default=False, description="Enable cross validation (true/false). If 1, pass valid documents for cross validation", example=False)


# Database connection function
def get_database_connection():
    """Connect to the MySQL database and return the connection object."""
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST", ""),
            port=int(os.getenv("DB_PORT", "3306")),
            user=os.getenv("DB_USER", "aiagentdb"),
            password=os.getenv("DB_PASSWORD", "Agents@1252"),
            database=os.getenv("DB_NAME", "stage_newskinny")
        )
        if connection.is_connected():
            return connection
    except Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection error: {str(e)}"
        )


def ensure_tables_exist(connection):
    """Create tblaiagents_config and tblaiagents_result tables if they do not exist."""

    # Create config table
    create_config_table = """
    CREATE TABLE IF NOT EXISTS tblaiagents_config (
        id INT PRIMARY KEY AUTO_INCREMENT,
        FPCID VARCHAR(255),
        checklistId VARCHAR(255),
        user_id VARCHAR(255),
        document_name VARCHAR(255),
        document_type VARCHAR(255),
        agent_name VARCHAR(255),
        tool VARCHAR(255),
        enable_ai_agent BOOLEAN DEFAULT TRUE,
        run_agent_on_doc BOOLEAN DEFAULT TRUE,
        to_cross_validate TINYINT(1) DEFAULT 0,
        feedback TEXT DEFAULT NULL,
        date DATE,
        Is_delete TINYINT(1) DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB;
    """

    # Create results table
    create_results_table = """
    CREATE TABLE IF NOT EXISTS tblaiagents_result (
        id INT PRIMARY KEY AUTO_INCREMENT,
        config_id INT NOT NULL,
        FPCID VARCHAR(255),
        LMRId VARCHAR(255),
        doc_id VARCHAR(255),
        checklistId VARCHAR(255),
        document_name VARCHAR(255),
        document_type VARCHAR(255),
        file_s3_location TEXT DEFAULT NULL,
        metadata_s3_path TEXT DEFAULT NULL,
        verified_result_s3_path TEXT DEFAULT NULL,
        cross_validation_report_path TEXT DEFAULT NULL,
        uploadedat DATETIME DEFAULT NULL,
        cross_validation TINYINT(1) DEFAULT 0,
        is_verified TINYINT(1) DEFAULT 0,
        document_status ENUM('pending', 'pass', 'fail', 'human_review') DEFAULT 'pending',
        Validation_status ENUM('pass', 'fail', 'human_review') DEFAULT NULL,
        borrower_type ENUM('borrower', 'co-borrower', 'unknown') DEFAULT NULL,
        validation_score INT DEFAULT NULL,
        doc_verification_result JSON DEFAULT NULL,
        Is_delete TINYINT(1) DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (config_id) REFERENCES tblaiagents_config(id) ON DELETE CASCADE
    ) ENGINE=InnoDB;
    """

    # Create detailed API logs table for agent-wise logging
    create_api_logs_table = """
    CREATE TABLE IF NOT EXISTS tblaiagents_api_logs (
        id INT PRIMARY KEY AUTO_INCREMENT,
        log_id VARCHAR(36) NOT NULL UNIQUE,
        agent_name VARCHAR(255) NOT NULL,
        agent_id INT DEFAULT NULL,
        endpoint VARCHAR(255) NOT NULL,
        http_method VARCHAR(10) NOT NULL,
        user_id VARCHAR(255) DEFAULT NULL,
        FPCID VARCHAR(255) DEFAULT NULL,
        LMRId VARCHAR(255) DEFAULT NULL,
        doc_id VARCHAR(255) DEFAULT NULL,
        checklistId VARCHAR(255) DEFAULT NULL,
        client_ip VARCHAR(45) DEFAULT NULL,
        user_agent TEXT DEFAULT NULL,
        request_headers JSON DEFAULT NULL,
        request_body JSON DEFAULT NULL,
        request_timestamp DATETIME(3) NOT NULL,
        process_start_time DATETIME(3) DEFAULT NULL,
        ocr_start_time DATETIME(3) DEFAULT NULL,
        ocr_end_time DATETIME(3) DEFAULT NULL,
        ocr_duration_ms INT DEFAULT NULL,
        classification_start_time DATETIME(3) DEFAULT NULL,
        classification_end_time DATETIME(3) DEFAULT NULL,
        classification_duration_ms INT DEFAULT NULL,
        extraction_start_time DATETIME(3) DEFAULT NULL,
        extraction_end_time DATETIME(3) DEFAULT NULL,
        extraction_duration_ms INT DEFAULT NULL,
        validation_start_time DATETIME(3) DEFAULT NULL,
        validation_end_time DATETIME(3) DEFAULT NULL,
        validation_duration_ms INT DEFAULT NULL,
        llm_calls_count INT DEFAULT 0,
        llm_total_tokens INT DEFAULT 0,
        llm_prompt_tokens INT DEFAULT 0,
        llm_completion_tokens INT DEFAULT 0,
        process_end_time DATETIME(3) DEFAULT NULL,
        total_processing_time_ms INT DEFAULT NULL,
        response_status_code INT DEFAULT NULL,
        response_status VARCHAR(50) DEFAULT NULL,
        validation_score INT DEFAULT NULL,
        document_type VARCHAR(255) DEFAULT NULL,
        error_message TEXT DEFAULT NULL,
        error_traceback TEXT DEFAULT NULL,
        pipeline_stages JSON DEFAULT NULL,
        response_summary JSON DEFAULT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_agent_name (agent_name),
        INDEX idx_user_id (user_id),
        INDEX idx_fpcid (FPCID),
        INDEX idx_request_timestamp (request_timestamp),
        INDEX idx_response_status (response_status)
    ) ENGINE=InnoDB;
    """

    try:
        cursor = connection.cursor()
        cursor.execute(create_config_table)
        cursor.execute(create_results_table)
        cursor.execute(create_api_logs_table)
        connection.commit()
        cursor.close()
    except Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create tables: {str(e)}"
        )


def get_client_ip(request: Request) -> str:
    """Extract client IP from request headers or connection."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def log_api_call(
    connection,
    log_id: str,
    agent_name: str,
    endpoint: str,
    http_method: str,
    request_timestamp: datetime,
    user_id: str = None,
    fpcid: str = None,
    lmrid: str = None,
    doc_id: str = None,
    checklist_id: str = None,
    client_ip: str = None,
    user_agent: str = None,
    request_headers: dict = None,
    request_body: dict = None,
    process_start_time: datetime = None,
    ocr_start_time: datetime = None,
    ocr_end_time: datetime = None,
    classification_start_time: datetime = None,
    classification_end_time: datetime = None,
    extraction_start_time: datetime = None,
    extraction_end_time: datetime = None,
    validation_start_time: datetime = None,
    validation_end_time: datetime = None,
    process_end_time: datetime = None,
    llm_calls_count: int = 0,
    llm_total_tokens: int = 0,
    llm_prompt_tokens: int = 0,
    llm_completion_tokens: int = 0,
    response_status_code: int = None,
    response_status: str = None,
    validation_score: int = None,
    document_type: str = None,
    error_message: str = None,
    error_traceback_str: str = None,
    pipeline_stages: list = None,
    response_summary: dict = None
) -> bool:
    """
    Log detailed API call information to tblaiagents_api_logs.
    
    Args:
        connection: Database connection
        log_id: Unique identifier for this log entry (UUID)
        agent_name: Name of the agent being called
        endpoint: API endpoint path
        http_method: HTTP method (GET, POST, etc.)
        request_timestamp: When the request was received
        user_id: User who made the request
        fpcid: FPC ID
        lmrid: LMR ID
        doc_id: Document ID
        checklist_id: Checklist ID
        client_ip: Client IP address
        user_agent: Client user agent string
        request_headers: Request headers (dict)
        request_body: Request body (dict)
        process_start_time: When processing started
        ocr_start_time/ocr_end_time: OCR stage timing
        classification_start_time/classification_end_time: Classification stage timing
        extraction_start_time/extraction_end_time: Extraction stage timing
        validation_start_time/validation_end_time: Validation stage timing
        process_end_time: When processing ended
        llm_calls_count: Number of LLM API calls made
        llm_total_tokens: Total tokens used
        llm_prompt_tokens: Prompt tokens used
        llm_completion_tokens: Completion tokens used
        response_status_code: HTTP response status code
        response_status: Validation status (pass/fail/error)
        validation_score: Validation score
        document_type: Detected document type
        error_message: Error message if any
        error_traceback_str: Error traceback if any
        pipeline_stages: List of pipeline stage details
        response_summary: Summary of response
    
    Returns:
        bool: True if logging successful, False otherwise
    """
    # Calculate durations
    ocr_duration = None
    if ocr_start_time and ocr_end_time:
        ocr_duration = int((ocr_end_time - ocr_start_time).total_seconds() * 1000)
    
    classification_duration = None
    if classification_start_time and classification_end_time:
        classification_duration = int((classification_end_time - classification_start_time).total_seconds() * 1000)
    
    extraction_duration = None
    if extraction_start_time and extraction_end_time:
        extraction_duration = int((extraction_end_time - extraction_start_time).total_seconds() * 1000)
    
    validation_duration = None
    if validation_start_time and validation_end_time:
        validation_duration = int((validation_end_time - validation_start_time).total_seconds() * 1000)
    
    total_duration = None
    if process_start_time and process_end_time:
        total_duration = int((process_end_time - process_start_time).total_seconds() * 1000)
    
    try:
        cursor = connection.cursor()
        
        insert_query = """
        INSERT INTO tblaiagents_api_logs (
            log_id, agent_name, endpoint, http_method, user_id, FPCID, LMRId, doc_id, checklistId,
            client_ip, user_agent, request_headers, request_body, request_timestamp,
            process_start_time, ocr_start_time, ocr_end_time, ocr_duration_ms,
            classification_start_time, classification_end_time, classification_duration_ms,
            extraction_start_time, extraction_end_time, extraction_duration_ms,
            validation_start_time, validation_end_time, validation_duration_ms,
            process_end_time, total_processing_time_ms, llm_calls_count, llm_total_tokens,
            llm_prompt_tokens, llm_completion_tokens,
            response_status_code, response_status, validation_score, document_type,
            error_message, error_traceback, pipeline_stages, response_summary
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s
        )
        """
        
        cursor.execute(insert_query, (
            log_id, agent_name, endpoint, http_method, user_id, fpcid, lmrid, doc_id, checklist_id,
            client_ip, user_agent,
            json.dumps(request_headers) if request_headers else None,
            json.dumps(request_body) if request_body else None,
            request_timestamp,
            process_start_time, ocr_start_time, ocr_end_time, ocr_duration,
            classification_start_time, classification_end_time, classification_duration,
            extraction_start_time, extraction_end_time, extraction_duration,
            validation_start_time, validation_end_time, validation_duration,
            process_end_time, total_duration, llm_calls_count, llm_total_tokens,
            llm_prompt_tokens, llm_completion_tokens,
            response_status_code, response_status, validation_score, document_type,
            error_message, error_traceback_str,
            json.dumps(pipeline_stages) if pipeline_stages else None,
            json.dumps(response_summary) if response_summary else None
        ))
        
        connection.commit()
        cursor.close()
        return True
    except Exception as e:
        print(f"Error logging API call: {str(e)}")
        return False


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "AI Agents Database API is running",
        "version": "1.0.0",
        "status": "healthy"
    }


@app.post("/create-agent-record", status_code=status.HTTP_201_CREATED)
async def create_agent_record(record: AgentRecordCreate):
    """
    Create a new agent configuration record in tblaiagents_config.

    All fields are required:
    - **FPCID**: FPC ID
    - **document_name**: Name of the document
    - **agent_name**: Name of the agent
    - **tool**: Tool used (e.g., ocr+llm)
    - **date**: Date in YYYY-MM-DD format
    - **checklistId**: Checklist ID
    - **user_id**: User ID
    - **enable_ai_agent**: Enable AI agent (default: true)
    - **run_agent_on_doc**: Run agent on document (default: true)
    - **to_cross_validate**: Enable cross validation (default: false). If 1, pass valid documents for cross validation

    Note: LMRId and doc_id will be provided via SQS message during document processing.
    Results will be stored in tblaiagents_result table.
    """
    connection = get_database_connection()

    try:
        ensure_tables_exist(connection)

        insert_query = """
        INSERT INTO tblaiagents_config (
            FPCID, checklistId, user_id, document_name, agent_name, tool,
            enable_ai_agent, run_agent_on_doc, to_cross_validate, date
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """

        cursor = connection.cursor()
        cursor.execute(insert_query, (
            record.FPCID,
            record.checklistId,
            record.user_id,
            record.document_name,
            record.agent_name,
            record.tool,
            record.enable_ai_agent,
            record.run_agent_on_doc,
            record.to_cross_validate,
            record.date
        ))
        connection.commit()

        # Get the auto-generated config_id
        config_id = cursor.lastrowid
        cursor.close()

        return {
            "success": True,
            "message": "Agent configuration created successfully",
            "config_id": config_id,
            "airecordid": str(config_id),  # Return config_id for use in SQS message
            "data": {
                "config_id": config_id,
                "FPCID": record.FPCID,
                "document_name": record.document_name,
                "agent_name": record.agent_name,
                "tool": record.tool,
                "date": record.date,
                "checklistId": record.checklistId,
                "user_id": record.user_id,
                "enable_ai_agent": record.enable_ai_agent,
                "run_agent_on_doc": record.run_agent_on_doc,
                "to_cross_validate": record.to_cross_validate
            }
        }

    except Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to insert data: {str(e)}"
        )
    finally:
        if connection.is_connected():
            connection.close()


@app.get("/api/documents/lmr/{lmrid}")
async def get_documents_by_lmrid(lmrid: int):
    """
    Fetch all documents for a specific LMRId.

    Returns all document records with their validation status, S3 paths, and verification results
    for the given LMRId across all FPCIDs.
    
    **Note:** Soft-deleted records (Is_delete = 1) are excluded from results.
    
    **Path Parameters:**
    - **lmrid**: LMR ID (integer)

    **Returns:**
    - 200: List of all documents for the given LMRId (excluding soft-deleted)
    - 404: No documents found
    - 500: Database error
    """
    connection = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                r.id,
                r.config_id,
                r.FPCID,
                r.LMRId,
                r.doc_id,
                c.document_name,
                c.agent_name,
                c.tool,
                r.document_status,
                r.Validation_status,
                r.borrower_type,
                r.file_s3_location,
                r.verified_result_s3_path,
                r.metadata_s3_path,
                r.cross_validation,
                r.cross_validation_report_path,
                r.doc_verification_result,
                r.validation_score,
                r.uploadedat,
                r.created_at,
                c.date,
                r.is_verified,
                c.user_id,
                c.checklistId,
                c.enable_ai_agent,
                c.run_agent_on_doc,
                c.to_cross_validate,
                c.feedback
            FROM tblaiagents_config c
            JOIN tblaiagents_result r ON c.id = r.config_id
            WHERE r.LMRId = %s
            AND (c.Is_delete IS NULL OR c.Is_delete = 0)
            AND (r.Is_delete IS NULL OR r.Is_delete = 0)
            ORDER BY r.created_at DESC
        """

        cursor.execute(query, (str(lmrid),))
        documents = cursor.fetchall()
        cursor.close()

        if not documents:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No documents found for LMRId={lmrid}"
            )

        # Format datetime fields to ISO string
        for doc in documents:
            if doc.get('uploadedat'):
                doc['uploadedat'] = doc['uploadedat'].isoformat()
            if doc.get('created_at'):
                doc['created_at'] = doc['created_at'].isoformat()
            if doc.get('date'):
                doc['date'] = doc['date'].isoformat()

        return {
            "lmrid": lmrid,
            "total_documents": len(documents),
            "documents": documents
        }

    except HTTPException:
        raise
    except Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    finally:
        if connection and connection.is_connected():
            connection.close()


@app.get("/api/airesults/{fpcid}/{lmrid}")
async def get_all_documents(fpcid: int, lmrid: int):
    """
    Fetch all documents for a specific FPCID and LMRId.

    Returns all document records with their validation status, S3 paths, and verification results.
    
    **Note:** Soft-deleted records (Is_delete = 1) are excluded from results.
    
    **Path Parameters:**
    - **fpcid**: FPC ID (integer)
    - **lmrid**: LMR ID (integer)

    **Returns:**
    - 200: List of all documents for the given FPCID/LMRId (excluding soft-deleted)
    - 404: No documents found
    - 500: Database error
    """
    connection = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                r.id,
                r.config_id,
                r.FPCID,
                r.LMRId,
                r.doc_id,
                r.checklistId,
                c.document_name,
                c.agent_name,
                c.tool,
                r.document_status,
                r.Validation_status,
                r.borrower_type,
                r.file_s3_location,
                r.verified_result_s3_path,
                r.metadata_s3_path,
                r.cross_validation,
                r.cross_validation_report_path,
                r.doc_verification_result,
                r.validation_score,
                r.uploadedat,
                r.created_at,
                c.date,
                r.is_verified,
                c.user_id,
                c.enable_ai_agent,
                c.run_agent_on_doc,
                c.to_cross_validate,
                c.feedback,
                r.Is_delete
            FROM tblaiagents_config c
            JOIN tblaiagents_result r ON c.id = r.config_id
            WHERE r.FPCID = %s AND r.LMRId = %s
            AND (c.Is_delete IS NULL OR c.Is_delete = 0)
            AND (r.Is_delete IS NULL OR r.Is_delete = 0)
            ORDER BY r.created_at DESC
        """

        cursor.execute(query, (str(fpcid), str(lmrid)))
        documents = cursor.fetchall()
        cursor.close()

        if not documents:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No documents found for FPCID={fpcid} and LMRId={lmrid}"
            )

        # Format datetime fields to ISO string
        for doc in documents:
            if doc.get('uploadedat'):
                doc['uploadedat'] = doc['uploadedat'].isoformat()
            if doc.get('created_at'):
                doc['created_at'] = doc['created_at'].isoformat()
            if doc.get('date'):
                doc['date'] = doc['date'].isoformat()

        return {
            "fpcid": fpcid,
            "lmrid": lmrid,
            "total_documents": len(documents),
            "documents": documents
        }

    except HTTPException:
        raise
    except Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    finally:
        if connection and connection.is_connected():
            connection.close()


@app.get("/api/documents/{fpcid}/{lmrid}/{doc_id}")
async def get_specific_document(fpcid: int, lmrid: int, doc_id: str):
    """
    Fetch a specific document by FPCID, LMRId, and doc_id.

    Returns the document record for the specified doc_id.
    
    **Note:** Soft-deleted records (Is_delete = 1) are excluded from results.
    
    **Path Parameters:**
    - **fpcid**: FPC ID (integer)
    - **lmrid**: LMR ID (integer)
    - **doc_id**: Document ID (string, e.g., "23", "24")

    **Returns:**
    - 200: Document details including validation status and extracted data (excluding soft-deleted)
    - 404: Document not found or has been deleted
    - 500: Database error
    """
    connection = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                r.id,
                r.config_id,
                r.FPCID,
                r.LMRId,
                r.doc_id,
                c.document_name,
                c.agent_name,
                c.tool,
                r.document_status,
                r.Validation_status,
                r.borrower_type,
                r.file_s3_location,
                r.verified_result_s3_path,
                r.metadata_s3_path,
                r.cross_validation,
                r.cross_validation_report_path,
                r.doc_verification_result,
                r.validation_score,
                r.uploadedat,
                r.created_at,
                c.date,
                r.is_verified,
                c.user_id,
                c.enable_ai_agent,
                c.run_agent_on_doc,
                c.to_cross_validate,
                c.feedback
            FROM tblaiagents_config c
            JOIN tblaiagents_result r ON c.id = r.config_id
            WHERE r.FPCID = %s AND r.LMRId = %s AND r.doc_id = %s
            AND (c.Is_delete IS NULL OR c.Is_delete = 0)
            AND (r.Is_delete IS NULL OR r.Is_delete = 0)
            ORDER BY r.created_at DESC
            LIMIT 1
        """

        cursor.execute(query, (str(fpcid), str(lmrid), doc_id))
        document = cursor.fetchone()
        cursor.close()

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with doc_id='{doc_id}' not found for FPCID={fpcid} and LMRId={lmrid}"
            )

        # Format datetime fields to ISO string
        if document.get('uploadedat'):
            document['uploadedat'] = document['uploadedat'].isoformat()
        if document.get('created_at'):
            document['created_at'] = document['created_at'].isoformat()
        if document.get('date'):
            document['date'] = document['date'].isoformat()

        return {
            "fpcid": fpcid,
            "lmrid": lmrid,
            "doc_id": doc_id,
            "document": document
        }

    except HTTPException:
        raise
    except Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    finally:
        if connection and connection.is_connected():
            connection.close()


@app.get("/api/checklist/{checklist_id}")
async def get_checklist_config(checklist_id: str):
    """
    Fetch agent configuration by checklistId from tblaiagents_config.

    Returns the agent configuration including enable_ai_agent, run_agent_on_doc,
    document_name, agent_name, tool, and to_cross_validate for the specified checklist.
    
    **Note:** Soft-deleted records (Is_delete = 1) are excluded from results.
    
    **Path Parameters:**
    - **checklist_id**: Checklist ID (string)

    **Returns:**
    - 200: Agent configuration details (excluding soft-deleted)
    - 404: No record found for the checklist or has been deleted
    - 500: Database error
    """
    connection = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                id,
                checklistId,
                enable_ai_agent,
                run_agent_on_doc,
                document_name,
                agent_name,
                tool,
                to_cross_validate,
                feedback
            FROM tblaiagents_config
            WHERE checklistId = %s
            AND (Is_delete IS NULL OR Is_delete = 0)
            ORDER BY created_at DESC
            LIMIT 1
        """

        cursor.execute(query, (checklist_id,))
        record = cursor.fetchone()
        cursor.close()

        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No record found for checklistId='{checklist_id}'"
            )

        return {
            "success": True,
            "checklistId": checklist_id,
            "data": record
        }

    except HTTPException:
        raise
    except Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    finally:
        if connection and connection.is_connected():
            connection.close()


@app.get("/api/config/{fpcid}/{checklist_id}")
async def get_config_by_fpcid_checklist(fpcid: str, checklist_id: str):
    """
    Fetch agent configuration by FPCID and checklistId from tblaiagents_config.

    Returns the agent configuration including all fields for the specified FPCID and checklistId.
    
    **Note:** Soft-deleted records (Is_delete = 1) are excluded from results.
    
    **Path Parameters:**
    - **fpcid**: FPC ID (string)
    - **checklist_id**: Checklist ID (string)

    **Returns:**
    - 200: Agent configuration details (excluding soft-deleted)
    - 404: No record found for the FPCID and checklistId combination or has been deleted
    - 500: Database error
    """
    connection = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                id,
                FPCID,
                checklistId,
                user_id,
                document_name,
                document_type,
                agent_name,
                tool,
                enable_ai_agent,
                run_agent_on_doc,
                to_cross_validate,
                feedback,
                date,
                created_at,
                updated_at
            FROM tblaiagents_config
            WHERE FPCID = %s AND checklistId = %s
            AND (Is_delete IS NULL OR Is_delete = 0)
            ORDER BY created_at DESC
            LIMIT 1
        """

        cursor.execute(query, (fpcid, checklist_id))
        record = cursor.fetchone()
        cursor.close()

        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No record found for FPCID='{fpcid}' and checklistId='{checklist_id}'"
            )

        # Format datetime fields to ISO string
        if record.get('created_at'):
            record['created_at'] = record['created_at'].isoformat()
        if record.get('updated_at'):
            record['updated_at'] = record['updated_at'].isoformat()
        if record.get('date'):
            record['date'] = record['date'].isoformat()

        return {
            "success": True,
            "fpcid": fpcid,
            "checklistId": checklist_id,
            "data": record
        }

    except HTTPException:
        raise
    except Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    finally:
        if connection and connection.is_connected():
            connection.close()


# Pydantic model for checklist update
class ChecklistConfigUpdate(BaseModel):
    """Model for updating checklist configuration"""
    enable_ai_agent: bool = Field(..., description="Enable AI agent (true/false)", example=True)
    run_agent_on_doc: bool = Field(..., description="Run agent on document (true/false)", example=True)
    document_name: str = Field(..., description="Name of the document", example="Driving license")
    agent_name: str = Field(..., description="Name of the agent", example="Identity Verification Agent")
    tool: str = Field(..., description="Tool used (e.g., ocr+llm)", example="ocr+llm")
    to_cross_validate: bool = Field(..., description="Enable cross validation (true/false). If 1, pass valid documents for cross validation", example=False)


# Pydantic model for feedback
class FeedbackRequest(BaseModel):
    """Model for saving feedback to config record"""
    feedback: str = Field(..., description="Feedback text to save", example="Document processing accuracy is good, but needs improvement in date extraction")


@app.put("/api/checklist/{checklist_id}")
async def update_checklist_config(checklist_id: str, config: ChecklistConfigUpdate):
    """
    Update agent configuration by checklistId in tblaiagents_config.

    Updates enable_ai_agent, run_agent_on_doc, document_name, agent_name, tool, and to_cross_validate
    for all config records with the specified checklistId.

    **Path Parameters:**
    - **checklist_id**: Checklist ID (string)

    **Request Body:**
    - **enable_ai_agent**: Enable AI agent (true/false)
    - **run_agent_on_doc**: Run agent on document (true/false)
    - **document_name**: Name of the document
    - **agent_name**: Name of the agent
    - **tool**: Tool used (e.g., ocr+llm)
    - **to_cross_validate**: Enable cross validation (true/false). If 1, pass valid documents for cross validation

    **Returns:**
    - 200: Configuration updated successfully
    - 404: No records found for the checklist
    - 500: Database error
    """
    connection = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor()

        # First check if records exist with this checklistId
        check_query = "SELECT COUNT(*) as count FROM tblaiagents_config WHERE checklistId = %s"
        cursor.execute(check_query, (checklist_id,))
        result = cursor.fetchone()

        if result[0] == 0:
            cursor.close()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No records found for checklistId='{checklist_id}'"
            )

        # Update the config records
        update_query = """
            UPDATE tblaiagents_config
            SET
                enable_ai_agent = %s,
                run_agent_on_doc = %s,
                document_name = %s,
                agent_name = %s,
                tool = %s,
                to_cross_validate = %s,
                updated_at = NOW()
            WHERE checklistId = %s
        """

        cursor.execute(update_query, (
            config.enable_ai_agent,
            config.run_agent_on_doc,
            config.document_name,
            config.agent_name,
            config.tool,
            config.to_cross_validate,
            checklist_id
        ))

        connection.commit()
        rows_affected = cursor.rowcount
        cursor.close()

        return {
            "success": True,
            "message": f"Configuration updated successfully for checklistId '{checklist_id}'",
            "rows_affected": rows_affected,
            "data": {
                "checklistId": checklist_id,
                "enable_ai_agent": config.enable_ai_agent,
                "run_agent_on_doc": config.run_agent_on_doc,
                "document_name": config.document_name,
                "agent_name": config.agent_name,
                "tool": config.tool,
                "to_cross_validate": config.to_cross_validate
            }
        }

    except HTTPException:
        raise
    except Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    finally:
        if connection and connection.is_connected():
            connection.close()


@app.get("/api/record/{config_id}")
async def get_record_by_id(config_id: int):
    """
    Fetch a specific configuration record by its config_id.

    Returns the complete configuration details and associated results for the specified config_id.
    
    **Note:** Soft-deleted records (Is_delete = 1) are excluded from results.
    
    **Path Parameters:**
    - **config_id**: Config ID (integer, e.g., 1, 2, 3)

    **Returns:**
    - 200: Record details including config and results (excluding soft-deleted)
    - 404: Record not found or has been deleted
    - 500: Database error
    """
    connection = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Get config (excluding soft-deleted)
        config_query = """
            SELECT
                id,
                FPCID,
                checklistId,
                user_id,
                document_name,
                agent_name,
                tool,
                enable_ai_agent,
                run_agent_on_doc,
                to_cross_validate,
                feedback,
                date,
                created_at,
                updated_at
            FROM tblaiagents_config
            WHERE id = %s
            AND (Is_delete IS NULL OR Is_delete = 0)
        """

        cursor.execute(config_query, (config_id,))
        config = cursor.fetchone()

        if not config:
            cursor.close()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Config record with id='{config_id}' not found or has been deleted"
            )
        
        # Get associated results (excluding soft-deleted)
        results_query = """
            SELECT
                id,
                FPCID,
                LMRId,
                doc_id,
                document_status,
                Validation_status,
                borrower_type,
                file_s3_location,
                metadata_s3_path,
                verified_result_s3_path,
                cross_validation_report_path,
                uploadedat,
                cross_validation,
                is_verified,
                validation_score,
                doc_verification_result,
                created_at,
                updated_at
            FROM tblaiagents_result
            WHERE config_id = %s
            AND (Is_delete IS NULL OR Is_delete = 0)
        """

        cursor.execute(results_query, (config_id,))
        results = cursor.fetchall()
        cursor.close()

        # Format datetime fields
        if config.get('created_at'):
            config['created_at'] = config['created_at'].isoformat()
        if config.get('updated_at'):
            config['updated_at'] = config['updated_at'].isoformat()
        if config.get('date'):
            config['date'] = config['date'].isoformat()

        for result in results:
            if result.get('uploadedat'):
                result['uploadedat'] = result['uploadedat'].isoformat()
            if result.get('created_at'):
                result['created_at'] = result['created_at'].isoformat()
            if result.get('updated_at'):
                result['updated_at'] = result['updated_at'].isoformat()

        return {
            "success": True,
            "config_id": config_id,
            "config": config,
            "results": results
        }

    except HTTPException:
        raise
    except Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    finally:
        if connection and connection.is_connected():
            connection.close()


@app.delete("/api/deleteconfig/{lmrid}/{fpcid}/{config_id}", status_code=status.HTTP_200_OK)
async def delete_config_by_lmrid_fpcid_configid(lmrid: str, fpcid: str, config_id: int):
    """
    Delete a configuration record from tblaiagents_config based on LMRId, FPCID, and config_id.

    This endpoint joins tblaiagents_config and tblaiagents_result to verify the records exist
    and then deletes the configuration record. This will also cascade delete all associated
    records from tblaiagents_result due to the foreign key constraint with ON DELETE CASCADE.

    **Path Parameters:**
    - **lmrid**: LMR ID (string, e.g., "1", "2")
    - **fpcid**: FPC ID (string, e.g., "3580")
    - **config_id**: Config ID (integer, e.g., 1, 2, 3)

    **Returns:**
    - 200: Configuration deleted successfully
    - 404: Configuration not found matching the criteria
    - 500: Database error
    """
    connection = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor()

        # First check if the record exists by joining both tables
        check_query = """
            SELECT c.id, c.FPCID, r.LMRId, COUNT(r.id) as result_count
            FROM tblaiagents_config c
            JOIN tblaiagents_result r ON c.id = r.config_id
            WHERE c.id = %s AND c.FPCID = %s AND r.LMRId = %s
            GROUP BY c.id, c.FPCID, r.LMRId
        """
        cursor.execute(check_query, (config_id, fpcid, lmrid))
        record = cursor.fetchone()

        if not record:
            cursor.close()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Config record with config_id='{config_id}', FPCID='{fpcid}', and LMRId='{lmrid}' not found"
            )

        result_count = record[3] if record else 0

        # Delete the config record (this will cascade delete associated results)
        # We need to ensure we're deleting the correct config that matches all criteria
        delete_query = """
            DELETE c FROM tblaiagents_config c
            INNER JOIN tblaiagents_result r ON c.id = r.config_id
            WHERE c.id = %s AND c.FPCID = %s AND r.LMRId = %s
        """
        cursor.execute(delete_query, (config_id, fpcid, lmrid))
        connection.commit()
        rows_affected = cursor.rowcount
        cursor.close()

        if rows_affected == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Config record with config_id='{config_id}', FPCID='{fpcid}', and LMRId='{lmrid}' not found or could not be deleted"
            )

        return {
            "success": True,
            "message": f"Configuration with config_id='{config_id}', FPCID='{fpcid}', and LMRId='{lmrid}' deleted successfully",
            "config_id": config_id,
            "fpcid": fpcid,
            "lmrid": lmrid,
            "associated_results_deleted": result_count
        }

    except HTTPException:
        raise
    except Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    finally:
        if connection and connection.is_connected():
            connection.close()


@app.delete("/api/deleteresult/{fpcid}/{lmrid}/{doc_id}", status_code=status.HTTP_200_OK)
async def delete_result_by_fpcid_lmrid_doc_id(fpcid: str, lmrid: str, doc_id: str):
    """
    Delete a result record from tblaiagents_result by FPCID, LMRId, and doc_id.

    **Path Parameters:**
    - **fpcid**: FPC ID (string, e.g., "3580")
    - **lmrid**: LMR ID (string, e.g., "1")
    - **doc_id**: Document ID (string, e.g., "23", "24")

    **Returns:**
    - 200: Result deleted successfully
    - 404: Result not found
    - 500: Database error
    """
    connection = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor()

        # First check if the record exists
        check_query = "SELECT id, FPCID, LMRId, doc_id FROM tblaiagents_result WHERE FPCID = %s AND LMRId = %s AND doc_id = %s"
        cursor.execute(check_query, (fpcid, lmrid, doc_id))
        records = cursor.fetchall()

        if not records:
            cursor.close()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No result records found with FPCID='{fpcid}', LMRId='{lmrid}', and doc_id='{doc_id}'"
            )

        # Delete all records matching FPCID, LMRId, and doc_id
        delete_query = "DELETE FROM tblaiagents_result WHERE FPCID = %s AND LMRId = %s AND doc_id = %s"
        cursor.execute(delete_query, (fpcid, lmrid, doc_id))
        connection.commit()
        rows_affected = cursor.rowcount
        cursor.close()

        return {
            "success": True,
            "message": f"Result record(s) with FPCID='{fpcid}', LMRId='{lmrid}', and doc_id='{doc_id}' deleted successfully",
            "fpcid": fpcid,
            "lmrid": lmrid,
            "doc_id": doc_id,
            "rows_deleted": rows_affected
        }

    except HTTPException:
        raise
    except Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    finally:
        if connection and connection.is_connected():
            connection.close()


@app.patch("/api/soft-delete/result", status_code=status.HTTP_200_OK)
async def soft_delete_result_by_keys(
    fpcid: str,
    lmrid: str,
    doc_id: str,
    checklistId: Optional[str] = None
):
    """
    Soft delete result record(s) based on FPCID, LMRId, and doc_id.
    
    This endpoint sets Is_delete = 1 for result records matching the criteria.
    Config records remain unaffected.
    
    **Query Parameters:**
    - **fpcid**: FPC ID (required)
    - **lmrid**: LMR ID (required)
    - **doc_id**: Document ID (required)
    - **checklistId**: Checklist ID (optional) - if provided, only deletes records with this checklistId
    
    **Behavior:**
    - If checklistId is provided: Only soft deletes records with exact match
    - If checklistId is NOT provided: Soft deletes ALL records matching FPCID + LMRId + doc_id
    
    **Example:**
    ```
    PATCH /api/soft-delete/result?fpcid=3580&lmrid=1&doc_id=23&checklistId=163
    ```
    """
    connection = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Build query based on whether checklistId is provided
        if checklistId:
            # Specific checklistId - only delete that specific record
            query = """
                SELECT id, FPCID, LMRId, doc_id, checklistId, document_name, Is_delete
                FROM tblaiagents_result 
                WHERE FPCID = %s AND LMRId = %s AND doc_id = %s AND checklistId = %s
            """
            params = (fpcid, lmrid, doc_id, checklistId)
        else:
            # No checklistId - delete all records with matching FPCID, LMRId, doc_id
            query = """
                SELECT id, FPCID, LMRId, doc_id, checklistId, document_name, Is_delete
                FROM tblaiagents_result 
                WHERE FPCID = %s AND LMRId = %s AND doc_id = %s
            """
            params = (fpcid, lmrid, doc_id)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No result records found for FPCID={fpcid}, LMRId={lmrid}, doc_id={doc_id}" + 
                       (f", checklistId={checklistId}" if checklistId else "")
            )
        
        # Check if any are already deleted
        already_deleted = [r for r in results if r.get('Is_delete') == 1]
        to_delete = [r for r in results if r.get('Is_delete') != 1]
        
        if not to_delete:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"All matching records are already deleted"
            )
        
        # Perform soft delete
        if checklistId:
            update_query = """
                UPDATE tblaiagents_result 
                SET Is_delete = 1, updated_at = CURRENT_TIMESTAMP
                WHERE FPCID = %s AND LMRId = %s AND doc_id = %s AND checklistId = %s
            """
            update_params = (fpcid, lmrid, doc_id, checklistId)
        else:
            update_query = """
                UPDATE tblaiagents_result 
                SET Is_delete = 1, updated_at = CURRENT_TIMESTAMP
                WHERE FPCID = %s AND LMRId = %s AND doc_id = %s
            """
            update_params = (fpcid, lmrid, doc_id)
        
        cursor.execute(update_query, update_params)
        connection.commit()
        
        return {
            "success": True,
            "message": f"Successfully soft deleted {len(to_delete)} record(s)",
            "deleted_records": {
                "count": len(to_delete),
                "FPCID": fpcid,
                "LMRId": lmrid,
                "doc_id": doc_id,
                "checklistId": checklistId,
                "deleted_checklist_ids": [r['checklistId'] for r in to_delete]
            },
            "already_deleted": {
                "count": len(already_deleted),
                "checklist_ids": [r['checklistId'] for r in already_deleted]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if connection:
            connection.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to soft delete result: {str(e)}"
        )
    finally:
        if connection:
            cursor.close()
            connection.close()


@app.patch("/api/soft-delete/config", status_code=status.HTTP_200_OK)
async def soft_delete_config_by_keys(
    fpcid: str,
    checklistId: str
):
    """
    Soft delete config record(s) and all associated result records (CASCADING).
    
    This endpoint sets Is_delete = 1 for:
    1. Config record(s) in tblaiagents_config matching FPCID + checklistId
    2. ALL associated result records in tblaiagents_result (based on config_id)
    
    **Query Parameters:**
    - **fpcid**: FPC ID (required)
    - **checklistId**: Checklist ID (required)
    
    **Cascading Behavior:**
    - Deletes config record
    - Automatically deletes ALL result records linked to that config (via config_id)
    
    **Example:**
    ```
    PATCH /api/soft-delete/config?fpcid=3580&checklistId=163
    ```
    """
    connection = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Check if config exists and is not already deleted
        cursor.execute("""
            SELECT id, FPCID, checklistId, document_name, agent_name, Is_delete
            FROM tblaiagents_config 
            WHERE FPCID = %s AND checklistId = %s
        """, (fpcid, checklistId))
        
        configs = cursor.fetchall()
        
        if not configs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No config records found for FPCID={fpcid}, checklistId={checklistId}"
            )
        
        # Filter out already deleted configs
        active_configs = [c for c in configs if c.get('Is_delete') != 1]
        
        if not active_configs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"All matching config records are already deleted"
            )
        
        # Get config IDs to delete
        config_ids = [c['id'] for c in active_configs]
        
        # Count associated result records that will be affected
        placeholders = ','.join(['%s'] * len(config_ids))
        cursor.execute(f"""
            SELECT COUNT(*) as count
            FROM tblaiagents_result 
            WHERE config_id IN ({placeholders}) AND Is_delete = 0
        """, config_ids)
        
        result_count = cursor.fetchone()['count']
        
        # Soft delete the config records
        cursor.execute(f"""
            UPDATE tblaiagents_config 
            SET Is_delete = 1, updated_at = CURRENT_TIMESTAMP
            WHERE FPCID = %s AND checklistId = %s AND Is_delete = 0
        """, (fpcid, checklistId))
        
        # Soft delete all associated result records (CASCADING)
        cursor.execute(f"""
            UPDATE tblaiagents_result 
            SET Is_delete = 1, updated_at = CURRENT_TIMESTAMP
            WHERE config_id IN ({placeholders}) AND Is_delete = 0
        """, config_ids)
        
        connection.commit()
        
        return {
            "success": True,
            "message": f"Successfully soft deleted {len(active_configs)} config record(s) and {result_count} associated result record(s)",
            "deleted_configs": {
                "count": len(active_configs),
                "FPCID": fpcid,
                "checklistId": checklistId,
                "config_ids": config_ids
            },
            "affected_results_count": result_count,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if connection:
            connection.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to soft delete config: {str(e)}"
        )
    finally:
        if connection:
            cursor.close()
            connection.close()


@app.patch("/api/soft-delete/results/bulk", status_code=status.HTTP_200_OK)
async def soft_delete_results_bulk(result_ids: List[int]):
    """
    Soft delete multiple result records at once.
    
    This endpoint sets Is_delete = 1 for multiple result records.
    Config records remain unaffected.
    
    **Request Body:**
    ```json
    [123, 456, 789]
    ```
    
    **Returns:**
    - Success message with count of deleted records and any failures
    """
    if not result_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="result_ids list cannot be empty"
        )
    
    connection = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Get existing records
        placeholders = ','.join(['%s'] * len(result_ids))
        cursor.execute(f"""
            SELECT id, FPCID, LMRId, document_name, Is_delete
            FROM tblaiagents_result 
            WHERE id IN ({placeholders})
        """, result_ids)
        
        existing_records = cursor.fetchall()
        existing_ids = {r['id'] for r in existing_records}
        already_deleted = [r['id'] for r in existing_records if r.get('Is_delete') == 1]
        to_delete = [r['id'] for r in existing_records if r.get('Is_delete') != 1]
        not_found = [rid for rid in result_ids if rid not in existing_ids]
        
        # Perform soft delete for valid records
        if to_delete:
            placeholders_delete = ','.join(['%s'] * len(to_delete))
            cursor.execute(f"""
                UPDATE tblaiagents_result 
                SET Is_delete = 1, updated_at = CURRENT_TIMESTAMP
                WHERE id IN ({placeholders_delete})
            """, to_delete)
            
            connection.commit()
        
        return {
            "success": True,
            "message": f"Bulk soft delete completed",
            "summary": {
                "total_requested": len(result_ids),
                "successfully_deleted": len(to_delete),
                "already_deleted": len(already_deleted),
                "not_found": len(not_found)
            },
            "details": {
                "deleted_ids": to_delete,
                "already_deleted_ids": already_deleted,
                "not_found_ids": not_found
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        if connection:
            connection.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to bulk soft delete results: {str(e)}"
        )
    finally:
        if connection:
            cursor.close()
            connection.close()


@app.patch("/api/restore/result", status_code=status.HTTP_200_OK)
async def restore_result_by_keys(
    fpcid: str,
    lmrid: str,
    doc_id: str,
    checklistId: Optional[str] = None
):
    """
    Restore soft-deleted result record(s) based on FPCID, LMRId, and doc_id.
    
    This endpoint sets Is_delete = 0 for result records matching the criteria.
    
    **Query Parameters:**
    - **fpcid**: FPC ID (required)
    - **lmrid**: LMR ID (required)
    - **doc_id**: Document ID (required)
    - **checklistId**: Checklist ID (optional) - if provided, only restores records with this checklistId
    
    **Behavior:**
    - If checklistId is provided: Only restores records with exact match
    - If checklistId is NOT provided: Restores ALL records matching FPCID + LMRId + doc_id
    
    **Use Case - Review Process:**
    When reviewing deleted records:
    - Original records: checklistId 1,2,3,4 (Is_delete = 1)
    - During review, new item added: checklistId 5 (creates new row, Is_delete = 0)
    - To restore only reviewed items: Use checklistId parameter (1,2,3,4)
    - This keeps new item (5) separate and only restores the specific reviewed items
    
    **Example:**
    ```
    PATCH /api/restore/result?fpcid=3580&lmrid=1&doc_id=23&checklistId=163
    ```
    """
    connection = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Build query based on whether checklistId is provided
        if checklistId:
            # Specific checklistId - only restore that specific record
            query = """
                SELECT id, FPCID, LMRId, doc_id, checklistId, document_name, Is_delete
                FROM tblaiagents_result 
                WHERE FPCID = %s AND LMRId = %s AND doc_id = %s AND checklistId = %s
            """
            params = (fpcid, lmrid, doc_id, checklistId)
        else:
            # No checklistId - restore all records with matching FPCID, LMRId, doc_id
            query = """
                SELECT id, FPCID, LMRId, doc_id, checklistId, document_name, Is_delete
                FROM tblaiagents_result 
                WHERE FPCID = %s AND LMRId = %s AND doc_id = %s
            """
            params = (fpcid, lmrid, doc_id)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No result records found for FPCID={fpcid}, LMRId={lmrid}, doc_id={doc_id}" + 
                       (f", checklistId={checklistId}" if checklistId else "")
            )
        
        # Check if any are not deleted (already active)
        already_active = [r for r in results if r.get('Is_delete') != 1]
        to_restore = [r for r in results if r.get('Is_delete') == 1]
        
        if not to_restore:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"All matching records are already active (not deleted)"
            )
        
        # Perform restore
        if checklistId:
            update_query = """
                UPDATE tblaiagents_result 
                SET Is_delete = 0, updated_at = CURRENT_TIMESTAMP
                WHERE FPCID = %s AND LMRId = %s AND doc_id = %s AND checklistId = %s AND Is_delete = 1
            """
            update_params = (fpcid, lmrid, doc_id, checklistId)
        else:
            update_query = """
                UPDATE tblaiagents_result 
                SET Is_delete = 0, updated_at = CURRENT_TIMESTAMP
                WHERE FPCID = %s AND LMRId = %s AND doc_id = %s AND Is_delete = 1
            """
            update_params = (fpcid, lmrid, doc_id)
        
        cursor.execute(update_query, update_params)
        connection.commit()
        
        return {
            "success": True,
            "message": f"Successfully restored {len(to_restore)} record(s)",
            "restored_records": {
                "count": len(to_restore),
                "FPCID": fpcid,
                "LMRId": lmrid,
                "doc_id": doc_id,
                "checklistId": checklistId,
                "restored_checklist_ids": [r['checklistId'] for r in to_restore]
            },
            "already_active": {
                "count": len(already_active),
                "checklist_ids": [r['checklistId'] for r in already_active]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if connection:
            connection.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restore result: {str(e)}"
        )
    finally:
        if connection:
            cursor.close()
            connection.close()


@app.patch("/api/restore/config", status_code=status.HTTP_200_OK)
async def restore_config_by_keys(
    fpcid: str,
    checklistId: str
):
    """
    Restore soft-deleted config record(s) and all associated result records (CASCADING).
    
    This endpoint sets Is_delete = 0 for:
    1. Config record(s) in tblaiagents_config matching FPCID + checklistId
    2. ALL associated result records in tblaiagents_result (based on config_id)
    
    **Query Parameters:**
    - **fpcid**: FPC ID (required)
    - **checklistId**: Checklist ID (required)
    
    **Cascading Behavior:**
    - Restores config record
    - Automatically restores ALL result records linked to that config (via config_id)
    
    **Example:**
    ```
    PATCH /api/restore/config?fpcid=3580&checklistId=163
    ```
    """
    connection = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Check if config exists and is deleted
        cursor.execute("""
            SELECT id, FPCID, checklistId, document_name, agent_name, Is_delete
            FROM tblaiagents_config 
            WHERE FPCID = %s AND checklistId = %s
        """, (fpcid, checklistId))
        
        configs = cursor.fetchall()
        
        if not configs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No config records found for FPCID={fpcid}, checklistId={checklistId}"
            )
        
        # Filter out already active configs
        deleted_configs = [c for c in configs if c.get('Is_delete') == 1]
        
        if not deleted_configs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"All matching config records are already active (not deleted)"
            )
        
        # Get config IDs to restore
        config_ids = [c['id'] for c in deleted_configs]
        
        # Count associated deleted result records
        placeholders = ','.join(['%s'] * len(config_ids))
        cursor.execute(f"""
            SELECT COUNT(*) as count
            FROM tblaiagents_result 
            WHERE config_id IN ({placeholders}) AND Is_delete = 1
        """, config_ids)
        
        result_count = cursor.fetchone()['count']
        
        # Restore the config records
        cursor.execute(f"""
            UPDATE tblaiagents_config 
            SET Is_delete = 0, updated_at = CURRENT_TIMESTAMP
            WHERE FPCID = %s AND checklistId = %s AND Is_delete = 1
        """, (fpcid, checklistId))
        
        # Restore all associated result records (CASCADING)
        cursor.execute(f"""
            UPDATE tblaiagents_result 
            SET Is_delete = 0, updated_at = CURRENT_TIMESTAMP
            WHERE config_id IN ({placeholders}) AND Is_delete = 1
        """, config_ids)
        
        connection.commit()
        
        return {
            "success": True,
            "message": f"Successfully restored {len(deleted_configs)} config record(s) and {result_count} associated result record(s)",
            "restored_configs": {
                "count": len(deleted_configs),
                "FPCID": fpcid,
                "checklistId": checklistId,
                "config_ids": config_ids
            },
            "affected_results_count": result_count,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if connection:
            connection.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restore config: {str(e)}"
        )
    finally:
        if connection:
            cursor.close()
            connection.close()


@app.get("/api/deleted/results", status_code=status.HTTP_200_OK)
async def get_deleted_results(
    fpcid: Optional[str] = None,
    lmrid: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    Get all soft-deleted result records with optional filtering.
    
    **Query Parameters:**
    - **fpcid**: Filter by FPCID (optional)
    - **lmrid**: Filter by LMRId (optional)
    - **limit**: Maximum number of records to return (default: 100)
    - **offset**: Number of records to skip (default: 0)
    
    **Returns:**
    - List of soft-deleted result records
    """
    connection = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Build query with filters
        query = """
            SELECT r.*, c.document_name as config_document_name, c.agent_name
            FROM tblaiagents_result r
            LEFT JOIN tblaiagents_config c ON r.config_id = c.id
            WHERE r.Is_delete = 1
        """
        params = []
        
        if fpcid:
            query += " AND r.FPCID = %s"
            params.append(fpcid)
        
        if lmrid:
            query += " AND r.LMRId = %s"
            params.append(lmrid)
        
        query += " ORDER BY r.updated_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # Get total count
        count_query = "SELECT COUNT(*) as total FROM tblaiagents_result WHERE Is_delete = 1"
        count_params = []
        
        if fpcid:
            count_query += " AND FPCID = %s"
            count_params.append(fpcid)
        
        if lmrid:
            count_query += " AND LMRId = %s"
            count_params.append(lmrid)
        
        cursor.execute(count_query, count_params)
        total = cursor.fetchone()['total']
        
        return {
            "success": True,
            "data": results,
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "returned": len(results)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch deleted results: {str(e)}"
        )
    finally:
        if connection:
            cursor.close()
            connection.close()


@app.get("/api/deleted/configs", status_code=status.HTTP_200_OK)
async def get_deleted_configs(
    fpcid: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    Get all soft-deleted config records with optional filtering.
    
    **Query Parameters:**
    - **fpcid**: Filter by FPCID (optional)
    - **limit**: Maximum number of records to return (default: 100)
    - **offset**: Number of records to skip (default: 0)
    
    **Returns:**
    - List of soft-deleted config records
    """
    connection = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Build query with filters
        query = """
            SELECT c.*, 
                   (SELECT COUNT(*) FROM tblaiagents_result WHERE config_id = c.id AND Is_delete = 1) as deleted_results_count
            FROM tblaiagents_config c
            WHERE c.Is_delete = 1
        """
        params = []
        
        if fpcid:
            query += " AND c.FPCID = %s"
            params.append(fpcid)
        
        query += " ORDER BY c.updated_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        configs = cursor.fetchall()
        
        # Get total count
        count_query = "SELECT COUNT(*) as total FROM tblaiagents_config WHERE Is_delete = 1"
        count_params = []
        
        if fpcid:
            count_query += " AND FPCID = %s"
            count_params.append(fpcid)
        
        cursor.execute(count_query, count_params)
        total = cursor.fetchone()['total']
        
        return {
            "success": True,
            "data": configs,
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "returned": len(configs)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch deleted configs: {str(e)}"
        )
    finally:
        if connection:
            cursor.close()
            connection.close()


@app.patch("/api/feedback/{fpcid}", status_code=status.HTTP_200_OK)
async def save_feedback(fpcid: str, feedback_request: FeedbackRequest):
    """
    Save feedback to configuration record(s) in tblaiagents_config.
    
    This endpoint allows you to add or update feedback for all agent configurations
    associated with a specific FPCID. The feedback is stored as text and can be updated multiple times.
    
    **Path Parameters:**
    - **fpcid**: FPC ID (string)
    
    **Request Body:**
    - **feedback**: Feedback text (required)
    
    **Returns:**
    - 200: Feedback saved successfully
    - 404: No configuration records found for the FPCID
    - 500: Database error
    
    **Example Request:**
    ```json
    {
      "feedback": "Document processing accuracy is good, but needs improvement in date extraction"
    }
    ```
    """
    connection = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)
        
        # First check if any records exist for this FPCID
        check_query = """
            SELECT id, FPCID, checklistId, document_name, feedback
            FROM tblaiagents_config
            WHERE FPCID = %s
            AND (Is_delete IS NULL OR Is_delete = 0)
        """
        
        cursor.execute(check_query, (fpcid,))
        records = cursor.fetchall()
        
        if not records:
            cursor.close()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No configuration records found for FPCID='{fpcid}'"
            )
        
        # Update the feedback for all records with this FPCID
        update_query = """
            UPDATE tblaiagents_config
            SET feedback = %s, updated_at = NOW()
            WHERE FPCID = %s
            AND (Is_delete IS NULL OR Is_delete = 0)
        """
        
        cursor.execute(update_query, (feedback_request.feedback, fpcid))
        connection.commit()
        rows_affected = cursor.rowcount
        cursor.close()
        
        if rows_affected == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Failed to update feedback for FPCID='{fpcid}'"
            )
        
        return {
            "success": True,
            "message": f"Feedback saved successfully for {rows_affected} configuration record(s)",
            "fpcid": fpcid,
            "rows_updated": rows_affected,
            "feedback": feedback_request.feedback,
            "updated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Error as e:
        if connection:
            connection.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    finally:
        if connection and connection.is_connected():
            connection.close()


@app.post("/processor/process-document", response_model=TestProcessResponse, status_code=status.HTTP_200_OK)
async def test_process_document(request: TestProcessRequest):
    """
    Test endpoint for processing documents through the complete pipeline without SQS/DB.

    This endpoint is designed for quick testing and development. It:
    - Does NOT poll SQS queue
    - Does NOT save results to database
    - Does NOT save results to S3
    - Returns complete pipeline results immediately
    - Auto-detects document name from OCR if not provided

    **Required Parameters:**
    - **FPCID**: FPC ID
    - **s3_file_url**: S3 file URL (s3://bucket/key)
    - **agent_name**: Name of the agent (e.g., "Identity Verification Agent")
    - **tool**: Tool to use ("ocr+llm" or "llm")

    **Optional Parameters:**
    - **document_name**: Name of the document (optional - will be auto-detected from OCR if not provided)
    - **LMRId**: LMR ID (default: "1")
    - **checklistId**: Checklist ID
    - **user_id**: User ID
    - **doc_id**: Document ID

    **Example Request (with document_name):**
    ```json
    {
      "FPCID": "3580",
      "s3_file_url": "s3://lendingwise-aiagent/LMRFileDocNew/3580/2024/12/20/1/upload/document/license.jpg",
      "document_name": "Driver's License",
      "agent_name": "Identity Verification Agent",
      "tool": "ocr+llm",
      "LMRId": "1",
      "checklistId": "163",
      "user_id": "12",
      "doc_id": "23"
    }
    ```

    **Example Request (without document_name - auto-detection):**
    ```json
    {
      "FPCID": "35899",
      "s3_file_url": "s3://lendingwise-aiagent/agent_testing/5543/0/Drivers license/test_Drivers license_2025-11-21_02-33-27_692015c765f98.jpg",
      "agent_name": "Identity Verification Agent",
      "tool": "ocr+llm"
    }
    ```

    **Returns:**
    - Complete pipeline results including:
      - Ingestion details
      - OCR results (with auto-detected document_name if not provided)
      - Classification results
      - Extraction results (extracted fields)
      - Validation results
      - Overall status (pass/fail/human_verification_needed)
    """
    return await process_document_test(request)


# ==================== Detailed Agent Logs Endpoints ====================

@app.get("/api/logs/agent/{agent_name}", status_code=status.HTTP_200_OK)
async def get_agent_detailed_logs(
    agent_name: str,
    user_id: Optional[str] = None,
    fpcid: Optional[str] = None,
    status_filter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """
    Get detailed API logs for a specific agent with comprehensive timing information.
    
    **Path Parameters:**
    - **agent_name**: Name of the agent to get logs for
    
    **Query Parameters:**
    - **user_id**: Filter by user ID (optional)
    - **fpcid**: Filter by FPCID (optional)
    - **status_filter**: Filter by response status - 'pass', 'fail', 'error' (optional)
    - **start_date**: Start date filter (YYYY-MM-DD format) (optional)
    - **end_date**: End date filter (YYYY-MM-DD format) (optional)
    - **limit**: Maximum records to return (default: 50, max: 500)
    - **offset**: Pagination offset (default: 0)
    
    **Returns:**
    - Detailed logs including:
      - User who hit the API
      - Request timestamp
      - Process start/end times
      - Individual stage timings (OCR, Classification, Extraction, Validation)
      - Total processing time
      - Response status and validation score
      - Error details if any
    """
    connection = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Ensure table exists
        ensure_tables_exist(connection)
        
        # Build query with filters
        query = """
            SELECT 
                log_id,
                agent_name,
                endpoint,
                http_method,
                user_id,
                FPCID,
                LMRId,
                doc_id,
                checklistId,
                client_ip,
                request_timestamp,
                process_start_time,
                ocr_start_time,
                ocr_end_time,
                ocr_duration_ms,
                classification_start_time,
                classification_end_time,
                classification_duration_ms,
                extraction_start_time,
                extraction_end_time,
                extraction_duration_ms,
                validation_start_time,
                validation_end_time,
                validation_duration_ms,
                process_end_time,
                total_processing_time_ms,
                llm_calls_count,
                llm_total_tokens,
                response_status_code,
                response_status,
                validation_score,
                document_type,
                error_message,
                pipeline_stages,
                created_at
            FROM tblaiagents_api_logs
            WHERE agent_name = %s
        """
        params = [agent_name]
        
        if user_id:
            query += " AND user_id = %s"
            params.append(user_id)
        
        if fpcid:
            query += " AND FPCID = %s"
            params.append(fpcid)
        
        if status_filter:
            query += " AND response_status = %s"
            params.append(status_filter)
        
        if start_date:
            query += " AND DATE(request_timestamp) >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND DATE(request_timestamp) <= %s"
            params.append(end_date)
        
        query += " ORDER BY request_timestamp DESC LIMIT %s OFFSET %s"
        params.extend([min(limit, 500), offset])
        
        cursor.execute(query, params)
        logs = cursor.fetchall()
        
        # Get total count
        count_query = "SELECT COUNT(*) as total FROM tblaiagents_api_logs WHERE agent_name = %s"
        count_params = [agent_name]
        
        if user_id:
            count_query += " AND user_id = %s"
            count_params.append(user_id)
        if fpcid:
            count_query += " AND FPCID = %s"
            count_params.append(fpcid)
        if status_filter:
            count_query += " AND response_status = %s"
            count_params.append(status_filter)
        if start_date:
            count_query += " AND DATE(request_timestamp) >= %s"
            count_params.append(start_date)
        if end_date:
            count_query += " AND DATE(request_timestamp) <= %s"
            count_params.append(end_date)
        
        cursor.execute(count_query, count_params)
        total = cursor.fetchone()['total']
        
        # Format datetime fields
        for log in logs:
            for key in ['request_timestamp', 'process_start_time', 'process_end_time',
                       'ocr_start_time', 'ocr_end_time', 'classification_start_time',
                       'classification_end_time', 'extraction_start_time', 'extraction_end_time',
                       'validation_start_time', 'validation_end_time', 'created_at']:
                if log.get(key):
                    log[key] = log[key].isoformat()
        
        cursor.close()
        
        return {
            "success": True,
            "agent_name": agent_name,
            "logs": logs,
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "returned": len(logs)
            },
            "filters_applied": {
                "user_id": user_id,
                "fpcid": fpcid,
                "status_filter": status_filter,
                "start_date": start_date,
                "end_date": end_date
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch agent logs: {str(e)}"
        )
    finally:
        if connection and connection.is_connected():
            connection.close()


@app.get("/api/logs/agent/{agent_name}/summary", status_code=status.HTTP_200_OK)
async def get_agent_logs_summary(
    agent_name: str,
    days: int = 7
):
    """
    Get summary statistics of API logs for a specific agent.
    
    **Path Parameters:**
    - **agent_name**: Name of the agent
    
    **Query Parameters:**
    - **days**: Number of days to analyze (default: 7)
    
    **Returns:**
    - Total API hits
    - Unique users count
    - Average processing time per stage
    - Success/failure rates
    - Peak usage hours
    - Top users by hit count
    """
    connection = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Ensure table exists
        ensure_tables_exist(connection)
        
        # Overall stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_hits,
                COUNT(DISTINCT user_id) as unique_users,
                COUNT(DISTINCT FPCID) as unique_fpcids,
                AVG(total_processing_time_ms) as avg_total_time_ms,
                AVG(ocr_duration_ms) as avg_ocr_time_ms,
                AVG(classification_duration_ms) as avg_classification_time_ms,
                AVG(extraction_duration_ms) as avg_extraction_time_ms,
                AVG(validation_duration_ms) as avg_validation_time_ms,
                SUM(response_status = 'pass') as pass_count,
                SUM(response_status = 'fail') as fail_count,
                SUM(response_status = 'error') as error_count,
                SUM(llm_total_tokens) as total_llm_tokens,
                SUM(llm_calls_count) as total_llm_calls,
                MIN(request_timestamp) as first_request,
                MAX(request_timestamp) as last_request
            FROM tblaiagents_api_logs
            WHERE agent_name = %s AND request_timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
        """, (agent_name, days))
        overall = cursor.fetchone()
        
        # Hourly distribution
        cursor.execute("""
            SELECT 
                HOUR(request_timestamp) as hour,
                COUNT(*) as hits
            FROM tblaiagents_api_logs
            WHERE agent_name = %s AND request_timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
            GROUP BY HOUR(request_timestamp)
            ORDER BY hour
        """, (agent_name, days))
        hourly_distribution = cursor.fetchall()
        
        # Top users
        cursor.execute("""
            SELECT 
                user_id,
                COUNT(*) as hit_count,
                AVG(total_processing_time_ms) as avg_processing_time_ms,
                SUM(response_status = 'pass') as pass_count,
                SUM(response_status = 'fail') as fail_count,
                MAX(request_timestamp) as last_request
            FROM tblaiagents_api_logs
            WHERE agent_name = %s AND request_timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
            AND user_id IS NOT NULL
            GROUP BY user_id
            ORDER BY hit_count DESC
            LIMIT 10
        """, (agent_name, days))
        top_users = cursor.fetchall()
        
        # Daily trend
        cursor.execute("""
            SELECT 
                DATE(request_timestamp) as date,
                COUNT(*) as hits,
                SUM(response_status = 'pass') as passes,
                SUM(response_status = 'fail') as fails,
                AVG(total_processing_time_ms) as avg_time_ms
            FROM tblaiagents_api_logs
            WHERE agent_name = %s AND request_timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
            GROUP BY DATE(request_timestamp)
            ORDER BY date DESC
        """, (agent_name, days))
        daily_trend = cursor.fetchall()
        
        # Format dates
        for user in top_users:
            if user.get('last_request'):
                user['last_request'] = user['last_request'].isoformat()
            user['avg_processing_time_ms'] = round(float(user['avg_processing_time_ms'] or 0), 2)
            user['pass_count'] = int(user['pass_count'] or 0)
            user['fail_count'] = int(user['fail_count'] or 0)
        
        for day in daily_trend:
            if day.get('date'):
                day['date'] = day['date'].isoformat()
            day['avg_time_ms'] = round(float(day['avg_time_ms'] or 0), 2)
            day['passes'] = int(day['passes'] or 0)
            day['fails'] = int(day['fails'] or 0)
        
        first_request = None
        last_request = None
        if overall.get('first_request'):
            first_request = overall['first_request'].isoformat()
        if overall.get('last_request'):
            last_request = overall['last_request'].isoformat()
        
        # Calculate success rate
        total = int(overall['total_hits'] or 0)
        passes = int(overall['pass_count'] or 0)
        success_rate = round((passes / total * 100), 2) if total > 0 else 0
        
        cursor.close()
        
        return {
            "success": True,
            "agent_name": agent_name,
            "period_days": days,
            "summary": {
                "total_hits": total,
                "unique_users": int(overall['unique_users'] or 0),
                "unique_fpcids": int(overall['unique_fpcids'] or 0),
                "success_rate_percent": success_rate,
                "pass_count": passes,
                "fail_count": int(overall['fail_count'] or 0),
                "error_count": int(overall['error_count'] or 0),
                "first_request": first_request,
                "last_request": last_request
            },
            "average_timings_ms": {
                "total_processing": round(float(overall['avg_total_time_ms'] or 0), 2),
                "ocr": round(float(overall['avg_ocr_time_ms'] or 0), 2),
                "classification": round(float(overall['avg_classification_time_ms'] or 0), 2),
                "extraction": round(float(overall['avg_extraction_time_ms'] or 0), 2),
                "validation": round(float(overall['avg_validation_time_ms'] or 0), 2)
            },
            "llm_usage": {
                "total_calls": int(overall['total_llm_calls'] or 0),
                "total_tokens": int(overall['total_llm_tokens'] or 0)
            },
            "hourly_distribution": hourly_distribution,
            "top_users": top_users,
            "daily_trend": daily_trend
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch agent logs summary: {str(e)}"
        )
    finally:
        if connection and connection.is_connected():
            connection.close()


@app.get("/api/logs/user/{user_id}", status_code=status.HTTP_200_OK)
async def get_user_api_logs(
    user_id: str,
    agent_name: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """
    Get all API logs for a specific user across all agents.
    
    **Path Parameters:**
    - **user_id**: User ID to get logs for
    
    **Query Parameters:**
    - **agent_name**: Filter by specific agent (optional)
    - **limit**: Maximum records to return (default: 50)
    - **offset**: Pagination offset (default: 0)
    
    **Returns:**
    - All API calls made by this user with timing details
    """
    connection = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Ensure table exists
        ensure_tables_exist(connection)
        
        query = """
            SELECT 
                log_id,
                agent_name,
                endpoint,
                FPCID,
                LMRId,
                doc_id,
                request_timestamp,
                total_processing_time_ms,
                response_status,
                validation_score,
                document_type,
                error_message
            FROM tblaiagents_api_logs
            WHERE user_id = %s
        """
        params = [user_id]
        
        if agent_name:
            query += " AND agent_name = %s"
            params.append(agent_name)
        
        query += " ORDER BY request_timestamp DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        logs = cursor.fetchall()
        
        # Get summary stats for this user
        cursor.execute("""
            SELECT 
                COUNT(*) as total_requests,
                COUNT(DISTINCT agent_name) as agents_used,
                AVG(total_processing_time_ms) as avg_processing_time_ms,
                SUM(response_status = 'pass') as pass_count,
                SUM(response_status = 'fail') as fail_count,
                MIN(request_timestamp) as first_request,
                MAX(request_timestamp) as last_request
            FROM tblaiagents_api_logs
            WHERE user_id = %s
        """, (user_id,))
        user_stats = cursor.fetchone()
        
        # Format datetime fields
        for log in logs:
            if log.get('request_timestamp'):
                log['request_timestamp'] = log['request_timestamp'].isoformat()
        
        first_request = None
        last_request = None
        if user_stats.get('first_request'):
            first_request = user_stats['first_request'].isoformat()
        if user_stats.get('last_request'):
            last_request = user_stats['last_request'].isoformat()
        
        cursor.close()
        
        return {
            "success": True,
            "user_id": user_id,
            "user_stats": {
                "total_requests": int(user_stats['total_requests'] or 0),
                "agents_used": int(user_stats['agents_used'] or 0),
                "avg_processing_time_ms": round(float(user_stats['avg_processing_time_ms'] or 0), 2),
                "pass_count": int(user_stats['pass_count'] or 0),
                "fail_count": int(user_stats['fail_count'] or 0),
                "first_request": first_request,
                "last_request": last_request
            },
            "logs": logs,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "returned": len(logs)
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch user logs: {str(e)}"
        )
    finally:
        if connection and connection.is_connected():
            connection.close()


@app.get("/api/logs/{log_id}", status_code=status.HTTP_200_OK)
async def get_single_log_detail(log_id: str):
    """
    Get complete details for a single API log entry.
    
    **Path Parameters:**
    - **log_id**: The unique log ID (UUID)
    
    **Returns:**
    - Complete log entry with all timing details, request/response data, and error traces
    """
    connection = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Ensure table exists
        ensure_tables_exist(connection)
        
        cursor.execute("""
            SELECT * FROM tblaiagents_api_logs WHERE log_id = %s
        """, (log_id,))
        
        log = cursor.fetchone()
        
        if not log:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Log entry with log_id='{log_id}' not found"
            )
        
        # Format all datetime fields
        datetime_fields = [
            'request_timestamp', 'process_start_time', 'process_end_time',
            'ocr_start_time', 'ocr_end_time', 'classification_start_time',
            'classification_end_time', 'extraction_start_time', 'extraction_end_time',
            'validation_start_time', 'validation_end_time', 'created_at'
        ]
        
        for field in datetime_fields:
            if log.get(field):
                log[field] = log[field].isoformat()
        
        cursor.close()
        
        return {
            "success": True,
            "log": log
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch log details: {str(e)}"
        )
    finally:
        if connection and connection.is_connected():
            connection.close()


@app.get("/api/logs/all/agents", status_code=status.HTTP_200_OK)
async def get_all_agents_logs_overview():
    """
    Get an overview of logs for all agents.
    
    **Returns:**
    - List of all agents with their hit counts, success rates, and average processing times
    """
    connection = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Ensure table exists
        ensure_tables_exist(connection)
        
        cursor.execute("""
            SELECT 
                agent_name,
                COUNT(*) as total_hits,
                COUNT(DISTINCT user_id) as unique_users,
                AVG(total_processing_time_ms) as avg_processing_time_ms,
                SUM(response_status = 'pass') as pass_count,
                SUM(response_status = 'fail') as fail_count,
                SUM(response_status = 'error') as error_count,
                MIN(request_timestamp) as first_request,
                MAX(request_timestamp) as last_request
            FROM tblaiagents_api_logs
            GROUP BY agent_name
            ORDER BY total_hits DESC
        """)
        
        agents = cursor.fetchall()
        
        # Calculate success rates and format dates
        for agent in agents:
            total = int(agent['total_hits'] or 0)
            passes = int(agent['pass_count'] or 0)
            agent['success_rate_percent'] = round((passes / total * 100), 2) if total > 0 else 0
            
            if agent.get('first_request'):
                agent['first_request'] = agent['first_request'].isoformat()
            if agent.get('last_request'):
                agent['last_request'] = agent['last_request'].isoformat()
            
            agent['avg_processing_time_ms'] = round(float(agent['avg_processing_time_ms'] or 0), 2)
            agent['pass_count'] = int(agent['pass_count'] or 0)
            agent['fail_count'] = int(agent['fail_count'] or 0)
            agent['error_count'] = int(agent['error_count'] or 0)
            agent['unique_users'] = int(agent['unique_users'] or 0)
        
        cursor.close()
        
        return {
            "success": True,
            "total_agents": len(agents),
            "agents": agents
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch agents overview: {str(e)}"
        )
    finally:
        if connection and connection.is_connected():
            connection.close()


@app.delete("/api/logs/agent/{agent_name}", status_code=status.HTTP_200_OK)
async def delete_agent_logs(
    agent_name: str,
    older_than_days: Optional[int] = None
):
    """
    Delete API logs for a specific agent.
    
    **Path Parameters:**
    - **agent_name**: Name of the agent
    
    **Query Parameters:**
    - **older_than_days**: Only delete logs older than this many days (optional)
                          If not provided, deletes ALL logs for the agent
    
    **Returns:**
    - Number of logs deleted
    """
    connection = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor()
        
        if older_than_days:
            cursor.execute("""
                DELETE FROM tblaiagents_api_logs 
                WHERE agent_name = %s AND request_timestamp < DATE_SUB(NOW(), INTERVAL %s DAY)
            """, (agent_name, older_than_days))
        else:
            cursor.execute("""
                DELETE FROM tblaiagents_api_logs WHERE agent_name = %s
            """, (agent_name,))
        
        rows_deleted = cursor.rowcount
        connection.commit()
        cursor.close()
        
        return {
            "success": True,
            "message": f"Deleted {rows_deleted} log entries for agent '{agent_name}'",
            "agent_name": agent_name,
            "rows_deleted": rows_deleted,
            "older_than_days": older_than_days
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete agent logs: {str(e)}"
        )
    finally:
        if connection and connection.is_connected():
            connection.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)