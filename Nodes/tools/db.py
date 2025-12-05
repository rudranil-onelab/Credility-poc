"""
Simple database helper to insert AI agent results into MySQL-compatible DB.

Environment variables expected (with reasonable defaults for dev):
  DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME

Target table: stage_newskinny.tblaigents
"""

import os
from typing import Optional, Dict, Any

import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
load_dotenv()


def _make_connection():
    """Create a MySQL connection using environment variables."""
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME", "stage_newskinny"),
        autocommit=True,
    )


def insert_tblaigents(row: Dict[str, Any]) -> None:
    """
    Insert a row into stage_newskinny.tblaiagents_config and tblaiagents_result.
    
    This function is DEPRECATED - records should be created via API first.
    This is kept for backward compatibility only.

    Expected keys in `row` (any missing will be stored as NULL):
      id, FPCID, LMRId, document_name, agent_name, tool,
      file_s3_location, date, document_status, uploadedat,
      metadata_s3_path, verified_result_s3_path, created_at
    """
    print("[DB] WARNING: insert_tblaigents is deprecated. Records should be created via API.")
    print("[DB] This function is kept for backward compatibility only.")
    
    # Note: In the new schema, config and results are separate tables
    # This function would need to be refactored to insert into both tables
    # For now, we'll just log a warning



def fetch_agent_context(FPCID: str, checklistId: str, document_name: str = None) -> Dict[str, Any]:
    """
    Fetch minimal agent context for a borrower/loan using FPCID + checklistId.
    If document_name is provided, it will try to match the specific document first,
    then fall back to any document for the FPCID/checklistId combination.

    Args:
        FPCID: FPC ID
        checklistId: Checklist ID (unique per document type for a borrower)
        document_name: Document name (optional)

    Returns a dict that may contain: document_name, agent_name, tool, id (airecordid).
    If no row is found or an error occurs, returns an empty dict.
    """
    conn = None
    cursor = None
    try:
        conn = _make_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Query from tblaiagents_config table (config data)
        # First try to find exact match with document_name if provided
        if document_name:
            sql = (
                "SELECT id, document_name, agent_name, tool FROM stage_newskinny.tblaiagents_config "
                "WHERE FPCID = %s AND checklistId = %s AND document_name = %s LIMIT 1"
            )
            cursor.execute(sql, [FPCID, checklistId, document_name])
            row = cursor.fetchone()
            if row:
                return row
        
        # Fallback: find any document for this FPCID/checklistId combination
        sql = (
            "SELECT id, document_name, agent_name, tool FROM stage_newskinny.tblaiagents_config "
            "WHERE FPCID = %s AND checklistId = %s LIMIT 1"
        )
        cursor.execute(sql, [FPCID, checklistId])
        row = cursor.fetchone()
        return row or {}
    except Error as e:
        print(f"[DB] fetch_agent_context failed: {e}")
        return {}
    finally:
        try:
            if cursor:
                cursor.close()
        except Exception:
            pass
        try:
            if conn and conn.is_connected():
                conn.close()
        except Exception:
            pass


def update_doc_id_if_not_set(FPCID: str, checklistId: str, document_name: str, doc_id: str, user_id: str = None) -> bool:
    """
    DEPRECATED: With new INSERT-based approach, doc_id is set during INSERT, not updated.
    
    This function is kept for backward compatibility but does nothing.
    doc_id is now passed directly to insert_result_record() during document processing.
    
    Args:
        FPCID: FPC ID
        checklistId: Checklist ID (unique per document type for a borrower)
        document_name: Document name
        doc_id: Document ID from SQS message
        user_id: User ID from SQS message (optional)
    
    Returns:
        True (always, for compatibility)
    """
    print(f"[DB] update_doc_id_if_not_set called - DEPRECATED (doc_id will be set during INSERT)")
    print(f"[DB] doc_id='{doc_id}' will be included in new result record")
    return True


def upsert_result_record(FPCID: str, airecordid: str, updates: Dict[str, Any], document_name: str | None = None, LMRId: str | None = None, doc_id: str | None = None, file_s3_location: str | None = None) -> None:
    """
    INSERT or UPDATE a result record in tblaiagents_result.
    
    RE-UPLOAD DETECTION LOGIC:
    - Checks if record exists by: FPCID + LMRId + checklistId + doc_id
    - If found → UPDATE (re-upload, replaces previous result)
    - If not found → INSERT (new upload)
    
    This means:
    - Same FPCID + LMRId + checklistId + doc_id = RE-UPLOAD → Updates existing record
    - Different doc_id = NEW DOCUMENT → Creates new record
    - File name doesn't matter (Adhar.png → dl_480.png both update same record)
    - Document name can change (user corrects document type)
    - Status can change (fail → pass on re-upload)
    
    Business Rule: One result per FPCID + LMRId + checklistId + doc_id combination

    Args:
        FPCID: FPC ID
        airecordid: AI record ID (config_id from tblaiagents_config table) - can be None for early failures
        updates: Dictionary of fields to insert/update (must include checklistId)
        document_name: Document name (can change on re-upload)
        LMRId: LMR ID
        doc_id: Document ID (REQUIRED for uniqueness)
        file_s3_location: S3 file location (can change on re-upload)
    """
    if not FPCID:
        print(f"[DB] upsert skipped: missing FPCID - FPCID={FPCID}")
        return
    
    # Allow airecordid to be None for early failures (OCR extraction failures, etc.)
    if not airecordid:
        print(f"[DB] WARNING: airecordid is None - this may be an early failure (OCR/classification)")
        print(f"[DB] Cannot insert result without config_id (airecordid) - skipping database insert")
        print(f"[DB] FPCID={FPCID}, LMRId={LMRId}, checklistId={updates.get('checklistId')}, doc_id={doc_id}")
        print(f"[DB] Please create a config record first using the API: POST /create-agent-record")
        return None
    
    conn = None
    cursor = None
    try:
        conn = _make_connection()
        cursor = conn.cursor()
        
        # Get file_s3_location and checklistId from updates if not provided
        if not file_s3_location:
            file_s3_location = updates.get("file_s3_location")
        checklistId = updates.get("checklistId")
        
        # Check if record exists for this upload (match by FPCID + LMRId + checklistId + doc_id)
        # This identifies a RE-UPLOAD of the SAME document
        # Different doc_id = NEW document → will create new record
        # For early failures (OCR), checklistId or doc_id might be None - handle gracefully
        existing_record = None
        if checklistId and doc_id:
            check_sql = """
            SELECT id FROM stage_newskinny.tblaiagents_result 
            WHERE FPCID = %s AND LMRId = %s AND checklistId = %s AND doc_id = %s
                LIMIT 1
            """
            cursor.execute(check_sql, (FPCID, LMRId, checklistId, doc_id))
            existing_record = cursor.fetchone()
        else:
            print(f"[DB] Skipping re-upload check (checklistId={checklistId}, doc_id={doc_id} - will INSERT new record)")
        
        if existing_record:
            # UPDATE existing record (RE-UPLOAD DETECTED)
            existing_id = existing_record[0]
            print(f"[DB] ✓ RE-UPLOAD DETECTED: Updating existing record id={existing_id}")
            print(f"[DB] Match: FPCID={FPCID}, LMRId={LMRId}, checklistId={checklistId}, doc_id={doc_id}")
            print(f"[DB] ✓ RESETTING: is_verified=0, Validation_status=NULL")
            print(f"[DB] ✓ document_status will be set by Nodes pipeline (pass or fail)")
            
            # Build UPDATE query dynamically based on provided fields
            # Include file_s3_location and document_name (may change on re-upload)
            update_fields = []
            update_values = []
            
            for field in ["file_s3_location", "document_name", "LMRId", "doc_id", "checklistId", 
                          "document_type", "metadata_s3_path", "verified_result_s3_path", 
                          "cross_validation", "is_verified", "document_status", "doc_verification_result", 
                          "borrower_type", "Validation_status"]:
                if field in updates and updates[field] is not None:
                    update_fields.append(f"{field} = %s")
                    update_values.append(updates[field])
                elif field == "file_s3_location" and file_s3_location:
                    update_fields.append(f"{field} = %s")
                    update_values.append(file_s3_location)
                elif field == "document_name" and document_name:
                    update_fields.append(f"{field} = %s")
                    update_values.append(document_name)
                elif field == "LMRId" and LMRId:
                    update_fields.append(f"{field} = %s")
                    update_values.append(LMRId)
                elif field == "doc_id" and doc_id:
                    update_fields.append(f"{field} = %s")
                    update_values.append(doc_id)
            
            # CRITICAL: On re-upload, reset cross-validation flags
            # This allows the document to be re-processed through Nodes pipeline
            # and then re-validated by cross-validation system
            # NOTE: We do NOT reset document_status or cross_validation here
            # - document_status: Set by Validation Check Node (pass/fail)
            # - cross_validation: Set by Borrower Identification Node (True if borrower/co-borrower match)
            # - Validation_status: Set by Validation Check Node (fail/human_review) or Cross-validation (pass/fail/human_review)
            update_fields.append("is_verified = 0")  # Reset: not yet cross-validated
            
            # Only reset Validation_status if it's NOT being explicitly set in updates
            # This allows Validation Check Node to set Validation_status="fail" for invalid documents
            if "Validation_status" not in updates or updates.get("Validation_status") is None:
                update_fields.append("Validation_status = NULL")  # Clear previous cross-validation result
            
            # Always update updated_at
            update_fields.append("updated_at = NOW()")
            
            if update_fields:
                update_sql = f"""
                UPDATE stage_newskinny.tblaiagents_result 
                SET {', '.join(update_fields)}
                WHERE id = %s
                """
                update_values.append(existing_id)
                
                print(f"[DB] SQL: {update_sql}")
                print(f"[DB] Values: {update_values}")
                
                cursor.execute(update_sql, update_values)
                conn.commit()
                print(f"[DB] ✓ Update completed: Result record id={existing_id} updated with new file")
                return existing_id
        else:
            # INSERT new record (FIRST UPLOAD or NEW doc_id)
            print(f"[DB] ✓ NEW UPLOAD: Inserting new record")
            print(f"[DB] FPCID={FPCID}, LMRId={LMRId}, checklistId={checklistId}, doc_id={doc_id}")
            
            insert_sql = """
            INSERT INTO stage_newskinny.tblaiagents_result (
                config_id, FPCID, LMRId, doc_id, checklistId, document_name, document_type,
                file_s3_location, metadata_s3_path, verified_result_s3_path, 
                cross_validation, is_verified, document_status, 
                doc_verification_result, borrower_type, Validation_status
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """
            
            values = [
                airecordid,  # config_id
                FPCID,
                LMRId,
                doc_id,
                updates.get("checklistId"),
                document_name,
                updates.get("document_type"),
                file_s3_location,
                updates.get("metadata_s3_path"),
                updates.get("verified_result_s3_path"),
                updates.get("cross_validation", False),
                updates.get("is_verified", False),
                updates.get("document_status", "pending"),
                updates.get("doc_verification_result"),
                updates.get("borrower_type"),  # Add borrower_type
                updates.get("Validation_status"),  # Don't set default - leave NULL until cross-validation completes
            ]
            
            print(f"[DB] SQL: {insert_sql}")
            print(f"[DB] Values: {values}")
            
            cursor.execute(insert_sql, values)
            result_id = cursor.lastrowid
            conn.commit()
            
            print(f"[DB] ✓ Insert completed: New result record created with id={result_id}")
            return result_id
            
    except Error as e:
        print(f"[DB] Upsert failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        try:
            if cursor:
                cursor.close()
        except Exception:
            pass
        try:
            if conn and conn.is_connected():
                conn.close()
        except Exception:
            pass


def update_result_record_fields(
    FPCID: str,
    LMRId: str,
    checklistId: str,
    doc_id: str,
    updates: Dict[str, Any]
) -> None:
    """
    Update specific fields in an existing result record WITHOUT resetting cross-validation flags.
    
    This function is used by Borrower Identification Node to update borrower_type and cross_validation
    without triggering the re-upload reset logic.
    
    Args:
        FPCID: FPC ID
        LMRId: LMR ID
        checklistId: Checklist ID
        doc_id: Document ID
        updates: Dictionary of fields to update (borrower_type, cross_validation, document_status, etc.)
    """
    print(f"[DB] update_result_record_fields called with: FPCID={FPCID} (type: {type(FPCID).__name__}), LMRId={LMRId} (type: {type(LMRId).__name__}), checklistId={checklistId} (type: {type(checklistId).__name__}), doc_id={doc_id} (type: {type(doc_id).__name__})")
    print(f"[DB] Updates dict: {list(updates.keys())}")
    
    # Validate and convert all values to strings to avoid type conversion issues
    # Check for None or empty values first
    if FPCID is None or LMRId is None or checklistId is None or doc_id is None:
        print(f"[DB] update_result_record_fields skipped: missing required fields (None values)")
        print(f"[DB] FPCID={FPCID} (type: {type(FPCID)}), LMRId={LMRId} (type: {type(LMRId)})")
        print(f"[DB] checklistId={checklistId} (type: {type(checklistId)}), doc_id={doc_id} (type: {type(doc_id)})")
        return
    
    # Convert to strings, handling various input types
    try:
        FPCID = str(FPCID).strip()
        LMRId = str(LMRId).strip()
        checklistId = str(checklistId).strip()
        doc_id = str(doc_id).strip()
    except Exception as e:
        print(f"[DB] update_result_record_fields skipped: error converting values to strings: {e}")
        return
    
    # Final validation - ensure no empty strings or "None" strings
    if not all([FPCID, LMRId, checklistId, doc_id]):
        print(f"[DB] update_result_record_fields skipped: empty string values after conversion")
        print(f"[DB] FPCID='{FPCID}', LMRId='{LMRId}', checklistId='{checklistId}', doc_id='{doc_id}'")
        return
    
    # Check for "None" string (from str(None))
    if FPCID.lower() == "none" or LMRId.lower() == "none" or checklistId.lower() == "none" or doc_id.lower() == "none":
        print(f"[DB] update_result_record_fields skipped: 'None' string detected (likely from str(None))")
        print(f"[DB] FPCID='{FPCID}', LMRId='{LMRId}', checklistId='{checklistId}', doc_id='{doc_id}'")
        return
    
    conn = None
    cursor = None
    try:
        conn = _make_connection()
        cursor = conn.cursor()
        
        # Build UPDATE query dynamically based on provided fields
        update_fields = []
        update_values = []
        
        # Allow updating these fields without reset
        allowed_fields = [
            "borrower_type", "cross_validation", "document_status", 
            "Validation_status", "doc_verification_result"
        ]
        
        for field in allowed_fields:
            if field in updates and updates[field] is not None:
                update_fields.append(f"{field} = %s")
                value = updates[field]
                # Convert boolean to int for MySQL (TINYINT columns)
                if isinstance(value, bool):
                    value = 1 if value else 0
                # Convert None to NULL (though this shouldn't happen due to the check above)
                elif value is None:
                    value = None  # MySQL will handle this as NULL
                update_values.append(value)
        
        if not update_fields:
            print(f"[DB] No fields to update")
            return
        
        # Always update updated_at
        update_fields.append("updated_at = NOW()")
        
        # First, check if record exists
        check_sql = """
        SELECT id, FPCID, LMRId, checklistId, doc_id, borrower_type, cross_validation, document_status 
        FROM stage_newskinny.tblaiagents_result 
        WHERE FPCID = %s AND LMRId = %s AND checklistId = %s AND doc_id = %s
        LIMIT 1
        """
        
        # Build WHERE clause
        update_sql = f"""
        UPDATE stage_newskinny.tblaiagents_result 
        SET {', '.join(update_fields)}
        WHERE FPCID = %s AND LMRId = %s AND checklistId = %s AND doc_id = %s
        """
        # Convert WHERE clause values to appropriate types
        # Try to convert to int if possible (for integer columns), otherwise keep as string
        def safe_int_convert(value):
            """Try to convert to int, fall back to string if conversion fails."""
            # First check for None
            if value is None:
                raise ValueError(f"Cannot convert None to int/string for WHERE clause. Value is None.")
            
            # Check for "None" string (from str(None))
            if isinstance(value, str):
                value_stripped = value.strip()
                if value_stripped.lower() == "none" or value_stripped == "":
                    raise ValueError(f"Cannot convert '{value}' to int for WHERE clause")
            
            try:
                # Try converting to int
                int_val = int(value)
                return int_val
            except (ValueError, TypeError) as e:
                # If conversion fails, return as string (but log a warning)
                print(f"[DB] WARNING: Could not convert '{value}' (type: {type(value).__name__}) to int, using as string")
                return str(value)
        
        try:
            where_values = [
                safe_int_convert(FPCID),
                safe_int_convert(LMRId),
                safe_int_convert(checklistId),
                safe_int_convert(doc_id)
            ]
        except ValueError as ve:
            print(f"[DB] ERROR: Failed to convert WHERE clause values: {ve}")
            print(f"[DB] FPCID={FPCID} (type: {type(FPCID)}), LMRId={LMRId} (type: {type(LMRId)})")
            print(f"[DB] checklistId={checklistId} (type: {type(checklistId)}), doc_id={doc_id} (type: {type(doc_id)})")
            raise
        
        # Check if record exists before updating
        try:
             cursor.execute(check_sql, where_values)
             existing_record = cursor.fetchone()
             if existing_record:
                 current_document_status = existing_record[7]  # document_status is at index 7
                 print(f"[DB] ✓ Record found: id={existing_record[0]}, current borrower_type={existing_record[5]}, cross_validation={existing_record[6]}, document_status={current_document_status}")
                 
                 # CRITICAL: Only update borrower_type if document_status is "pass"
                 # Failed documents should not have borrower_type or cross_validation set
                 if current_document_status != "pass":
                     print(f"[DB] ⚠️  SKIP: Document status is '{current_document_status}' (not 'pass') - not updating borrower_type or cross_validation")
                     print(f"[DB] ⚠️  Only documents with document_status='pass' can have borrower_type and cross_validation set")
                     return  # Don't update if document failed validation
             else:
                 print(f"[DB] ⚠️  WARNING: No record found matching WHERE clause!")
                 print(f"[DB] Searching for: FPCID={FPCID}, LMRId={LMRId}, checklistId={checklistId}, doc_id={doc_id}")
                 # Try to find similar records
                 search_sql = """
                 SELECT id, FPCID, LMRId, checklistId, doc_id 
                 FROM stage_newskinny.tblaiagents_result 
                 WHERE FPCID = %s AND LMRId = %s
                 LIMIT 5
                 """
                 cursor.execute(search_sql, [where_values[0], where_values[1]])
                 similar_records = cursor.fetchall()
                 if similar_records:
                     print(f"[DB] Found {len(similar_records)} similar record(s) with same FPCID and LMRId:")
                     for rec in similar_records:
                         print(f"[DB]   - id={rec[0]}, FPCID={rec[1]}, LMRId={rec[2]}, checklistId={rec[3]}, doc_id={rec[4]}")
                 else:
                     print(f"[DB] No similar records found. Record may not exist yet.")
                 return  # Don't update if record doesn't exist
        except Exception as check_error:
            print(f"[DB] WARNING: Error checking for existing record: {check_error}")
            # Continue with update anyway
        
        update_values.extend(where_values)
        
        print(f"[DB] Updating result record: FPCID={FPCID}, LMRId={LMRId}, checklistId={checklistId}, doc_id={doc_id}")
        print(f"[DB] Fields: {list(updates.keys())}")
        print(f"[DB] Update values: {update_values}")
        print(f"[DB] Update values types: {[type(v).__name__ for v in update_values]}")
        print(f"[DB] SQL: {update_sql}")
        
        # Final validation - ensure no None values in update_values
        none_indices = [i for i, v in enumerate(update_values) if v is None]
        if none_indices:
            print(f"[DB] ERROR: Found None values in update_values at indices: {none_indices}")
            print(f"[DB] Update values: {update_values}")
            raise ValueError(f"Cannot execute SQL with None values in parameters at indices: {none_indices}")
        
        try:
            cursor.execute(update_sql, update_values)
        except Exception as sql_error:
            print(f"[DB] SQL execution error: {sql_error}")
            print(f"[DB] Error type: {type(sql_error)}")
            print(f"[DB] Error args: {sql_error.args if hasattr(sql_error, 'args') else 'N/A'}")
            print(f"[DB] Update values: {update_values}")
            print(f"[DB] Update values types: {[type(v).__name__ for v in update_values]}")
            import traceback
            traceback.print_exc()
            raise
        rows_affected = cursor.rowcount
        conn.commit()
        
        if rows_affected > 0:
            print(f"[DB] ✓ Updated {rows_affected} record(s)")
        else:
            print(f"[DB] ⚠️  No records updated - document may not exist")
        
    except Error as e:
        print(f"[DB] Update failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if cursor:
                cursor.close()
        except Exception:
            pass
        try:
            if conn and conn.is_connected():
                conn.close()
        except Exception:
            pass


def update_tblaigents_by_keys(FPCID: str, airecordid: str, updates: Dict[str, Any], document_name: str | None = None, LMRId: str | None = None) -> None:
    """
    DEPRECATED: Use upsert_result_record() for initial uploads or update_result_record_fields() for updates.
    
    This function is kept for backward compatibility.
    
    Update only selected nullable fields for an existing row identified by FPCID + airecordid + document_name.

    Args:
        FPCID: FPC ID
        airecordid: AI record ID (id/config_id from tblaiagents_config table)
        updates: Dictionary of fields to update
        document_name: Document name
        LMRId: LMR ID (optional, will be updated if provided)
    """
    # Check if this is a Borrower Identification update (has borrower_type and cross_validation)
    if "borrower_type" in updates and "cross_validation" in updates and updates.get("checklistId") and updates.get("doc_id"):
        # Use the new function that doesn't reset flags
        print("[DB] Using update_result_record_fields for Borrower Identification update")
        
        # Ensure all values are strings and not None - validate first
        checklist_id = updates.get("checklistId")
        doc_id_val = updates.get("doc_id")
        lmrid_val = LMRId or updates.get("LMRId")
        
        # Validate all required fields are present
        if not all([FPCID, lmrid_val, checklist_id, doc_id_val]):
            print(f"[DB] ERROR: Missing required fields for update_result_record_fields")
            print(f"[DB] FPCID={FPCID}, LMRId={lmrid_val}, checklistId={checklist_id}, doc_id={doc_id_val}")
            print(f"[DB] Updates dict keys: {list(updates.keys())}")
            return
        
        # Convert to strings - all values should be non-None at this point
        try:
            fpcid_str = str(FPCID)
            lmrid_str = str(lmrid_val)
            checklist_str = str(checklist_id)
            doc_id_str = str(doc_id_val)
        except Exception as e:
            print(f"[DB] ERROR: Failed to convert values to strings: {e}")
            print(f"[DB] FPCID={FPCID} (type: {type(FPCID)}), LMRId={lmrid_val} (type: {type(lmrid_val)})")
            print(f"[DB] checklistId={checklist_id} (type: {type(checklist_id)}), doc_id={doc_id_val} (type: {type(doc_id_val)})")
            return
        
        return update_result_record_fields(
            FPCID=fpcid_str,
            LMRId=lmrid_str,
            checklistId=checklist_str,
            doc_id=doc_id_str,
            updates=updates
        )
    else:
        # For Nodes pipeline: UPSERT (update existing or insert new)
        print("[DB] WARNING: update_tblaigents_by_keys called - using UPSERT for document upload")
        return upsert_result_record(FPCID, airecordid, updates, document_name, LMRId, updates.get("doc_id"), updates.get("file_s3_location"))
