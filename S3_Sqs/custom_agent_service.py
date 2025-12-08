"""
Custom Agent Service - Core business logic for dynamic API agents.

This service handles:
- Creating custom validation agents with user-defined prompts
- Validating documents against custom rules
- Recording API usage and analytics
"""

import json
import time
import re
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from openai import OpenAI
import mysql.connector
from mysql.connector import Error


class CustomAgentService:
    """Service for managing custom validation agents."""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    def create_agent(
        self,
        agent_name: str,
        display_name: str,
        prompt: str,
        mode: str = "ocr+llm",
        tamper_check: bool = False,
        creator_id: str = None,
        reference_images: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new custom validation agent.
        
        Args:
            agent_name: Unique identifier for the agent (used in URL)
            display_name: Human-readable name
            prompt: Natural language validation rules
            mode: Processing mode ('ocr+llm' or 'llm')
            tamper_check: Enable tampering detection (default: False)
            creator_id: User who created this agent
            reference_images: List of S3 URLs for reference images (up to 5)
        
        Returns:
            Dict with agent_id, agent_name, endpoint, mode, tamper_check, reference_images
        """
        # Validate agent_name format (lowercase, numbers, underscores only)
        if not re.match(r'^[a-z0-9_]+$', agent_name):
            raise ValueError("Agent name must contain only lowercase letters, numbers, and underscores")
        
        if len(agent_name) < 3:
            raise ValueError("Agent name must be at least 3 characters long")
        
        if len(agent_name) > 100:
            raise ValueError("Agent name must be at most 100 characters long")
        
        # Validate mode
        if mode not in ['ocr+llm', 'llm']:
            raise ValueError("Mode must be 'ocr+llm' or 'llm'")
        
        cursor = self.db.cursor()
        
        try:
            # Ensure tamper_check column exists (add if missing)
            try:
                # Check if column exists first (MySQL compatible)
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM information_schema.COLUMNS 
                    WHERE TABLE_SCHEMA = DATABASE() 
                    AND TABLE_NAME = 'tblcustom_agents' 
                    AND COLUMN_NAME = 'tamper_check'
                """)
                column_exists = cursor.fetchone()[0] > 0
                
                if not column_exists:
                    cursor.execute("""
                        ALTER TABLE tblcustom_agents 
                        ADD COLUMN tamper_check BOOLEAN DEFAULT FALSE
                    """)
                    self.db.commit()
                    print("[DB] Added tamper_check column to tblcustom_agents table")
            except Exception as e:
                print(f"[DB] Note: Could not add tamper_check column: {e}")
                pass  # Column may already exist or other issue
            
            # Ensure reference_images column exists (add if missing)
            try:
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM information_schema.COLUMNS 
                    WHERE TABLE_SCHEMA = DATABASE() 
                    AND TABLE_NAME = 'tblcustom_agents' 
                    AND COLUMN_NAME = 'reference_images'
                """)
                column_exists = cursor.fetchone()[0] > 0
                
                if not column_exists:
                    cursor.execute("""
                        ALTER TABLE tblcustom_agents 
                        ADD COLUMN reference_images TEXT DEFAULT NULL
                    """)
                    self.db.commit()
                    print("[DB] Added reference_images column to tblcustom_agents table")
            except Exception as e:
                print(f"[DB] Note: Could not add reference_images column: {e}")
                pass  # Column may already exist or other issue
            
            # Check if agent already exists
            cursor.execute(
                "SELECT id FROM tblcustom_agents WHERE agent_name = %s",
                (agent_name,)
            )
            if cursor.fetchone():
                raise ValueError(f"Agent '{agent_name}' already exists")
            
            # Generate endpoint
            endpoint = f"/api/agent/{agent_name}/validate"
            
            # Serialize reference_images to JSON string
            reference_images_json = json.dumps(reference_images) if reference_images else None
            
            # Insert new agent with tamper_check and reference_images
            cursor.execute("""
                INSERT INTO tblcustom_agents 
                (agent_name, display_name, prompt, endpoint, mode, tamper_check, creator_id, is_active, total_hits, reference_images)
                VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE, 0, %s)
            """, (agent_name, display_name, prompt, endpoint, mode, tamper_check, creator_id, reference_images_json))
            
            self.db.commit()
            agent_id = cursor.lastrowid
            
            return {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "endpoint": endpoint,
                "mode": mode,
                "tamper_check": tamper_check,
                "reference_images": reference_images
            }
        finally:
            cursor.close()
    
    def get_agent(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get agent configuration by name."""
        cursor = self.db.cursor(dictionary=True)
        try:
            cursor.execute(
                "SELECT * FROM tblcustom_agents WHERE agent_name = %s AND is_active = TRUE",
                (agent_name,)
            )
            agent = cursor.fetchone()
            
            if agent:
                # Format timestamps
                if agent.get('created_at'):
                    agent['created_at'] = agent['created_at'].isoformat()
                if agent.get('updated_at'):
                    agent['updated_at'] = agent['updated_at'].isoformat()
                # Parse reference_images JSON
                if agent.get('reference_images'):
                    try:
                        agent['reference_images'] = json.loads(agent['reference_images'])
                    except (json.JSONDecodeError, TypeError):
                        agent['reference_images'] = None
                else:
                    agent['reference_images'] = None
            
            return agent
        finally:
            cursor.close()
    
    def get_agent_by_id(self, agent_id: int) -> Optional[Dict[str, Any]]:
        """Get agent configuration by ID."""
        cursor = self.db.cursor(dictionary=True)
        try:
            cursor.execute(
                "SELECT * FROM tblcustom_agents WHERE id = %s",
                (agent_id,)
            )
            agent = cursor.fetchone()
            
            if agent:
                if agent.get('created_at'):
                    agent['created_at'] = agent['created_at'].isoformat()
                if agent.get('updated_at'):
                    agent['updated_at'] = agent['updated_at'].isoformat()
                # Parse reference_images JSON
                if agent.get('reference_images'):
                    try:
                        agent['reference_images'] = json.loads(agent['reference_images'])
                    except (json.JSONDecodeError, TypeError):
                        agent['reference_images'] = None
                else:
                    agent['reference_images'] = None
            
            return agent
        finally:
            cursor.close()
    
    def update_agent(
        self,
        agent_name: str,
        display_name: str = None,
        prompt: str = None,
        mode: str = None,
        tamper_check: bool = None,
        is_active: bool = None,
        reference_images: List[str] = None
    ) -> bool:
        """Update an existing agent."""
        cursor = self.db.cursor()
        
        try:
            # Build dynamic update query
            updates = []
            params = []
            
            if display_name is not None:
                updates.append("display_name = %s")
                params.append(display_name)
            
            if prompt is not None:
                updates.append("prompt = %s")
                params.append(prompt)
            
            if mode is not None:
                if mode not in ['ocr+llm', 'llm']:
                    raise ValueError("Mode must be 'ocr+llm' or 'llm'")
                updates.append("mode = %s")
                params.append(mode)
            
            if tamper_check is not None:
                updates.append("tamper_check = %s")
                params.append(tamper_check)
            
            if is_active is not None:
                updates.append("is_active = %s")
                params.append(is_active)
            
            if reference_images is not None:
                updates.append("reference_images = %s")
                params.append(json.dumps(reference_images) if reference_images else None)
            
            if not updates:
                return False
            
            params.append(agent_name)
            
            cursor.execute(f"""
                UPDATE tblcustom_agents 
                SET {', '.join(updates)}
                WHERE agent_name = %s
            """, params)
            
            self.db.commit()
            return cursor.rowcount > 0
        finally:
            cursor.close()
    
    def delete_agent(self, agent_name: str) -> bool:
        """Soft delete (deactivate) an agent."""
        return self.update_agent(agent_name, is_active=False)
    
    def list_agents(
        self,
        creator_id: str = None,
        is_active: bool = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List agents with optional filtering."""
        cursor = self.db.cursor(dictionary=True)
        
        try:
            query = "SELECT * FROM tblcustom_agents WHERE 1=1"
            params = []
            
            if creator_id is not None:
                query += " AND creator_id = %s"
                params.append(creator_id)
            
            if is_active is not None:
                query += " AND is_active = %s"
                params.append(is_active)
            
            query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            agents = cursor.fetchall()
            
            # Format timestamps and parse reference_images
            for agent in agents:
                if agent.get('created_at'):
                    agent['created_at'] = agent['created_at'].isoformat()
                if agent.get('updated_at'):
                    agent['updated_at'] = agent['updated_at'].isoformat()
                # Parse reference_images JSON
                if agent.get('reference_images'):
                    try:
                        agent['reference_images'] = json.loads(agent['reference_images'])
                    except (json.JSONDecodeError, TypeError):
                        agent['reference_images'] = None
                else:
                    agent['reference_images'] = None
            
            return agents
        finally:
            cursor.close()
    
    def validate_document_with_prompt(
        self,
        extracted_fields: Dict[str, Any],
        user_prompt: str,
        document_type: str = None
    ) -> Dict[str, Any]:
        """
        Validate extracted document fields against custom prompt rules.
        Uses LLM to interpret the natural language validation rules.
        
        Args:
            extracted_fields: Extracted data from document
            user_prompt: User's custom validation rules
            document_type: Detected document type
        
        Returns:
            Dict with status, score, reason, checks
        """
        system_prompt = f"""You are a document validation expert.

Your task is to validate extracted document fields against USER-DEFINED rules.
The user's rules are the ONLY criteria for pass/fail - ignore any standard validation rules.

You will receive:
1. Extracted fields from a document (JSON)
2. User's validation rules in natural language

IMPORTANT:
- User's rules have ABSOLUTE PRIORITY
- If user says "pass if age > 54", ONLY check that - nothing else matters unless specified
- Calculate dates/ages accurately using today's date: {datetime.now().strftime("%Y-%m-%d")}
- Be strict about numeric requirements (e.g., "10 digits" means exactly 10)
- If a required field is missing, that check should fail

CRITICAL - FIELD NAME MATCHING:
- Field names in extracted data may vary (e.g., "Father's Name", "fatherName", "father_name", "Father Name" are ALL the same field)
- Use SEMANTIC matching - look for fields with similar meaning, not exact string matches
- "fatherName" or "Father's Name" or "Father Name" = same field (parent name)
- "motherName" or "Mother's Name" or "Mother Name" = same field (parent name)
- "dateOfBirth" or "Date of Birth" or "DOB" or "dob" = same field (birth date)
- "PAN" or "PAN number" or "pan_number" or "panNumber" = same field (PAN number)
- If data EXISTS in ANY variation of the field name, consider that field as PRESENT
- A field with a valid, readable value should PASS the "present and readable" check

Return JSON:
{{
  "status": "pass" or "fail",
  "score": 0-100 (percentage of rules passed),
  "reason": ["human-readable explanation for each check"],
  "checks": [
    {{"rule": "description of rule", "passed": true/false, "value": "actual value found", "message": "explanation"}}
  ]
}}

Return ONLY valid JSON, no markdown or explanations."""

        user_message = f"""
DOCUMENT TYPE: {document_type or "Unknown"}

EXTRACTED FIELDS:
{json.dumps(extracted_fields, indent=2, ensure_ascii=False)}

USER'S VALIDATION RULES:
{user_prompt}

Validate the document against ALL the user's rules above. Return JSON result.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Ensure required fields exist
            result.setdefault("status", "error")
            result.setdefault("score", 0)
            result.setdefault("reason", [])
            result.setdefault("checks", [])
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Custom validation failed: {e}")
            return {
                "status": "error",
                "score": 0,
                "reason": [f"Validation error: {str(e)}"],
                "checks": [],
                "error": str(e)
            }
    
    def record_result(
        self,
        agent_id: int,
        agent_name: str,
        api_endpoint: str,
        user_id: str,
        result: Dict[str, Any],
        request_ip: str = None,
        processing_time_ms: int = None
    ) -> int:
        """
        Record an API result/hit for analytics.
        
        Args:
            agent_id: FK to tblcustom_agents
            agent_name: Agent name (denormalized)
            api_endpoint: Which endpoint was hit
            user_id: Who used the API
            result: JSON result containing status, score, reason, file_name, doc_extracted_json
            request_ip: Client IP address
            processing_time_ms: How long the request took
        
        Returns:
            result_id: ID of the inserted record
        """
        cursor = self.db.cursor()
        
        try:
            # Insert result record
            cursor.execute("""
                INSERT INTO tblcustom_agent_results 
                (agent_id, agent_name, api_endpoint, user_id, result, request_ip, processing_time_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                agent_id,
                agent_name,
                api_endpoint,
                user_id,
                json.dumps(result, ensure_ascii=False),
                request_ip,
                processing_time_ms
            ))
            
            result_id = cursor.lastrowid
            
            # Increment total_hits counter on agent
            cursor.execute("""
                UPDATE tblcustom_agents 
                SET total_hits = total_hits + 1
                WHERE id = %s
            """, (agent_id,))
            
            self.db.commit()
            return result_id
            
        finally:
            cursor.close()
    
    def get_agents_by_creator(
        self,
        creator_id: str,
        is_active: bool = None
    ) -> List[Dict[str, Any]]:
        """Get all agents created by a specific user."""
        cursor = self.db.cursor(dictionary=True)
        
        try:
            query = "SELECT * FROM tblcustom_agents WHERE creator_id = %s"
            params = [creator_id]
            
            if is_active is not None:
                query += " AND is_active = %s"
                params.append(is_active)
            
            query += " ORDER BY created_at DESC"
            
            cursor.execute(query, params)
            agents = cursor.fetchall()
            
            for agent in agents:
                if agent.get('created_at'):
                    agent['created_at'] = agent['created_at'].isoformat()
                if agent.get('updated_at'):
                    agent['updated_at'] = agent['updated_at'].isoformat()
                # Parse reference_images JSON
                if agent.get('reference_images'):
                    try:
                        agent['reference_images'] = json.loads(agent['reference_images'])
                    except (json.JSONDecodeError, TypeError):
                        agent['reference_images'] = None
                else:
                    agent['reference_images'] = None
            
            return agents
        finally:
            cursor.close()
    
    def get_agent_users(
        self,
        agent_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get all users who have used a specific agent with their stats."""
        cursor = self.db.cursor(dictionary=True)
        
        try:
            # Get unique users with their stats
            cursor.execute("""
                SELECT 
                    user_id,
                    COUNT(*) as total_requests,
                    SUM(JSON_UNQUOTE(JSON_EXTRACT(result, '$.status')) = 'pass') as pass_count,
                    SUM(JSON_UNQUOTE(JSON_EXTRACT(result, '$.status')) = 'fail') as fail_count,
                    MIN(created_at) as first_used,
                    MAX(created_at) as last_used
                FROM tblcustom_agent_results
                WHERE agent_name = %s AND user_id IS NOT NULL
                GROUP BY user_id
                ORDER BY total_requests DESC
                LIMIT %s OFFSET %s
            """, (agent_name, limit, offset))
            
            users = cursor.fetchall()
            
            # Get total unique users count
            cursor.execute("""
                SELECT COUNT(DISTINCT user_id) as total
                FROM tblcustom_agent_results
                WHERE agent_name = %s AND user_id IS NOT NULL
            """, (agent_name,))
            total_result = cursor.fetchone()
            total_unique_users = total_result['total'] if total_result else 0
            
            # Format data
            for user in users:
                if user.get('first_used'):
                    user['first_used'] = user['first_used'].isoformat()
                if user.get('last_used'):
                    user['last_used'] = user['last_used'].isoformat()
                user['pass_count'] = int(user['pass_count'] or 0)
                user['fail_count'] = int(user['fail_count'] or 0)
            
            return {
                "total_unique_users": total_unique_users,
                "users": users,
                "has_more": offset + limit < total_unique_users
            }
        finally:
            cursor.close()
    
    def get_agent_stats(self, agent_name: str) -> Dict[str, Any]:
        """Get detailed usage statistics for an agent."""
        cursor = self.db.cursor(dictionary=True)
        
        try:
            # Get overall stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_hits,
                    SUM(JSON_UNQUOTE(JSON_EXTRACT(result, '$.status')) = 'pass') as pass_count,
                    SUM(JSON_UNQUOTE(JSON_EXTRACT(result, '$.status')) = 'fail') as fail_count,
                    SUM(JSON_UNQUOTE(JSON_EXTRACT(result, '$.status')) = 'error') as error_count,
                    AVG(processing_time_ms) as avg_processing_time_ms,
                    COUNT(DISTINCT user_id) as unique_users
                FROM tblcustom_agent_results
                WHERE agent_name = %s
            """, (agent_name,))
            overall = cursor.fetchone()
            
            # Get today's stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as hits,
                    SUM(JSON_UNQUOTE(JSON_EXTRACT(result, '$.status')) = 'pass') as pass_count,
                    SUM(JSON_UNQUOTE(JSON_EXTRACT(result, '$.status')) = 'fail') as fail_count
                FROM tblcustom_agent_results
                WHERE agent_name = %s AND DATE(created_at) = CURDATE()
            """, (agent_name,))
            today = cursor.fetchone()
            
            # Get this week's stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as hits,
                    SUM(JSON_UNQUOTE(JSON_EXTRACT(result, '$.status')) = 'pass') as pass_count,
                    SUM(JSON_UNQUOTE(JSON_EXTRACT(result, '$.status')) = 'fail') as fail_count
                FROM tblcustom_agent_results
                WHERE agent_name = %s AND created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
            """, (agent_name,))
            week = cursor.fetchone()
            
            # Get this month's stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as hits,
                    SUM(JSON_UNQUOTE(JSON_EXTRACT(result, '$.status')) = 'pass') as pass_count,
                    SUM(JSON_UNQUOTE(JSON_EXTRACT(result, '$.status')) = 'fail') as fail_count
                FROM tblcustom_agent_results
                WHERE agent_name = %s AND created_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
            """, (agent_name,))
            month = cursor.fetchone()
            
            # Calculate success rate
            total = int(overall['total_hits'] or 0)
            passes = int(overall['pass_count'] or 0)
            success_rate = round((passes / total * 100), 2) if total > 0 else 0
            
            return {
                "total_hits": total,
                "pass_count": passes,
                "fail_count": int(overall['fail_count'] or 0),
                "error_count": int(overall['error_count'] or 0),
                "success_rate": success_rate,
                "avg_processing_time_ms": round(overall['avg_processing_time_ms'] or 0, 2),
                "unique_users": int(overall['unique_users'] or 0),
                "today": {
                    "hits": int(today['hits'] or 0),
                    "pass": int(today['pass_count'] or 0),
                    "fail": int(today['fail_count'] or 0)
                },
                "this_week": {
                    "hits": int(week['hits'] or 0),
                    "pass": int(week['pass_count'] or 0),
                    "fail": int(week['fail_count'] or 0)
                },
                "this_month": {
                    "hits": int(month['hits'] or 0),
                    "pass": int(month['pass_count'] or 0),
                    "fail": int(month['fail_count'] or 0)
                }
            }
        finally:
            cursor.close()
    
    def get_agent_logs(
        self,
        agent_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get detailed validation logs for an agent.
        
        Returns all validation results including:
        - user_id, status, score, reason
        - file_input (S3 URL of input document)
        - doc_extracted_json, document_type
        - tampering details
        - OCR extraction details
        - processing_time_ms, created_at
        
        Args:
            agent_name: Name of the agent
            limit: Maximum number of logs to return
            offset: Offset for pagination
        
        Returns:
            Dict with logs array and pagination info
        """
        cursor = self.db.cursor(dictionary=True)
        
        try:
            # Get total count
            cursor.execute("""
                SELECT COUNT(*) as total
                FROM tblcustom_agent_results
                WHERE agent_name = %s
            """, (agent_name,))
            total_result = cursor.fetchone()
            total_logs = total_result['total'] if total_result else 0
            
            # Get logs with all details
            cursor.execute("""
                SELECT 
                    id,
                    agent_id,
                    agent_name,
                    api_endpoint,
                    user_id,
                    result,
                    request_ip,
                    processing_time_ms,
                    created_at
                FROM tblcustom_agent_results
                WHERE agent_name = %s
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """, (agent_name, limit, offset))
            
            raw_logs = cursor.fetchall()
            
            # Process logs and expand the result JSON
            logs = []
            for log in raw_logs:
                # Parse the result JSON
                result_data = {}
                if log.get('result'):
                    try:
                        result_data = json.loads(log['result']) if isinstance(log['result'], str) else log['result']
                    except (json.JSONDecodeError, TypeError):
                        result_data = {}
                
                # Build comprehensive log entry
                log_entry = {
                    "id": log.get('id'),
                    "user_id": log.get('user_id'),
                    "success": True,  # If record exists, request was successful
                    "status": result_data.get('status', 'unknown'),
                    "score": result_data.get('score', 0),
                    "reason": result_data.get('reason', []),
                    "file_name": result_data.get('file_name'),
                    "file_input": result_data.get('file_input'),  # S3 URL of input document
                    "doc_extracted_json": result_data.get('doc_extracted_json', {}),
                    "document_type": result_data.get('document_type'),
                    "processing_time_ms": log.get('processing_time_ms'),
                    "agent_name": log.get('agent_name'),
                    # Tampering details
                    "tampering_score": result_data.get('tampering_score'),
                    "tampering_status": result_data.get('tampering_status'),
                    "tampering_details": result_data.get('tampering_details'),
                    # OCR extraction quality
                    "ocr_extraction_status": result_data.get('ocr_extraction_status'),
                    "ocr_extraction_confidence": result_data.get('ocr_extraction_confidence'),
                    "ocr_extraction_reason": result_data.get('ocr_extraction_reason'),
                    # Metadata
                    "request_ip": log.get('request_ip'),
                    "created_at": log['created_at'].isoformat() if log.get('created_at') else None
                }
                
                logs.append(log_entry)
            
            return {
                "total_logs": total_logs,
                "logs": logs,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "has_more": offset + limit < total_logs
                }
            }
        finally:
            cursor.close()
    
    def get_creator_stats(self, creator_id: str) -> Dict[str, Any]:
        """Get aggregated stats for all agents created by a user."""
        cursor = self.db.cursor(dictionary=True)
        
        try:
            # Get agent counts
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_agents,
                    SUM(is_active = TRUE) as active_agents,
                    SUM(is_active = FALSE) as inactive_agents,
                    SUM(total_hits) as total_api_hits
                FROM tblcustom_agents
                WHERE creator_id = %s
            """, (creator_id,))
            summary = cursor.fetchone()
            
            if not summary or summary['total_agents'] == 0:
                return None
            
            # Get per-agent breakdown
            cursor.execute("""
                SELECT 
                    a.agent_name,
                    a.display_name,
                    a.total_hits,
                    a.mode,
                    a.is_active,
                    COUNT(DISTINCT r.user_id) as unique_users
                FROM tblcustom_agents a
                LEFT JOIN tblcustom_agent_results r ON a.agent_name = r.agent_name
                WHERE a.creator_id = %s
                GROUP BY a.id, a.agent_name, a.display_name, a.total_hits, a.mode, a.is_active
                ORDER BY a.total_hits DESC
            """, (creator_id,))
            agents = cursor.fetchall()
            
            for agent in agents:
                agent['unique_users'] = int(agent['unique_users'] or 0)
            
            # Get total unique users across all agents
            cursor.execute("""
                SELECT COUNT(DISTINCT r.user_id) as total_unique_users
                FROM tblcustom_agents a
                JOIN tblcustom_agent_results r ON a.agent_name = r.agent_name
                WHERE a.creator_id = %s
            """, (creator_id,))
            users_result = cursor.fetchone()
            
            # Get recent activity
            cursor.execute("""
                SELECT 
                    SUM(DATE(r.created_at) = CURDATE()) as hits_today,
                    SUM(r.created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)) as hits_this_week,
                    SUM(r.created_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)) as hits_this_month
                FROM tblcustom_agents a
                JOIN tblcustom_agent_results r ON a.agent_name = r.agent_name
                WHERE a.creator_id = %s
            """, (creator_id,))
            activity = cursor.fetchone()
            
            return {
                "summary": {
                    "total_agents_created": int(summary['total_agents'] or 0),
                    "active_agents": int(summary['active_agents'] or 0),
                    "inactive_agents": int(summary['inactive_agents'] or 0),
                    "total_api_hits": int(summary['total_api_hits'] or 0),
                    "total_unique_users": int(users_result['total_unique_users'] or 0) if users_result else 0
                },
                "agents_breakdown": agents,
                "recent_activity": {
                    "hits_today": int(activity['hits_today'] or 0) if activity else 0,
                    "hits_this_week": int(activity['hits_this_week'] or 0) if activity else 0,
                    "hits_this_month": int(activity['hits_this_month'] or 0) if activity else 0
                }
            }
        finally:
            cursor.close()
    
    def get_agent_hit_count(self, agent_name: str) -> Optional[int]:
        """Get simple hit count for an agent."""
        cursor = self.db.cursor(dictionary=True)
        
        try:
            cursor.execute(
                "SELECT total_hits FROM tblcustom_agents WHERE agent_name = %s",
                (agent_name,)
            )
            result = cursor.fetchone()
            return result['total_hits'] if result else None
        finally:
            cursor.close()

