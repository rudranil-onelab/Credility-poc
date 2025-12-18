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
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import mysql.connector
from mysql.connector import Error

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Nodes.tools.bedrock_client import get_bedrock_client, strip_json_code_fences



def create_agentv2(
        self,
        agent_name: str,
        display_name: str,
        prompt: str,
        mode: str = "ocr+llm",
        tamper_check: bool = False,
        creator_id: str = None,
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
        
        Returns:
            Dict with agent_id, agent_name, endpoint, mode, tamper_check
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
            
            # Ensure image_descriptions column exists (add if missing)
            try:
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM information_schema.COLUMNS 
                    WHERE TABLE_SCHEMA = DATABASE() 
                    AND TABLE_NAME = 'tblcustom_agents' 
                    AND COLUMN_NAME = 'image_descriptions'
                """)
                column_exists = cursor.fetchone()[0] > 0
                
                if not column_exists:
                    cursor.execute("""
                        ALTER TABLE tblcustom_agents 
                        ADD COLUMN image_descriptions TEXT DEFAULT NULL
                    """)
                    self.db.commit()
                    print("[DB] Added image_descriptions column to tblcustom_agents table")
            except Exception as e:
                print(f"[DB] Note: Could not add image_descriptions column: {e}")
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
            # Serialize reference_images and image_descriptions to JSON strings
            # reference_images_json = json.dumps(reference_images) if reference_images else None
            # image_descriptions_json = json.dumps(image_descriptions) if image_descriptions else None
            
            # Insert new agent with tamper_check, reference_images, and image_descriptions
            reference_images_json = json.dumps([])  # Default to empty list
            image_descriptions_json = json.dumps([])  # Default to empty list
            cursor.execute("""
                INSERT INTO tblcustom_agents 
                (agent_name, display_name, prompt, endpoint, mode, tamper_check, creator_id, is_active, total_hits, reference_images, image_descriptions)
                VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE, 0, %s, %s)
            """, (agent_name, display_name, prompt, endpoint, mode, tamper_check, creator_id, reference_images_json, image_descriptions_json))
            
            self.db.commit()
            agent_id = cursor.lastrowid
            
            return {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "endpoint": endpoint,
                "mode": mode,
                "tamper_check": tamper_check,
                # "reference_images": reference_images,
                # "image_descriptions": image_descriptions
            }
        finally:
            cursor.close()
    