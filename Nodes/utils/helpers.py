"""
Utility functions for the Lendingwise AI pipeline.
"""

import os
import json
import datetime
import socket
from typing import Dict, Any, Optional
from ..config.settings import OUTPUT_DIR


def ensure_state_dict(state: Any) -> Dict[str, Any]:
    """Convert state to dictionary format."""
    try:
        # If it's a Pydantic model
        return state.model_dump()  # type: ignore[attr-defined]
    except Exception:
        pass
    if isinstance(state, dict):
        return dict(state)
    return {}


def save_json_file(data: Dict[str, Any], file_path: str) -> bool:
    """Save data to JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving JSON file {file_path}: {e}")
        return False


def get_filename_without_extension(file_path: str) -> str:
    """Get filename without extension."""
    return os.path.splitext(os.path.basename(file_path))[0]


def normalize_pipeline_result(result: Any, state_class) -> Optional[Any]:
    """Normalize pipeline result to proper state class."""
    if isinstance(result, dict):
        try:
            return state_class(**result)
        except Exception:
            return None
    return result


def _extract_agent_info_from_state(state: Any) -> Dict[str, Optional[str]]:
    """Safely extract agent info from pipeline state."""
    try:
        ingestion = getattr(state, "ingestion", None)
        if ingestion is None:
            return {"agent_name": None, "agent_type": None, "tool": None}
        return {
            "agent_name": getattr(ingestion, "agent_name", None),
            "agent_type": getattr(ingestion, "agent_type", None),
            "tool": getattr(ingestion, "tool", None),
        }
    except Exception:
        return {"agent_name": None, "agent_type": None, "tool": None}


def log_agent_event(state: Any, node_name: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Append a JSONL log entry per agent under outputs/agent_logs/.

    The log filename is derived from agent_name when available, otherwise 'unknown'.
    """
    agent = _extract_agent_info_from_state(state)
    agent_name = agent.get("agent_name") or "unknown"

    log_dir = os.path.join(OUTPUT_DIR, "agent_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{agent_name}.log")

    # Collect common context
    context: Dict[str, Any] = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "node": node_name,
        "message": message,
        "agent_name": agent_name,
        "agent_type": agent.get("agent_type"),
        "tool": agent.get("tool"),
    }

    # Optionally include S3 and doc info if present
    try:
        ingestion = getattr(state, "ingestion", None)
        if ingestion is not None:
            context["s3_bucket"] = getattr(ingestion, "s3_bucket", None)
            context["s3_key"] = getattr(ingestion, "s3_key", None)
            context["document_type"] = getattr(ingestion, "document_type", None)
    except Exception:
        pass

    try:
        ocr = getattr(state, "ocr", None)
        if ocr is not None:
            context["ocr_mode"] = getattr(ocr, "mode", None)
            context["ocr_doc_category"] = getattr(ocr, "doc_category", None)
    except Exception:
        pass

    if extra and isinstance(extra, dict):
        context.update(extra)

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(context, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Warning: failed to write agent log {log_path}: {e}")


def check_network_connectivity(host: str = "8.8.8.8", port: int = 53, timeout: int = 3) -> bool:
    """Check if network connectivity is available by attempting to connect to a DNS server."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except (socket.error, OSError):
        return False


def check_dns_resolution(hostname: str = "aws.amazon.com") -> bool:
    """Check if DNS resolution is working."""
    try:
        socket.gethostbyname(hostname)
        return True
    except (socket.gaierror, OSError):
        return False


def diagnose_network_issue() -> str:
    """Diagnose network connectivity issues and return a helpful message."""
    issues = []
    
    if not check_network_connectivity():
        issues.append("No internet connectivity detected")
    
    if not check_dns_resolution():
        issues.append("DNS resolution is failing")
    
    if issues:
        return (
            "Network connectivity issue detected:\n"
            f"  - {'; '.join(issues)}\n"
            "Please check:\n"
            "  1. Your internet connection\n"
            "  2. DNS server settings\n"
            "  3. Firewall/proxy configuration\n"
            "  4. VPN connection (if applicable)\n"
            "  5. AWS service availability"
        )
    return ""
