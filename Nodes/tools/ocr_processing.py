"""
OCR processing utilities for document analysis.
"""

from typing import Dict, Any, List
import json

from ..tools.aws_services import run_textract_async_s3, run_analyze_id_s3, generate_presigned_url
from ..tools.llm_services import chat_json, classify_via_image, extract_via_image, remove_raw_text_fields
from ..config.settings import OPENAI_MODEL, ROUTE_LABELS


def group_blocks_by_page(blocks: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Group Textract blocks by page number."""
    pages: Dict[int, List[Dict[str, Any]]] = {}
    for b in blocks:
        page = b.get("Page", 1)
        pages.setdefault(page, []).append(b)
    return pages


def resolve_kv_pairs_from_page_blocks(page_blocks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Resolve key-value pairs from page blocks."""
    by_id = {b.get("Id"): b for b in page_blocks if b.get("Id")}
    kvs: List[Dict[str, str]] = []

    for b in page_blocks:
        if b.get("BlockType") != "KEY_VALUE_SET":
            continue
        entity = b.get("EntityTypes", [])
        if "KEY" not in entity:
            continue

        key_text_parts: List[str] = []
        value_text_parts: List[str] = []

        for rel in b.get("Relationships", []) or []:
            rtype = rel.get("Type")
            if rtype == "CHILD":
                for cid in rel.get("Ids", []) or []:
                    cb = by_id.get(cid, {})
                    t = cb.get("Text")
                    if t:
                        key_text_parts.append(t)

            elif rtype == "VALUE":
                for vid in rel.get("Ids", []) or []:
                    vb = by_id.get(vid, {})
                    for vrel in vb.get("Relationships", []) or []:
                        if vrel.get("Type") == "CHILD":
                            for vcid in vrel.get("Ids", []) or []:
                                vcb = by_id.get(vcid, {})
                                vt = vcb.get("Text")
                                if vt:
                                    value_text_parts.append(vt)

        k = " ".join(key_text_parts).strip()
        v = " ".join(value_text_parts).strip()
        if k or v:
            kvs.append({"key": k, "value": v})

    return kvs


def cells_from_page_blocks(page_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract cells from page blocks."""
    by_id = {blk.get("Id"): blk for blk in page_blocks if blk.get("Id")}
    cells: List[Dict[str, Any]] = []
    for b in page_blocks:
        if b.get("BlockType") != "CELL":
            continue
        text_parts: List[str] = []
        for rel in b.get("Relationships", []) or []:
            if rel.get("Type") != "CHILD":
                continue
            for cid in rel.get("Ids", []) or []:
                cb = by_id.get(cid, {})
                if cb.get("BlockType") == "WORD":
                    t = cb.get("Text")
                    if t:
                        text_parts.append(t)
                elif cb.get("BlockType") == "SELECTION_ELEMENT":
                    status = cb.get("SelectionStatus")
                    if status:
                        text_parts.append(f"[CHECKBOX:{status}]")
                elif cb.get("BlockType") == "LINE":
                    t = cb.get("Text")
                    if t:
                        text_parts.append(t)
        cell_text = " ".join(text_parts).strip()
        cells.append({
            "row": b.get("RowIndex"),
            "col": b.get("ColumnIndex"),
            "text": cell_text
        })
    return cells


def lines_words_from_page_blocks(page_blocks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Extract lines and words from page blocks."""
    lines, words = [], []
    for b in page_blocks:
        bt = b.get("BlockType")
        if bt == "LINE":
            t = b.get("Text", "")
            if t:
                lines.append(t)
        elif bt == "WORD":
            t = b.get("Text", "")
            if t:
                words.append(t)
    return {"lines": lines, "words": words}


def llm_route_from_ocr_page(page_view: Dict[str, Any]) -> str:
    """Route document type from OCR page data."""
    lines = page_view.get("lines", [])
    cells = page_view.get("cells", [])

    payload = {
        "head_lines": lines[:60],
        "tail_lines": lines[-30:],
        "cells_preview": [
            {"row": c.get("row"), "col": c.get("col"), "text": (c.get("text") or "")[:80]}
            for c in cells[:80]
        ],
    }

    system = (
        "You are a cautious document-type classifier for business and identity documents. "
        f"Choose exactly one label from {ROUTE_LABELS}. "
        "Return a single JSON object exactly in the form {\"doc_type\":\"<label>\"}. "
        "Rules:\n"
        "- If not highly confident, return 'unknown'. Never guess.\n"
        "- Base your decision only on the provided snippets.\n"
        "- Only output a JSON object; no extra text."
    )

    out = chat_json(OPENAI_MODEL, system, payload) or {}
    label = out.get("doc_type", "unknown")
    return label if label in ROUTE_LABELS else "unknown"


def route_document_type_from_ocr(simplified: Dict[str, Any]) -> str:
    """Route document type from OCR data."""
    if not simplified.get("pages"):
        return "unknown"
    first_page_key = sorted(simplified["pages"].keys(), key=lambda x: int(x))[0]
    return llm_route_from_ocr_page(simplified["pages"][first_page_key])


def analyze_id_to_kvs(resp: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Convert AnalyzeID response to generic kv list.
    Structure:
      IdentityDocuments -> [ { DocumentIndex, DocumentType, Fields:[{Type:{Text}, ValueDetection:{Text}}] } ]
    """
    kvs: List[Dict[str, str]] = []
    try:
        docs = resp.get("IdentityDocuments", [])
        for doc in docs:
            dtype = doc.get("DocumentType")
            if dtype:
                kvs.append({"key": "DocumentType", "value": str(dtype)})
            for f in doc.get("Fields", []):
                k = (f.get("Type", {}) or {}).get("Text", "")
                v = (f.get("ValueDetection", {}) or {}).get("Text", "")
                if k or v:
                    kvs.append({"key": k, "value": v})
    except Exception:
        pass
    return kvs
