"""
Main pipeline orchestrator for document validation.

Pipeline flow:
1. Ingestion - Load document from S3
2. OCR - Extract text using Textract
3. Classification - Identify document type
4. Extraction - Extract structured data
5. Validation - Validate document (pass/fail)
"""

from langgraph.graph import END, StateGraph

from .config.state_models import PipelineState
from .nodes.ingestion_node import Ingestion
from .nodes.ocr_node import OCR
from .nodes.classification_node import Classification
from .nodes.extraction_node import Extract
from .nodes.validation_check_node import ValidationCheck
from .nodes.workflow_router import Classified_or_not


def create_pipeline() -> StateGraph:
    """
    Create and configure the main pipeline workflow.
    
    Pipeline flow:
    1. Ingestion -> OCR -> Classification
    2. If classification fails -> retry OCR
    3. If classification passes -> Extraction -> Validation -> END
    """
    workflow = StateGraph(PipelineState)

    # Add nodes
    workflow.add_node("Ingestion", Ingestion)
    workflow.add_node("OCR", OCR)
    workflow.add_node("Document Classification", Classification)
    workflow.add_node("Document Data Extraction", Extract)
    workflow.add_node("Validation Check", ValidationCheck)

    # Set entry point
    workflow.set_entry_point("Ingestion")
    
    # Add edges
    workflow.add_edge("Ingestion", "OCR")
    workflow.add_edge("OCR", "Document Classification")

    # Add conditional edges for classification
    workflow.add_conditional_edges(
        "Document Classification",
        Classified_or_not,
        {"Document Data Extraction": "Document Data Extraction", "OCR": "OCR"}
    )
    
    # Add validation check after extraction
    workflow.add_edge("Document Data Extraction", "Validation Check")
    
    # End after validation
    workflow.add_edge("Validation Check", END)

    return workflow


def get_compiled_pipeline():
    """
    Get the compiled pipeline ready for execution.
    """
    workflow = create_pipeline()
    return workflow.compile()
