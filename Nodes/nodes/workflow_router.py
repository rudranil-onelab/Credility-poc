"""
Workflow routing logic for the pipeline.
"""

from ..config.state_models import PipelineState


def Classified_or_not(state: PipelineState) -> str:
    """
    Route based on classification results.
    If classification failed, route back to OCR; else proceed to extraction.
    """
    if state.classification is None:
        return "OCR"
    return "OCR" if not state.classification.passed else "Document Data Extraction"
