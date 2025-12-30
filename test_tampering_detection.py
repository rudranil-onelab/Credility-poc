#!/usr/bin/env python3
"""
Quick test script for Visual Tampering Detection feature.

Usage:
    python test_tampering_detection.py <path_to_document_image>

Example:
    python test_tampering_detection.py test_documents/aadhaar.jpg
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Nodes.config.state_models import PipelineState, IngestionState
from Nodes.nodes.ocr_node import OCR
from Nodes.nodes.classification_node import Classification
from Nodes.nodes.extraction_node import Extract
from Nodes.nodes.validation_check_node import ValidationCheck


def test_tampering_detection(file_path: str):
    """Test tampering detection on a document."""
    
    print("\n" + "=" * 80)
    print("🧪 TESTING VISUAL TAMPERING DETECTION")
    print("=" * 80)
    print(f"\nFile: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return None
    
    # Check file type
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in ['.jpg', '.jpeg', '.png']:
        print(f"⚠️  Warning: Visual tampering detection works best with image files (JPG/PNG)")
        print(f"   File type: {file_ext}")
    
    # Create pipeline state
    state = PipelineState()
    
    # Setup ingestion (local file)
    content_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.pdf': 'application/pdf'
    }
    
    state.ingestion = IngestionState(
        s3_bucket="local",
        s3_key=file_path,
        document_name=os.path.basename(file_path),
        tool="llm",
        content_type=content_type_map.get(file_ext, 'application/octet-stream'),
        FPCID="TEST",
        LMRId="TEST"
    )
    
    try:
        # Run pipeline steps
        print("\n" + "-" * 80)
        print("Step 1/4: Running OCR...")
        print("-" * 80)
        state = OCR(state)
        
        if not state.ocr:
            print("❌ OCR failed")
            return None
        
        print("\n" + "-" * 80)
        print("Step 2/4: Running Classification...")
        print("-" * 80)
        state = Classification(state)
        
        if not state.classification:
            print("❌ Classification failed")
            return None
        
        print(f"\n✓ Document Type Detected: {state.classification.detected_doc_type}")
        
        print("\n" + "-" * 80)
        print("Step 3/4: Running Extraction...")
        print("-" * 80)
        state = Extract(state)
        
        if not state.extraction:
            print("❌ Extraction failed")
            return None
        
        print(f"\n✓ Fields Extracted: {len(state.extraction.extracted or {})}")
        
        print("\n" + "-" * 80)
        print("Step 4/4: Running Validation (with Tampering Detection)...")
        print("-" * 80)
        print("\n⚠️  NOTE: Visual tampering detection requires S3 URL generation.")
        print("   For local files, this test will skip visual analysis.")
        print("   Upload to S3 first for full tampering detection testing.\n")
        
        state = ValidationCheck(state)
        
        # Print final results
        print("\n" + "=" * 80)
        print("📊 FINAL RESULTS")
        print("=" * 80)
        
        if state.extraction and state.extraction.passed:
            print("\n✅ VALIDATION PASSED")
        else:
            print("\n❌ VALIDATION FAILED")
            if state.extraction and hasattr(state.extraction, 'message'):
                print(f"   Reason: {state.extraction.message}")
        
        print("\n" + "=" * 80)
        print("✅ Test completed!")
        print("=" * 80)
        
        return state
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("=" * 80)
        print("Visual Tampering Detection - Test Script")
        print("=" * 80)
        print("\nUsage:")
        print(f"  python {sys.argv[0]} <path_to_document_image>")
        print("\nExample:")
        print(f"  python {sys.argv[0]} test_documents/aadhaar.jpg")
        print("\nNote:")
        print("  - Visual tampering detection works best with JPG/PNG images")
        print("  - For full testing, upload to S3 first (local files skip visual analysis)")
        print("  - See TAMPERING_DETECTION_TEST_GUIDE.md for detailed instructions")
        print("=" * 80)
        sys.exit(1)
    
    file_path = sys.argv[1]
    result = test_tampering_detection(file_path)
    
    if result is None:
        sys.exit(1)

