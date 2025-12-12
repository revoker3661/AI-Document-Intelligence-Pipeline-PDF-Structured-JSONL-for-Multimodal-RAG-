import os
import json
from unstructured.partition.pdf import partition_pdf
from unstructured_inference.models.tables import UnstructuredTableTransformerModel
from unstructured_inference.models.detectron2 import Detectron2LayoutModel

INPUT_PDF = "pdf_check/input/test.pdf"
OUTPUT_JSON = "pdf_check/output/parsed_full.json"

os.makedirs("pdf_check/output", exist_ok=True)

print("============================================================")
print("🔍 ENHANCED PIPELINE TEST (Version-Compatible)")
print("============================================================")

results = {
    "text_elements": [],
    "tables": [],
    "layout": []
}

# ---------- TEXT EXTRACTION ----------
print("\n[1] Extracting text with partition_pdf ...")
try:
    # 'hi_res' strategy works without extract_images_in_pdf
    elements = partition_pdf(filename=INPUT_PDF, strategy="hi_res")
    results["text_elements"] = [{"type": getattr(e, "category", "Text"), "text": str(e)} for e in elements[:30]]
    print(f"✅ Got {len(elements)} text elements")
except Exception as e:
    print(f"❌ Text extraction failed: {e}")

# ---------- TABLE EXTRACTION ----------
print("\n[2] Extracting tables with TableTransformer ...")
try:
    table_model = UnstructuredTableTransformerModel()
    # v0.7.23 uses .process_document not .predict
    tables = table_model.process_document(INPUT_PDF)
    for i, t in enumerate(tables):
        results["tables"].append({
            "page": getattr(t, "metadata", None),
            "text": str(t)
        })
    print(f"✅ Extracted {len(tables)} tables")
except Exception as e:
    print(f"❌ Table extraction failed: {e}")

# ---------- LAYOUT EXTRACTION ----------
print("\n[3] Detecting layout (figures, tables, etc.) with Detectron2 ...")
try:
    layout_model = Detectron2LayoutModel()
    # v0.7.23 uses .process_document instead of .predict
    layout = layout_model.process_document(INPUT_PDF)
    for l in layout:
        results["layout"].append({
            "type": getattr(l, "type", "unknown"),
            "bbox": getattr(l, "coordinates", None),
            "page": getattr(l.metadata, "page_number", None) if hasattr(l, "metadata") else None
        })
    print(f"✅ Detected {len(layout)} layout elements")
except Exception as e:
    print(f"❌ Layout extraction failed: {e}")

# ---------- SAVE RESULTS ----------
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print("\n📂 Results saved to", OUTPUT_JSON)
print("ALL TESTS COMPLETED 🎉")
