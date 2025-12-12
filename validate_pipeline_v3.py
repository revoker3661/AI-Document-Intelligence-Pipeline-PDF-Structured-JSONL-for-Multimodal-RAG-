import os
import json
import fitz  # PyMuPDF
from unstructured.partition.pdf import partition_pdf
from unstructured_inference.models.tables import UnstructuredTableTransformerModel
from unstructured_inference.models.detectron2 import Detectron2LayoutModel

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PDF = os.path.join(BASE_DIR, "pdf_check", "input", "test.pdf")
OUTPUT_JSON = os.path.join(BASE_DIR, "pdf_check", "output", "parsed_v3.json")

results = {"text_elements": [], "tables": [], "layout": []}

print("\n============================================================")
print("🔍 ENHANCED PIPELINE TEST (v3 - Version Compatible)")
print("============================================================\n")

# 1. Text Extraction
print("[1] Extracting text with partition_pdf ...")
try:
    elements = partition_pdf(filename=INPUT_PDF)
    results["text_elements"] = [{"type": el.category, "text": el.text} for el in elements if hasattr(el, "text")]
    print(f"✅ Extracted {len(results['text_elements'])} text elements")
except Exception as e:
    print("❌ Text extraction failed:", e)

# 2. Table Extraction
print("\n[2] Extracting tables with TableTransformer ...")
try:
    model = UnstructuredTableTransformerModel()
    pdf = fitz.open(INPUT_PDF)
    tables_out = []

    for i, page in enumerate(pdf):
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")
        preds = model.predict(img_bytes)
        tables_out.append({"page": i + 1, "preds": str(preds)})

    results["tables"] = tables_out
    print(f"✅ Extracted tables from {len(tables_out)} pages")
except Exception as e:
    print("❌ Table extraction failed:", e)

# 3. Layout Detection
print("\n[3] Detecting layout (figures, tables, etc.) with Detectron2 ...")
try:
    # PubLayNet config shortcut (provided by layoutparser / detectron2 hub)
    config = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
    layout_model = Detectron2LayoutModel(config)
    layout = layout_model.predict(INPUT_PDF)
    results["layout"] = str(layout)[:500]  # truncate long output
    print("✅ Layout extraction ran successfully")
except Exception as e:
    print("❌ Layout extraction failed:", e)

# Save results
os.makedirs(os.path.join(BASE_DIR, "pdf_check", "output"), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n📂 Results saved to {OUTPUT_JSON}")
print("ALL TESTS COMPLETED 🎉")
