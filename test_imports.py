# test_imports.py
try:
    from unstructured_inference.models.detectron2.table import load_table_model
    print("✅ load_table_model import successful")
except Exception as e:
    print("❌ load_table_model import failed:", e)

from unstructured.partition.pdf import partition_pdf
print("✅ partition_pdf import successful")
