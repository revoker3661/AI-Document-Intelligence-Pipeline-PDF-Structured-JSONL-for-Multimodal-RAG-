import os
import torch
import detectron2
import unstructured
import unstructured_inference
import subprocess, json

print("="*60)
print("🔍 ENVIRONMENT VERSION CHECK")
print("="*60)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Detectron2:", detectron2.__version__)

# ---- Unstructured version fix ----
try:
    print("unstructured:", unstructured.__version__)
except AttributeError:
    try:
        import unstructured.__version__ as un_v
        print("unstructured:", getattr(un_v, "__version__", "unknown"))
    except Exception as e:
        print("unstructured: version not found", e)

# ---- Unstructured-inference version ----
print("unstructured-inference:", getattr(unstructured_inference, "__version__", "N/A"))

print("\n" + "="*60)
print("🔍 GLIBCXX LIB CHECK")
print("="*60)
lib_path = subprocess.check_output(
    "ldd $(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), \"lib\", \"libtorch_cpu.so\"))') | grep libstdc++",
    shell=True, text=True
).strip()
print(lib_path)

print("\n" + "="*60)
print("🔍 DETECTRON2 INFERENCE TEST")
print("="*60)
dummy = torch.randn(2, 3, 224, 224).cuda()
print("Dummy tensor on CUDA ✅", dummy.shape)

print("\n" + "="*60)
print("🔍 UNSTRUCTURED PDF TEST")
print("="*60)

from unstructured.partition.pdf import partition_pdf

pdf_path = "pdf_check/input/test.pdf"
output_path = "pdf_check/output/parsed.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

try:
    elements = partition_pdf(pdf_path)   # no extract_images_in_pdf
    print(f"✅ PDF processed, got {len(elements)} elements")
    sample = [{"type": e.category, "text": (e.text or "")[:80]} for e in elements[:5]]
    print("Sample extracted:", json.dumps(sample, indent=2))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([{"type": e.category, "text": e.text} for e in elements], f, indent=2, ensure_ascii=False)

    print(f"📂 Results saved to {output_path}")
except Exception as e:
    print("❌ partition_pdf failed:", e)

print("\nALL TESTS COMPLETED 🎉")
