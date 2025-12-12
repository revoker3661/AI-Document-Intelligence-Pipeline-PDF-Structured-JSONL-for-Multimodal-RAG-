import torch, sys
import importlib

print("===== SANITY CHECK START =====")

# Torch + CUDA
print("\n⚡ Torch / CUDA:")
print("Torch version:", torch.__version__)
print("CUDA version (compiled):", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# TorchVision
try:
    import torchvision
    print("\n✅ TorchVision:", torchvision.__version__)
except Exception as e:
    print("\n❌ TorchVision issue:", e)

# Detectron2
try:
    import detectron2
    print("✅ Detectron2:", detectron2.__version__)
except Exception as e:
    print("❌ Detectron2 issue:", e)

# PaddleOCR
try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="en", use_gpu=True, show_log=False)
    print("✅ PaddleOCR ready (GPU enabled)")
except Exception as e:
    print("❌ PaddleOCR issue:", e)

# Unstructured Inference
try:
    from unstructured.partition.pdf import partition_pdf
    print("✅ Unstructured partition_pdf imported")
except Exception as e:
    print("❌ Unstructured issue:", e)

print("\n===== SANITY CHECK END =====")
