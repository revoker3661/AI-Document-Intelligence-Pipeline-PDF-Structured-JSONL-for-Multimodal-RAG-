import torch

print("==================================================")
print("🔍 PYTORCH CHECK")
print("==================================================")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available?: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Torch CUDA version: {torch.version.cuda}")
    x = torch.rand(2, 2).cuda()
    print(f"CUDA tensor test successful: {x}")

print("\n==================================================")
print("🔍 DETECTRON2 CHECK")
print("==================================================")
try:
    import detectron2
    from detectron2.utils.logger import setup_logger

    print(f"Detectron2 version: {detectron2.__version__}")
    setup_logger()
    print("✅ Detectron2 imported successfully.")
except Exception as e:
    print("❌ Detectron2 import failed!")
    print(e)
