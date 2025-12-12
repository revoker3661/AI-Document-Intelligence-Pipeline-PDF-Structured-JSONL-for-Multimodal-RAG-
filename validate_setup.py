# validate_setup.py (Full and Correct Version)

import os
import sys
import subprocess
import logging
from pathlib import Path
import layoutparser as lp
import torch
import detectron2

# Basic logging setup
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def validate_all():
    logging.info("="*60)
    logging.info("🚀 STARTING FINAL SETUP VALIDATION 🚀")
    logging.info("="*60)

    # --- 1. GLIBCXX Check ---
    logging.info("\n[1/4] Checking C++ Library (GLIBCXX)...")
    try:
        lib_path = os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cpu.so")
        output = subprocess.check_output(["ldd", lib_path]).decode()
        libstdc_line = [line for line in output.splitlines() if "libstdc++" in line][0]
        if "/usr/lib/x86_64-linux-gnu" in libstdc_line:
            logging.info("  -> ✅ SUCCESS: Using system's modern libstdc++. GLIBCXX issue is fixed.")
        else:
            logging.warning("  -> ⚠️ WARNING: Using Conda's libstdc++. GLIBCXX errors might occur.")
    except Exception as e:
        logging.error(f"  -> ❌ FAILED: Could not check GLIBCXX: {e}")

    # --- 2. Core Libraries & GPU Check ---
    logging.info("\n[2/4] Checking Core Libraries and GPU...")
    try:
        import paddleocr
        import unstructured
        import unstructured_inference
        logging.info(f"  - PyTorch version: {torch.__version__} (CUDA Available: {torch.cuda.is_available()})")
        logging.info(f"  - Detectron2 version: {detectron2.__version__}")
        logging.info("  -> ✅ SUCCESS: Core libraries imported correctly.")
    except ImportError as e:
        logging.error(f"  -> ❌ FAILED: A required library is not installed: {e}")
        return

    # --- 3. Model Cache Check ---
    logging.info("\n[3/4] Checking Model Caches (indirectly by loading)...")
    
    # --- 4. Live Model Loading Test ---
    logging.info("\n[4/4] Performing Live Model Loading Test...")
    try:
        logging.info("  -> Initializing PubLayNet model via LayoutParser...")
        _ = lp.Detectron2LayoutModel(
            config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
        )
        logging.info("  -> ✅ SUCCESS: Detectron2 (PubLayNet) model initialized.")
        
        from paddleocr import PaddleOCR
        _ = PaddleOCR(use_gpu=True, lang='en', show_log=False)
        logging.info("  -> ✅ SUCCESS: PaddleOCR engine initialized.")
    except Exception as e:
        logging.error(f"  -> ❌ FAILED: Could not initialize a model: {e}", exc_info=True)

    logging.info("\n"+"="*60)
    logging.info("✅ VALIDATION COMPLETE ✅")
    logging.info("="*60)

if __name__ == "__main__":
    validate_all()