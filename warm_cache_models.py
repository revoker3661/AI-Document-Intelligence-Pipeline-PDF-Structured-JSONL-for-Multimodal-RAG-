# warm_cache_models.py (Full and Correct Version)

import logging
import nltk
from pathlib import Path
import layoutparser as lp
from paddleocr import PaddleOCR
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# Basic logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

def warm_all_caches():
    """
    Initializes all required models to trigger their download to the local cache.
    Run this once in a new environment with internet access.
    """
    logging.info("Starting model cache warming process...")

    # --- 1. Detectron2 (PubLayNet) Model via LayoutParser ---
    try:
        logging.info("  -> Caching Detectron2 PubLayNet model via LayoutParser...")
        # This is the most reliable way to cache the model LayoutParser uses
        _ = lp.Detectron2LayoutModel(
            config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
        )
        logging.info("  -> ✅ Detectron2 PubLayNet model cached successfully.")
    except Exception as e:
        logging.error(f"  -> ❌ Failed to cache Detectron2 model: {e}", exc_info=True)

    # --- 2. PaddleOCR Models ---
    try:
        logging.info("  -> Caching PaddleOCR English models...")
        # Initializing the OCR engine triggers model downloads
        _ = PaddleOCR(use_gpu=True, lang='en', show_log=False)
        logging.info("  -> ✅ PaddleOCR models cached successfully.")
    except Exception as e:
        logging.error(f"  -> ❌ Failed to cache PaddleOCR models: {e}", exc_info=True)

    # --- 3. NLTK Data ---
    try:
        logging.info("  -> Caching NLTK data (punkt, averaged_perceptron_tagger)...")
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        logging.info("  -> ✅ NLTK data cached successfully.")
    except Exception as e:
        logging.error(f"  -> ❌ Failed to cache NLTK data: {e}")
        
    logging.info("\n--- Cache Locations ---")
    logging.info(f"LayoutParser/Detectron2 Models are typically in: {Path.home()}/.torch/iopath_cache/")
    logging.info(f"PaddleOCR Models are in: {Path.home()}/.paddleocr/")
    logging.info(f"NLTK Data is in: {Path.home()}/nltk_data/")
    logging.info("-------------------------")
    logging.info("Cache warming process complete!")

if __name__ == "__main__":
    warm_all_caches()