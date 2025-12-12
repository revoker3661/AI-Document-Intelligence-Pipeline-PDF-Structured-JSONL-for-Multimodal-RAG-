# check_transformer.py (Final, Researched & Corrected Version)

import torch
# Sahi classes import karna
from transformers import TableTransformerForObjectDetection, AutoImageProcessor
import traceback

model_name = "microsoft/table-transformer-structure-recognition"

print(f"🔄 Attempting to download and load the model AND its image processor: '{model_name}'...")
print("   (This might take a few minutes if it's the first time)...")

try:
    # Model ko uski specific class se load karna (ye sahi tha)
    model = TableTransformerForObjectDetection.from_pretrained(model_name)
    
    # Text Tokenizer ki jagah, sahi Image Processor load karna
    # Yahi sabse bada aur important fix hai
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    
    # Model ko GPU par bhejna
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    print("\n" + "="*60)
    print(f"✅ SUCCESS: Hugging Face Table Transformer model and its image processor loaded successfully on '{device}'.")
    print("   - Your environment is now 100% ready.")
    print("="*60)

except Exception as e:
    print("\n" + "="*60)
    print(f"❌ FAILED: Could not load the model.")
    print(f"   - Error: {e}")
    print("\n--- Full Error Traceback ---")
    traceback.print_exc()
    print("="*60)