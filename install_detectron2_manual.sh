#!/bin/bash
set -e

echo "=================================================="
echo "🚀 Option 2: Manual Detectron2 Installation"
echo "=================================================="

cd ~/Medical_AI_Processing/detectron2

# Clean old builds
rm -rf build/ detectron2.egg-info

# Build and install in develop mode
python setup.py build develop

# Verify installation
python - <<'PYCODE'
import torch, detectron2
print("Torch ✅", torch.__version__)
print("Detectron2 ✅", detectron2.__version__)
PYCODE

echo "=================================================="
echo "✅ Detectron2 installed successfully with Option 2"
echo "=================================================="
