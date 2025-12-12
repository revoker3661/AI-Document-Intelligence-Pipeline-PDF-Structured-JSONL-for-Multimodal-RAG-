#!/bin/bash
set -e

echo "=================================================="
echo "🚀 Installing Detectron2 (editable mode)"
echo "=================================================="

# Go into detectron2 directory relative to script location
cd "$(dirname "$0")/detectron2"

# Install detectron2 without build isolation
pip install -e . --no-build-isolation

echo "=================================================="
echo "✅ Detectron2 installation finished!"
echo "=================================================="
