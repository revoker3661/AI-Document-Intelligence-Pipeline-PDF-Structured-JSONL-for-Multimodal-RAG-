#!/bin/bash
set -e

echo "=================================================="
echo "🚀 Detectron2 Installation Script (v0.6)"
echo "=================================================="

# Step 1: Go to project root
cd ~/Medical_AI_Processing || exit 1

# Step 2: Clone detectron2 if not already present
if [ ! -d "detectron2" ]; then
    echo "📥 Cloning Detectron2 repo..."
    git clone https://github.com/facebookresearch/detectron2.git
    cd detectron2
    git checkout v0.6
else
    echo "🔄 Detectron2 folder already exists. Updating..."
    cd detectron2
    git fetch --all
    git checkout v0.6
fi

# Step 3: Install dependencies
echo "📦 Installing build dependencies..."
pip install -U pip setuptools wheel cython ninja pybind11

# Step 4: Build & install detectron2
echo "🔨 Building Detectron2..."
pip install -e . --no-build-isolation

# Step 5: Run sanity test
echo "=================================================="
echo "🧪 Running Detectron2 Sanity Test"
echo "=================================================="

python - <<'EOF'
import torch
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

print("✅ Detectron2 imported successfully!")

# Simple config test
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

predictor = DefaultPredictor(cfg)
dummy_input = torch.rand(1, 3, 224, 224).cuda()
print("✅ Dummy input tensor ready on GPU")

# Run forward pass
outputs = predictor.model(dummy_input)
print("✅ Detectron2 forward pass successful!")
EOF
