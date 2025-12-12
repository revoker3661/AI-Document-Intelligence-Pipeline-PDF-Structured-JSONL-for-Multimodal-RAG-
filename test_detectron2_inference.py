import torch, detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import numpy as np
import cv2

print("Torch version:", torch.__version__)
print("Detectron2 version:", detectron2.__version__)

# ✅ Detect CUDA
print("CUDA available?:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Running on GPU:", torch.cuda.get_device_name(0))

# Load config + pretrained model (Faster R-CNN COCO)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
)
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

# Create a dummy image (white background with black square)
image = np.full((480, 640, 3), 255, dtype=np.uint8)
cv2.rectangle(image, (200, 150), (400, 350), (0, 0, 0), -1)

# Run inference
outputs = predictor(image)
print("\n✅ Inference successful!")
print("Predicted classes:", outputs["instances"].pred_classes.cpu().numpy())
print("Predicted boxes:", outputs["instances"].pred_boxes.tensor.cpu().numpy())
