"""
config.py
Project configuration file.
"""

import os

# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "experiments",
    "exp_004_seg_inference",
    "models",
    "model.onnx"
)

IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "main_data",
    "exp_004_seg_inference"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "experiments",
    "exp_004_seg_inference",
    "outputs"
)

MASK_DIR = os.path.join(
    PROJECT_ROOT,
    "experiments",
    "exp_004_seg_inference",
    "masks"
)

LOG_DIR = os.path.join(
    PROJECT_ROOT,
    "experiments",
    "exp_004_seg_inference",
    "logs"
)

METRIC_DIR = os.path.join(
    PROJECT_ROOT,
    "experiments",
    "exp_004_seg_inference",
    "metrics"
)

JSON_PATH = os.path.join(
    PROJECT_ROOT,
    "num_to_class.json"
)

# =====================================================
# MODEL
# =====================================================

INPUT_WIDTH = 640
INPUT_HEIGHT = 640

NUM_CLASSES = 10

CONF_THRESHOLD = 0.50

MASK_THRESHOLD = 0.50

BACKGROUND_CLASS = 10

# =====================================================
# COLORS
# =====================================================

SEED = 42