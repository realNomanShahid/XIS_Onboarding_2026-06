"""
utils.py
Utility functions used throughout the project.
"""

import os
import json
import logging
import numpy as np
import cv2

from .config import *


# ------------------------------------------------------
# Create folders
# ------------------------------------------------------

def create_directories():

    directories = [

        OUTPUT_DIR,
        MASK_DIR,
        LOG_DIR,
        METRIC_DIR

    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# ------------------------------------------------------
# Load class mapping
# ------------------------------------------------------

def load_class_names():

    with open(JSON_PATH, "r") as f:
        return json.load(f)


# ------------------------------------------------------
# Logger
# ------------------------------------------------------

def get_logger():

    logger = logging.getLogger("Segmentation")

    logger.setLevel(logging.INFO)

    if not logger.handlers:

        handler = logging.FileHandler(
            os.path.join(LOG_DIR, "inference.log")
        )

        formatter = logging.Formatter(
            "%(asctime)s - %(message)s"
        )

        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger


# ------------------------------------------------------
# Save binary mask
# ------------------------------------------------------

def save_mask(mask, filename):

    mask = (mask * 255).astype(np.uint8)

    cv2.imwrite(

        os.path.join(MASK_DIR, filename),

        mask

    )