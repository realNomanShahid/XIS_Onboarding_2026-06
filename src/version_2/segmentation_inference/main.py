from preprocess import Preprocessor
from inference import ONNXInference
from postprocess import PostProcessor
from visualize import Visualizer

import cv2
import os
import csv
import time
import numpy as np

# =====================================================
# CLASS NAMES
# =====================================================

CLASS_NAMES = {
    0: "Valeo-MX",
    1: "solder",
    2: "solder-chip",
    3: "solder-large",
    4: "solder-long",
    5: "solder-middle",
    6: "solder-pin",
    7: "solder-square",
    8: "throughhole",
    9: "voids"
}

# =====================================================
# PATHS
# =====================================================

MODEL_PATH = r"C:\Users\arman\Downloads\rfdetr-seg-small.onnx"

IMAGE_PATH = r"C:\Users\arman\Downloads\23L2_bmp.rf.Ur4zutP5YxYkcLlo9xTh.bmp"

OUTPUT_DIR = r"D:\xis-onboarding\experiments\exp_004_seg_inference\outputs"

INPUT_SIZE = (384, 384)


# =====================================================
# SAVE CSV
# =====================================================

def save_csv(result, output_dir):

    csv_path = os.path.join(
        output_dir,
        f"results_{int(time.time())}.csv"
    )

    with open(csv_path, mode="w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["class", "score", "x1", "y1", "x2", "y2"])

        for i in range(len(result["boxes"])):

            x1, y1, x2, y2 = result["boxes"][i]

            label_id = int(result["labels"][i])
            label_name = CLASS_NAMES.get(label_id, f"Class-{label_id}")

            writer.writerow([
                label_name,
                float(result["scores"][i]),
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ])

    print(f"CSV Saved: {csv_path}")


# =====================================================
# MAIN
# =====================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading model...")
    model = ONNXInference(MODEL_PATH)

    print("Loading image...")
    preprocessor = Preprocessor(INPUT_SIZE)

    data = preprocessor.preprocess(IMAGE_PATH)

    image_tensor = data["input_tensor"]
    original_image = data["original_image"]
    original_size = data["original_size"]

    print("Running inference...")
    outputs = model.infer(image_tensor)

    print("Postprocessing...")

    POST_MAP = r"D:\xis-onboarding\src\version_2\segmentation_inference\num_to_class.json"

    postprocessor = PostProcessor(class_map_path=POST_MAP)

    result = postprocessor.process(outputs, original_size)

    print("=" * 50)
    print("RESULT")
    print("=" * 50)
    print(result)
    print("=" * 50)

    # =====================================================
    # VISUALIZATION
    # =====================================================

    print("Visualizing...")

    visualizer = Visualizer()

    output_image = visualizer.draw(original_image, result)

    img_path = os.path.join(OUTPUT_DIR, "result.jpg")

    cv2.imwrite(img_path, output_image)

    print("Image Saved:", img_path)

    # =====================================================
    # SAVE CSV
    # =====================================================

    save_csv(result, OUTPUT_DIR)


if __name__ == "__main__":
    main()