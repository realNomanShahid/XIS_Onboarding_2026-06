import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from rfdetr import RFDETRSegSmall


def load_model(model_path, patch_size=12, num_classes=10):
    model = RFDETRSegSmall(
        pretrain_weights=model_path,
        patch_size=patch_size,
        num_classes=num_classes
    )
    return model


def run_inference(model, image_path, output_path, alpha=0.4):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_overlay = image_rgb.copy()

    results = model.predict(image_rgb)

    boxes = results.xyxy
    scores = results.confidence
    class_names = results.data["class_name"]
    masks = results.mask

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(boxes), 3), dtype=np.uint8)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        score = float(scores[i])
        cls = class_names[i]
        color = [int(c) for c in colors[i]]

        # Apply mask if available
        if masks is not None and len(masks) > i:
            instance_mask = masks[i]

            if isinstance(instance_mask, np.ndarray):
                if instance_mask.ndim == 2:
                    mask_overlay[instance_mask > 0] = color

                elif instance_mask.ndim == 3:
                    cv2.fillPoly(mask_overlay, [instance_mask.astype(np.int32)], color)

        # Draw bounding box
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)

        # Label
        cv2.putText(
            image_rgb,
            f"{cls} {score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    # Blend mask + image
    final_image = cv2.addWeighted(mask_overlay, alpha, image_rgb, 1 - alpha, 0)

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, final_bgr)

    print(f"[INFO] Output saved at: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", default="output.jpg", help="Path to save output image")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--patch", type=int, default=12, help="Patch size")

    args = parser.parse_args()

    model = load_model(
        model_path=args.model,
        patch_size=args.patch,
        num_classes=args.classes
    )

    run_inference(model, args.image, args.output)


if __name__ == "__main__":
    main()