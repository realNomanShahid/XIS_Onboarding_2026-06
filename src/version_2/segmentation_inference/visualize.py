import cv2
import numpy as np


class Visualizer:

    def __init__(self):
        self.colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

    def draw(self, image, result):

        boxes = result["boxes"]
        labels = result["labels"]
        masks = result["masks"]

        overlay = image.copy()

        for i in range(len(boxes)):

            x1, y1, x2, y2 = boxes[i]
            label = labels[i]
            mask = masks[i]

            color = self.colors[i % len(self.colors)]

            # -------------------------
            # MASK (FIXED - no stacking)
            # -------------------------
            colored_mask = np.zeros_like(image, dtype=np.uint8)

            colored_mask[:, :, 0] = mask * color[0]
            colored_mask[:, :, 1] = mask * color[1]
            colored_mask[:, :, 2] = mask * color[2]

            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.35, 0)

            # -------------------------
            # BOX
            # -------------------------
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # -------------------------
            # LABEL (FIXED)
            # -------------------------
            cv2.putText(
                overlay,
                str(label),
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        return overlay