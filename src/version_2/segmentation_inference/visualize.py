import cv2
import numpy as np

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


class Visualizer:

    def __init__(self):

        self.colors = [
            (0,255,0),
            (255,0,0),
            (0,0,255),
            (255,255,0),
            (255,0,255),
            (0,255,255),
            (150,100,255),
            (255,150,100),
            (100,255,150),
            (180,180,0)
        ]

    def draw(self, image, result):

        boxes = result["boxes"]
        labels = result["labels"]
        scores = result["scores"]      # confidence
        masks = result.get("masks")    # optional

        output = image.copy()

        h, w = image.shape[:2]

        for i in range(len(boxes)):

            x1, y1, x2, y2 = map(int, boxes[i])

            class_id = int(labels[i])
            score = float(scores[i])

            class_name = CLASS_NAMES.get(class_id, str(class_id))

            color = self.colors[class_id % len(self.colors)]

            # ----------------------------
            # Draw mask (same idea as notebook)
            # ----------------------------
            if masks is not None:

                mask = masks[i]

                # resize exactly like notebook
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )

                mask = (mask > 0).astype(np.uint8)

                colored = np.zeros_like(output)

                colored[:, :, 0] = mask * color[0]
                colored[:, :, 1] = mask * color[1]
                colored[:, :, 2] = mask * color[2]

                output = cv2.addWeighted(output, 1.0, colored, 0.35, 0)

            # ----------------------------
            # Draw box
            # ----------------------------
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # ----------------------------
            # Draw label (same as notebook)
            # ----------------------------
            text = f"{class_name} {score:.2f}"

            cv2.putText(
                output,
                text,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )

        return output