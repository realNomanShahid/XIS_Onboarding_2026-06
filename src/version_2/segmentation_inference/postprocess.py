import cv2
import numpy as np
import json


class PostProcessor:

    def __init__(self,
                 class_map_path="class_to_idx.json",
                 conf_threshold=0.5,
                 mask_threshold=0.5,
                 background_class=10):

        # load class mapping (index → class name/id)
        with open(class_map_path, "r") as f:
            class_to_idx = json.load(f)

        self.idx_to_class = {
            v: k for k, v in class_to_idx.items()
}

        self.CONF_THRESHOLD = conf_threshold
        self.MASK_THRESHOLD = mask_threshold
        self.BACKGROUND_CLASS = background_class

    # ----------------------------------------------------
    # Sigmoid
    # ----------------------------------------------------

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # ----------------------------------------------------
    # Convert box
    # ----------------------------------------------------

    @staticmethod
    def convert_box(box, img_w, img_h):

        cx, cy, w, h = box

        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h

        return [int(x1), int(y1), int(x2), int(y2)]

    # ----------------------------------------------------
    # Main Postprocess
    # ----------------------------------------------------

    def process(self, outputs, original_size):

        img_h, img_w = original_size

        dets = outputs["dets"][0]
        labels = outputs["labels"][0]
        masks = outputs["masks"][0]

        final_boxes = []
        final_scores = []
        final_labels = []
        final_masks = []

        for i in range(len(dets)):

            logits = labels[i]
            probs = self.sigmoid(logits)

            class_id = int(np.argmax(probs))
            confidence = float(probs[class_id])

            # background filter
            if class_id == self.BACKGROUND_CLASS:
                continue

            # confidence filter
            if confidence < self.CONF_THRESHOLD:
                continue

            # box
            box = self.convert_box(dets[i], img_w, img_h)

            # mask
            mask = self.sigmoid(masks[i])
            mask = (mask > self.MASK_THRESHOLD).astype(np.uint8)

            mask = cv2.resize(
                mask,
                (img_w, img_h),
                interpolation=cv2.INTER_NEAREST
            )

            # class mapping (FIXED)
            class_name = self.idx_to_class.get(class_id, str(class_id))

            final_boxes.append(box)
            final_scores.append(confidence)
            final_labels.append(class_name)
            final_masks.append(mask)

        return {
            "boxes": np.array(final_boxes),
            "scores": np.array(final_scores),
            "labels": final_labels,
            "masks": final_masks
        }