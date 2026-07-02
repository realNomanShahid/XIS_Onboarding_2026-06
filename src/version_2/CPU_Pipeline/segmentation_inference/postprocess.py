import cv2
import numpy as np
import json


class PostProcessor:

    def __init__(self,
                 conf_threshold=0.5,
                 mask_threshold=0.5,
                 background_class=10):

        self.idx_to_class = {
            0: "Valeo-MX",
            1: "solder",
            2: "solder-chip",
            3: "solder-large",
            4: "solder-long",
            5: "solder-middle",
            6: "solder-pin",
            7: "solder-square",
            8: "throughhole",
            9: "voids",
            10: "background"
        }

        self.CONF_THRESHOLD = conf_threshold
        self.MASK_THRESHOLD = mask_threshold
        self.BACKGROUND_CLASS = background_class

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(-x))

    def convert_box(self, box, img_w, img_h):

        cx, cy, w, h = box

        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h

        return [int(x1), int(y1), int(x2), int(y2)]

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

            probs = self.sigmoid(labels[i])

            class_id = int(np.argmax(probs))
            conf = float(probs[class_id])

            if conf < self.CONF_THRESHOLD:
                continue

            if class_id == self.BACKGROUND_CLASS:
                continue

            box = self.convert_box(dets[i], img_w, img_h)

            mask = self.sigmoid(masks[i])
            mask = (mask > self.MASK_THRESHOLD).astype(np.uint8)
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

            final_boxes.append(box)
            final_scores.append(conf)
            final_labels.append(class_id)
            final_masks.append(mask)

        return {
            "boxes": np.array(final_boxes),
            "scores": np.array(final_scores),
            "labels": np.array(final_labels),
            "masks": final_masks
        }