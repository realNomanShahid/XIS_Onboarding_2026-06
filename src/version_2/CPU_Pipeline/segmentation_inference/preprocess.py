import cv2
import numpy as np


class Preprocessor:

    def __init__(self, input_size=(640, 640)):

        self.width = input_size[0]
        self.height = input_size[1]

    def preprocess(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(image_path)

        original = image.copy()

        original_h, original_w = image.shape[:2]

        # BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        image = cv2.resize(image, (self.width, self.height))

        # Normalize
        image = image.astype(np.float32) / 255.0

        # HWC → CHW
        image = np.transpose(image, (2, 0, 1))

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return {
            "input_tensor": image.astype(np.float32),
            "original_image": original,
            "original_size": (original_h, original_w)
        }