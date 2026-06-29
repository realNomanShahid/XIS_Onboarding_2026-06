import cv2
import numpy as np


class Preprocessor:
    """
    Handles image preprocessing before ONNX inference.
    """

    def __init__(self,
                 input_size=(224, 224),
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):

        self.input_size = input_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def preprocess(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(image_path)

        original = image.copy()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, self.input_size)

        image = image.astype(np.float32) / 255.0

        image = (image - self.mean) / self.std

        image = np.transpose(image, (2, 0, 1))

        image = np.expand_dims(image, axis=0)

        return image.astype(np.float32), original