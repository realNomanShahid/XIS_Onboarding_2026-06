import json
import numpy as np


class PostProcessor:
    

    def __init__(self, mapping_path="class_to_idx.json"):

        # Load class_to_idx mapping
        with open(mapping_path, "r") as f:
            class_to_idx = json.load(f)

        # Reverse the mapping: index -> class_id
        self.idx_to_class = {
            int(v): int(k)
            for k, v in class_to_idx.items()
        }

    @staticmethod
    def softmax(logits):

        logits = logits - np.max(logits)

        exp = np.exp(logits)

        return exp / np.sum(exp)

    def process(self, outputs):

        logits = outputs.squeeze()

        probabilities = self.softmax(logits)

        predicted_index = int(np.argmax(probabilities))

        confidence = float(probabilities[predicted_index])

        # Convert model index back to original class ID
        class_id = self.idx_to_class[predicted_index]

        return {
            "class_id": class_id,
            "confidence": confidence,
            "probabilities": probabilities
        }