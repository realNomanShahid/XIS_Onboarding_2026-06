import cv2

class Visualizer:

    def draw(self, image, result):

        class_id = result["class_id"]
        confidence = result["confidence"] * 100

        text1 = f"Predicted Class : {class_id}"
        text2 = f"Confidence      : {confidence:.2f}%"

        # White rectangle background
        cv2.rectangle(image,
                      (10, 10),
                      (350, 90),
                      (255, 255, 255),
                      -1)

        cv2.putText(image,
                    text1,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2)

        cv2.putText(image,
                    text2,
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2)

        return image