from preprocess import Preprocessor
from inference import ONNXClassifier
from postprocess import PostProcessor
from visualize import Visualizer
from utils import create_output_directory

import cv2


MODEL_PATH = r"C:\Users\arman\OneDrive\Desktop\preprocess\convnext.onnx"
IMAGE_PATH = r"C:\Users\arman\Downloads\32.jpg"


def main():

    create_output_directory()

    print("Loading model...")
    model = ONNXClassifier(MODEL_PATH)

    print("Loading image...")
    preprocessor = Preprocessor()

    image_tensor, original_image = preprocessor.preprocess(IMAGE_PATH)

    print("Running inference...")
    outputs = model.predict(image_tensor)

    print("Postprocessing...")
    postprocessor = PostProcessor()

    result = postprocessor.process(outputs)

    print(f"Predicted Class : {result['class_id']}")
    print(f"Confidence      : {result['confidence']*100:.2f}%")

    print("Visualizing...")

    visualizer = Visualizer()

    output_image = visualizer.draw(original_image.copy(), result)

    cv2.imwrite("output/result.jpg", output_image)

    print("Saved : output/result.jpg")

    cv2.imshow("Prediction", output_image)

    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()