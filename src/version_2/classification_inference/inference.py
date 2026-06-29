import onnxruntime as ort


class ONNXClassifier:

    def __init__(self, model_path):

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image_tensor):

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: image_tensor}
        )

        return outputs[0]