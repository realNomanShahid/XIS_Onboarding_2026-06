import onnxruntime as ort


class ONNXInference:

    def __init__(self, model_path):

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

        outputs = self.session.get_outputs()

        self.output_names = [o.name for o in outputs]

        print("=" * 60)
        print("ONNX Runtime Loaded")
        print("=" * 60)
        print("Input :", self.input_name)
        print("Outputs :", self.output_names)
        print("=" * 60)

    def infer(self, input_tensor):

        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_tensor}
        )

        return {
            "dets": outputs[0],
            "labels": outputs[1],
            "masks": outputs[2]
        }