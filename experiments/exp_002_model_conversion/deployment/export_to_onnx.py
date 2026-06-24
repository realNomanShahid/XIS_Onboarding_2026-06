import os
import time
import torch
import torchvision
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image


device = torch.device("cpu")

MODEL_PATH = "../checkpoints/best_model.pth"
ONNX_PATH = "convnext.onnx"
IMAGE_PATH = "../28.jpg"

def preprocess_image(img_path):
    if os.path.exists(img_path):
        img = Image.open(img_path).convert('RGB')
    else:
        img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

input_tensor = preprocess_image(IMAGE_PATH).to(device)


model = torchvision.models.convnext_tiny(num_classes=102)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

with torch.no_grad():
    pytorch_output = model(input_tensor)


dummy_input = torch.randn(1, 3, 224, 224, device=device)
torch.onnx.export(
    model, dummy_input, ONNX_PATH, export_params=True, opset_version=18,
    do_constant_folding=True, input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}, dynamo=False
)


onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)

ort_session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
input_numpy = input_tensor.numpy()
ort_outputs = ort_session.run(['output'], {'input': input_numpy})


np.testing.assert_allclose(pytorch_output.numpy(), ort_outputs[0], rtol=1e-03, atol=1e-05)
print("Pipeline Validation Script Executed Successfully.")