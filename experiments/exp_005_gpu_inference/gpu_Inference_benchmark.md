#  Inference Pipeline

## Overview

The GPU implementation and benchmarks were performed on **Google Colab (Tesla T4 GPU)** because a local NVIDIA GPU was not available.

---

## Pipeline

```
Input Image
     ↓
Image Preprocessing
     ↓
Model Inference
     ↓
Post-processing
     ↓
Visualization
```

---

## Implemented Features

- Image preprocessing
- ONNX Runtime CPU inference
- ONNX Runtime CUDA inference
- TensorRT FP16 inference
- GPU preprocessing (Resize & Normalization)
- Bounding box and segmentation mask visualization
- Performance benchmarking
- CPU vs GPU vs TensorRT comparison

---

## Hardware

### Local Machine
- CPU-based inference (Day 6)

### Google Colab
- NVIDIA Tesla T4 GPU
- CUDA-enabled ONNX Runtime
- TensorRT FP16

---

## Performance Results

| Pipeline | Latency (ms) | FPS | Speedup vs CPU |
|----------|-------------:|----:|---------------:|
| CPU | 937.98 | 1.07 | 1× |
| ONNX Runtime CUDA | 31.32 | 31.93 | 29.95× |
| TensorRT FP16 | 27.93 | 35.80 | 33.6× |

---

## Summary

- Successfully built the CPU inference pipeline.
- Accelerated inference using ONNX Runtime with CUDA.
- Converted the ONNX model to a TensorRT FP16 engine.
- Implemented GPU-based preprocessing.
- Compared CPU, ONNX Runtime CUDA, and TensorRT performance.
- Achieved approximately **33.6×** speedup over the CPU implementation.

---

## Notes

- TensorRT benchmarking was performed using **Google Colab (Tesla T4 GPU)**.
- Industrial camera integration and live streaming were not included because the required hardware was unavailable.
- Image decoding was performed using OpenCV (`cv2.imread()`), while preprocessing and inference were GPU accelerated.