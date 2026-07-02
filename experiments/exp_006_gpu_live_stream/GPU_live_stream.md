# Day 8 - Remote TensorRT Segmentation Pipeline

## Objective

Run TensorRT segmentation on a GPU (Google Colab) and send results back to the local machine.

---

## Pipeline

1. Read image from local folder.
2. Encode image as JPEG.
3. Send image to FastAPI server using HTTP.
4. FastAPI receives the image.
5. TensorRT performs segmentation.
6. Draw masks and bounding boxes.
7. Encode result as JPEG.
8. Send processed image back to the local machine.
9. Display output using OpenCV.

---

## Project Structure

```
Client (Windows)
│
├── runner.py
├── performance_metrics.txt
├── main_logs.txt
├── issues_and_fixes.txt
└── images/
```

```
Server (Google Colab)
│
├── FastAPI
├── TensorRT Engine
├── model.engine
└── inference.ipynb
```

---

## Output

Output window:

- Segmentation masks
- Bounding boxes
- Class labels
- FPS
- Latency

Generated files:

- performance_metrics.txt
- main_logs.txt
- issues_and_fixes.txt

---

## Performance

Example:

```
Average FPS     : 0.58

Average Latency : 1913.46 ms

Min Latency     : 1103.46 ms

Max Latency     : 4877.83 ms
```

Note:

The above values are **end-to-end pipeline** performance. They include image transfer, network latency, FastAPI processing, TensorRT inference, post-processing, and returning the result.

Pure TensorRT inference benchmark:

```
Latency : 28.10 ms

FPS     : 35.58
```

This benchmark measures only TensorRT execution on the GPU.

---

## Why GenTL and Harvester were not used

- No physical industrial camera was available.
- The project requirement was to validate the remote inference pipeline.
- Images were read from a local folder instead of a camera.

---

## Why OpenCV was used

- Easy to test the complete pipeline without camera hardware.
- Same inference pipeline can later receive frames from Harvester without changing the TensorRT server.

---

## Future Work

- Replace image folder with Harvester camera stream.
- Connect a real Basler industrial camera.
- Remove ngrok and run the server on a local GPU.
- Measure real-time industrial camera performance.