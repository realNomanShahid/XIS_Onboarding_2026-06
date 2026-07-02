# Day 9 — IP Camera Streaming into GPU Inference Pipeline

## Task

Replace the industrial camera feed with a consumer IP camera (phone streaming
over RTSP), add automatic reconnection handling so the pipeline survives
stream drops, and benchmark performance against the industrial camera setup
from Day 8.

## What Was Built

**Client (Windows laptop)**
- Connects to a phone IP camera stream using cv2.VideoCapture (TCP for stability over Wi-Fi)
- Sends frames to a remote GPU inference server (TensorRT on Colab via ngrok) over RTSP
- isplays real-time results with bounding boxes and colored segmentation masks in an OpenCV window

**Reconnection handling**
- Tracks consecutive frame read failures and reconnects after 30 failed reads
- Reopens stream if cap.isOpened() returns false
- Reopens stream if cap.isOpened() returns false


## Results

| Metric | Value |
|---|---|
| Total runtime | 185.2 s |
| Frames processed | 58 |
| Reconnects triggered | 0 |
| Frame-read failures | 0 |
| Failed HTTP requests | 0 |
| GPU-only latency (avg) | 200.6 ms |
| Network overhead (avg) | 2877.7 ms |
| Round-trip latency (avg) | 3078.3 ms |
| Average FPS | 0.40 |

## Comparison vs Industrial Camera (Day 8)

- GPU inference time is consistent at ~50–420 ms, same as Day 8
- This confirms the model and server are working normally
- The camera source is not affecting inference performance
- Main delay comes from network overhead (~2.9 s average, up to 11.3 s spikes)
- Latency is caused by:
IP camera HTTP/MJPEG streaming delay
ngrok free-tier tunnel limitations
The inference pipeline itself is not the bottleneck
- Conclusion (Day 9):
- IP cameras are cheaper and easier to deploy
- But they have much higher and less stable latency
- Compared to a direct industrial camera + local GPU setup
- System stability:
- No stream drops or reconnects occurred during testing
- Reconnection logic is implemented and verified (tested earlier via forced failures)
- It was not triggered during this benchmark run