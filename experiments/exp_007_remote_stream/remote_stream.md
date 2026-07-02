# Day 9 — IP Camera Streaming into GPU Inference Pipeline

## Task

Replace the industrial camera feed with a consumer IP camera (phone streaming
over RTSP), add automatic reconnection handling so the pipeline survives
stream drops, and benchmark performance against the industrial camera setup
from Day 8.

## What Was Built

**Client (Windows laptop)**
- Connects to a phone-based IP camera stream (`IP Camera Viewer` app) via
  `cv2.VideoCapture`, forced to TCP transport for reliability over Wi-Fi.
- Sends each captured frame to the remote GPU inference server (TensorRT
  model running on Colab, exposed via ngrok) over HTTP.
- Displays the annotated result (bounding boxes + per-class colored masks)
  live in an OpenCV window.

**Reconnection handling**
- Tracks consecutive failed frame reads; after 30 failures in a row, the
  stream is released and reopened.
- Also reconnects immediately if `cap.isOpened()` reports the stream is down.
- If no successful frame arrives for 30 seconds straight, the pipeline treats
  the stream as dead and exits (rather than retrying forever).

**Performance logging**
- Every frame's round-trip time, GPU-only inference time (from the server's
  `X-Latency-ms` header), and network overhead (round-trip minus GPU time)
  are recorded.
- On exit (ESC, Ctrl+C, or stream timeout), a summary report is written to
  `performance_metrics.txt`: total runtime, reconnect count, failure counts,
  frame throughput, and min/avg/max for latency and FPS.

**Server (Colab, unchanged from Day 8)**
- TensorRT engine served via FastAPI, with a dedicated worker thread owning
  the CUDA context to avoid cross-thread CUDA errors.
- No server-side changes were needed for Day 9 — reconnection is purely a
  client-side concern, since the server only ever sees individual JPEG
  frames over RTSP, not the camera stream itself.

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

The GPU-only inference time (~50-420ms) is consistent with Day 8, confirming
the model and server are unaffected by the camera source. The bottleneck is
entirely network overhead (~2.9s avg, spiking to 11.3s) — this comes from the
IP camera's own HTTP/MJPEG streaming latency plus the ngrok free-tier tunnel,
not from the inference pipeline itself. This matches the expected outcome for
Day 9: **IP cameras are cheaper and easier to deploy, but trade off
significantly higher and less predictable latency compared to a direct
industrial camera + local GPU setup.**

No stream drops or reconnects occurred during this run, so reconnection logic
is implemented and tested for correctness (via induced failures during
development) but wasn't exercised in this particular benchmark session.