import cv2
import requests
import time
import numpy as np
import os
from datetime import datetime

RTSP_URL = "http://10.83.59.139:8080/video"  # your IP camera stream URL
COLAB_URL = "https://confess-county-december.ngrok-free.dev/infer"

RECONNECT_DELAY = 2.0        # seconds to wait before retrying
MAX_CONSECUTIVE_FAILS = 30   # give up reading and force a reconnect after this many bad reads
STREAM_TIMEOUT = 30.0        # seconds with no successful frame before we call it "stream ended"

METRICS_FILE = "performance_metrics.txt"

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

session = requests.Session()


def open_stream():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# =========================
# METRICS TRACKING
# =========================

round_trip_list = []
gpu_only_list = []
network_overhead_list = []
fps_list = []

reconnect_count = 0
frame_read_fail_count_total = 0
request_fail_count = 0
frames_processed = 0

session_start = time.perf_counter()
last_successful_frame_time = time.perf_counter()
stream_ended_reason = "manual exit (ESC)"


def write_metrics_report():
    total_runtime = time.perf_counter() - session_start

    with open(METRICS_FILE, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("IP Camera -> Remote GPU Inference: Performance Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"RTSP/Stream URL: {RTSP_URL}\n")
        f.write(f"Inference endpoint: {COLAB_URL}\n\n")

        f.write(f"Total runtime: {total_runtime:.2f} s\n")
        f.write(f"Stream end reason: {stream_ended_reason}\n\n")

        f.write("--- Reconnection Handling ---\n")
        f.write(f"Total reconnects triggered: {reconnect_count}\n")
        f.write(f"Total frame-read failures (before reconnect threshold): {frame_read_fail_count_total}\n")
        f.write(f"Total failed HTTP requests to GPU server: {request_fail_count}\n\n")

        f.write("--- Frames ---\n")
        f.write(f"Frames successfully processed: {frames_processed}\n")
        if total_runtime > 0:
            f.write(f"Overall throughput: {frames_processed / total_runtime:.2f} frames/sec (session avg)\n")
        f.write("\n")

        if round_trip_list:
            f.write("--- Latency (ms) ---\n")
            f.write(f"round_trip_ms   : avg={np.mean(round_trip_list):.1f}  "
                    f"min={np.min(round_trip_list):.1f}  max={np.max(round_trip_list):.1f}\n")
            f.write(f"gpu_only_ms     : avg={np.mean(gpu_only_list):.1f}  "
                    f"min={np.min(gpu_only_list):.1f}  max={np.max(gpu_only_list):.1f}\n")
            f.write(f"network_overhead_ms: avg={np.mean(network_overhead_list):.1f}  "
                    f"min={np.min(network_overhead_list):.1f}  max={np.max(network_overhead_list):.1f}\n\n")

            f.write("--- FPS ---\n")
            f.write(f"avg_fps (per-request): avg={np.mean(fps_list):.2f}  "
                    f"min={np.min(fps_list):.2f}  max={np.max(fps_list):.2f}\n")
        else:
            f.write("No successful frames were processed — no latency/FPS stats available.\n")

    print(f"\n✅ Saved {METRICS_FILE}")


try:
    cap = open_stream()
    fail_count = 0

    while True:
        # Detect a truly dead stream (no good frame in STREAM_TIMEOUT seconds)
        if time.perf_counter() - last_successful_frame_time > STREAM_TIMEOUT:
            stream_ended_reason = f"stream timeout: no frame for {STREAM_TIMEOUT:.0f}s"
            print(f"⏱️ {stream_ended_reason}")
            break

        if not cap.isOpened():
            print("⚠️ Stream not open, reconnecting...")
            reconnect_count += 1
            cap.release()
            time.sleep(RECONNECT_DELAY)
            cap = open_stream()
            continue

        ret, frame = cap.read()

        if not ret or frame is None:
            fail_count += 1
            frame_read_fail_count_total += 1
            print(f"⚠️ Frame read failed ({fail_count}/{MAX_CONSECUTIVE_FAILS})")
            if fail_count >= MAX_CONSECUTIVE_FAILS:
                print("🔁 Too many failures — reconnecting stream...")
                reconnect_count += 1
                cap.release()
                time.sleep(RECONNECT_DELAY)
                cap = open_stream()
                fail_count = 0
            continue

        fail_count = 0
        last_successful_frame_time = time.perf_counter()

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue

        t0 = time.perf_counter()
        try:
            resp = session.post(
                COLAB_URL,
                files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                timeout=5,
            )
            resp.raise_for_status()
        except Exception as e:
            request_fail_count += 1
            print("Request failed:", e)
            continue

        round_trip_ms = (time.perf_counter() - t0) * 1000
        server_latency_ms = float(resp.headers.get("X-Latency-ms", 0))
        network_overhead_ms = round_trip_ms - server_latency_ms

        fps = 1000 / round_trip_ms if round_trip_ms > 0 else 0
        fps_list_window = fps_list[-30:] if len(fps_list) >= 1 else []
        avg_fps = (sum(fps_list_window) + fps) / (len(fps_list_window) + 1) if fps_list_window else fps

        # record for final report (full history, not just rolling window)
        round_trip_list.append(round_trip_ms)
        gpu_only_list.append(server_latency_ms)
        network_overhead_list.append(network_overhead_ms)
        fps_list.append(fps)
        frames_processed += 1

        print(
            f"round_trip={round_trip_ms:.1f}ms  "
            f"gpu_only={server_latency_ms:.1f}ms  "
            f"network_overhead={network_overhead_ms:.1f}ms  "
            f"avg_fps={avg_fps:.2f}"
        )

        img_array = np.frombuffer(resp.content, np.uint8)
        result_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if result_img is None:
            continue

        cv2.putText(result_img, f"FPS: {avg_fps:.2f} (GPU-only: {server_latency_ms:.0f}ms)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Remote GPU Inference Test", cv2.resize(result_img, (420, 420)))

        if cv2.waitKey(1) & 0xFF == 27:
            stream_ended_reason = "manual exit (ESC)"
            break

except KeyboardInterrupt:
    stream_ended_reason = "manual exit (Ctrl+C)"

finally:
    cap.release()
    cv2.destroyAllWindows()
    write_metrics_report()