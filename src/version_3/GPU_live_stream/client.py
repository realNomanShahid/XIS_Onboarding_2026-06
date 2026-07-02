import os
import cv2
import time
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor

IMAGE_FOLDER = r"D:\test\images"
COLAB_URL = "https://confess-county-december.ngrok-free.dev/infer"

files = os.listdir(IMAGE_FOLDER)
image_files = sorted([f for f in files if f.lower().strip().endswith(".bmp")])

if len(image_files) == 0:
    print("❌ No images found after filtering")
    exit()

print("✅ Images found:", len(image_files))

session = requests.Session()  # reuse TCP/TLS connection across requests

def send_frame(path):
    frame = cv2.imread(path)
    if frame is None:
        return None, None, None

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return None, None, None

    t0 = time.perf_counter()
    r = session.post(
        COLAB_URL,
        files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
        timeout=30
    )
    r.raise_for_status()
    total_ms = (time.perf_counter() - t0) * 1000

    img_array = np.frombuffer(r.content, np.uint8)
    result_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return result_img, total_ms, path

fps_list = []
all_fps = []
all_latency = []

# Overlap network calls: prefetch next requests while displaying current one
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(send_frame, os.path.join(IMAGE_FOLDER, name)) for name in image_files]

    for idx, future in enumerate(futures):
        try:
            result_img, total_ms, path = future.result()
        except Exception as e:
            print("❌ Request failed:", e)
            continue

        if result_img is None:
            continue

        fps = 1000 / total_ms if total_ms > 0 else 0
        all_fps.append(fps)
        all_latency.append(total_ms)
        fps_list.append(fps)
        fps_list = fps_list[-30:]
        avg_fps = sum(fps_list) / len(fps_list)

        display = result_img.copy()
        cv2.putText(display, f"Frame: {idx+1}/{len(image_files)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"FPS: {avg_fps:.2f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, f"Latency: {total_ms:.1f} ms", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("RT-DETR Remote Segmentation", cv2.resize(display, (540, 540)))
        key = cv2.waitKey(1)
        if key == 27:
            break

print("✅ Done")
# =========================
# SAVE PERFORMANCE METRICS
# =========================

if all_latency:

    avg_fps = sum(all_fps) / len(all_fps)
    avg_latency = sum(all_latency) / len(all_latency)

    min_latency = min(all_latency)
    max_latency = max(all_latency)

    report = f"""==========================
Remote TensorRT Inference
==========================

Image Size      : 384 x 384

Total Frames    : {len(all_latency)}

Average FPS     : {avg_fps:.2f}

Average Latency : {avg_latency:.2f} ms

Min Latency     : {min_latency:.2f} ms

Max Latency     : {max_latency:.2f} ms

Inference Engine: TensorRT

Hardware        : Google Colab T4 GPU

Communication   : FastAPI + ngrok

Client          : Windows Laptop
"""

    with open("performance_metrics.txt", "w") as f:
        f.write(report)

    print(report)
    print("✅ Saved performance_metrics.txt")
cv2.destroyAllWindows()
