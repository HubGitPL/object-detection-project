import mss
import cv2
import numpy as np
from ultralytics import YOLO
from pynput.mouse import Controller
import time

CLASS_NAMES = {0: "Counter terrorist", 1: "T"}


def get_target_class():
    print("\nAvailable classes:")
    for idx, name in CLASS_NAMES.items():
        print(f"{idx}: {name}")

    while True:
        try:
            choice = int(input("\nSelect class to track (0 or 1): "))
            if choice in CLASS_NAMES:
                return choice
            print("Invalid choice. Please enter 0 or 1.")
        except ValueError:
            print("Please enter a valid number.")


def capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def get_center_point(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def main():
    model = YOLO("yolov12n.pt")
    mouse = Controller()
    target_class = get_target_class()

    print(f"\nTracking '{CLASS_NAMES[target_class]}'. Press Ctrl+C to stop.\n")

    frame_count = 0
    fps_time = time.time()

    try:
        while True:
            frame = capture_screen()
            results = model(frame, verbose=False)

            frame_count += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                print(f"FPS: {fps:.1f}", end="\r")
                frame_count = 0
                fps_time = time.time()

            for result in results:
                boxes = result.boxes.cpu().numpy()
                boxes = sorted(boxes, key=lambda b: b.conf[0], reverse=True)
                
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = box.conf[0]

                    if cls == target_class and conf > 0.3:
                        print(
                            f"Detected {CLASS_NAMES[cls]} with confidence {conf:.2f}"
                        )
                        x1, y1, x2, y2 = box.xyxy[0]
                        cx, cy = get_center_point([x1, y1, x2, y2])

                        mouse.position = (cx, cy)
                        print(f"Moved mouse to: ({cx}, {cy})")
                        break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
