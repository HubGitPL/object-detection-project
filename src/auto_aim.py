import mss
import cv2
import numpy as np
from ultralytics import YOLO
from pynput.mouse import Controller
import time
import os
from datetime import datetime

CLASS_NAMES = {0: "Counter terrorist", 1: "T"}
SAVE_DIR = "saved_frames"
MAX_SAVED_IMAGES = 10
FILE_EXTS = ('.png', '.jpg', '.jpeg')
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

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


def sort_boxes_by_center_distance(boxes, center_x, center_y):
    """Return boxes sorted by their center distance to (center_x, center_y).

    Uses squared distance to avoid unnecessary sqrt computation.
    """
    def dist_sq(b):
        try:
            x1, y1, x2, y2 = b.xyxy[0]
        except Exception:
            return float('inf')
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dx = cx - center_x
        dy = cy - center_y
        return dx * dx + dy * dy

    return sorted(boxes, key=dist_sq)


def count_saved_images(dir_path: str) -> int:
    try:
        return sum(1 for f in os.listdir(dir_path) if f.lower().endswith(FILE_EXTS))
    except FileNotFoundError:
        return 0


def main():
    model = YOLO("best.pt")
    mouse = Controller()
    target_class = get_target_class()

    center_x = SCREEN_WIDTH // 2
    center_y = SCREEN_HEIGHT // 2

    # Ensure save directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\nTracking '{CLASS_NAMES[target_class]}'. Press Ctrl+C to stop.\n")

    frame_count = 0
    fps_time = time.time()

    try:
        while True:
            frame = capture_screen()
            results = model(frame, name=f"{time.time()}.jpg", verbose=True, save=False)

            frame_count += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                print(f"FPS: {fps:.1f}", end="\r")
                frame_count = 0
                fps_time = time.time()

            for result in results:
                boxes = result.boxes.cpu().numpy()
                # Sort boxes by distance from screen center (closest first)
                boxes = sort_boxes_by_center_distance(boxes, center_x, center_y)
                
                for box in boxes:
                    
                    cls = int(box.cls[0])
                    conf = box.conf[0]

                    if cls == target_class and conf > 0.6:
                        print(
                            f"Detected {CLASS_NAMES[cls]} with confidence {conf:.2f}"
                        )

                        # Save full frame to disk with timestamp and confidence
                        current_count = count_saved_images(SAVE_DIR)
                        if current_count >= MAX_SAVED_IMAGES:
                            print(f"Save limit reached ({current_count}/{MAX_SAVED_IMAGES}), skipping save.")
                        else:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            safe_name = CLASS_NAMES[cls].replace(' ', '_')
                            filename = f"{safe_name}_{int(conf*100)}_{timestamp}.png"
                            filepath = os.path.join(SAVE_DIR, filename)
                            try:
                                cv2.imwrite(filepath, frame)
                                print(f"Saved frame to: {filepath}")
                            except Exception as e:
                                print(f"Failed to save frame: {e}")
                        x1, y1, x2, y2 = box.xyxy[0]
                        cx, cy = get_center_point([x1, y1, x2, y2])
                        # Calculate relative movement from screen center
                        dx = int(cx) - center_x
                        dy = int(cy) - center_y
                        # Calculate distance to target
                        distance = (dx**2 + dy**2)**0.5
                        
                        if distance < 300:
                            # If closer than 200px, move exactly to target
                            mouse.move(dx, dy)
                            print(f"Moved mouse to target: ({dx}, {dy})")
                        else:
                            # Move 200px in direction of target
                            ratio = 300 / distance
                            move_dx = int(dx * ratio)
                            move_dy = int(dy * ratio)
                            mouse.move(move_dx, move_dy)
                            print(f"Moved mouse 200px towards target: ({move_dx}, {move_dy})")
                        break

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
