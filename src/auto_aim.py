import mss
import cv2
import numpy as np
from ultralytics import YOLO
from pynput.mouse import Controller, Button
import time
import os
from datetime import datetime
import math

CLASS_NAMES = {0: "CT", 1: "T"}
SAVE_DIR = "saved_frames"
MAX_SAVED_IMAGES = 10
FILE_EXTS = ('.png', '.jpg', '.jpeg')
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

def get_target_class():
    print("\nAvailable classes:")
    print(f"0: {CLASS_NAMES[0]}")
    print(f"1: {CLASS_NAMES[1]}")
    print("2: Everyone (both CT and T)")

    while True:
        try:
            choice = int(input("\nSelect class to track (0, 1, or 2): "))
            if choice in [0, 1, 2]:
                return choice
            print("Invalid choice. Please enter 0, 1, or 2.")
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

def main():
    model = YOLO("best.pt")
    mouse = Controller()
    target_class = get_target_class()

    center_x = SCREEN_WIDTH // 2
    center_y = SCREEN_HEIGHT // 2

    # Ensure save directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\nTracking '", end="")
    if target_class == 2:
        print("Everyone", end="")
    else:
        print(CLASS_NAMES[target_class], end="")
    print("'. Press Ctrl+C to stop.\n")

    frame_count = 0
    fps_time = time.time()

    try:
        while True:
            frame = capture_screen()
            results = model(frame, name=f"{time.time()}.jpg", verbose=False, save=False)

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

                    # Check if target matches based on selection
                    if target_class == 2:
                        # Track everyone (both CT and T)
                        target_matches = conf > 0.6
                    else:
                        # Track specific class
                        target_matches = (cls == target_class and conf > 0.6)
                    
                    if target_matches:
                        x1, y1, x2, y2 = box.xyxy[0]
                        # Target upper part of the box (head area) instead of center
                        target_x = int((x1 + x2) / 2)  # Center X
                        target_y = int(y1 + (y2 - y1) * 0.15)  # Upper 0.15% of box height
                        # Calculate relative movement from screen center
                        dx = target_x - center_x
                        dy = target_y - center_y
                        # Calculate distance to target
                        distance = (dx**2 + dy**2)**0.5
                        
                        # Calculate angle towards target
                        angle = math.atan2(dy, dx)
                        
                        # Move 100px in the direction of the angle
                        move_distance = min(100, distance)
                        move_dx = int(move_distance * math.cos(angle))
                        move_dy = int(move_distance * math.sin(angle))
                        
                        mouse.move(move_dx, move_dy)
                        print(f"Moved mouse towards target (angle: {math.degrees(angle):.1f}°): ({move_dx}, {move_dy})")
                        
                        # Current cursor position is at center + movement offset
                        cursor_x = center_x + move_dx
                        cursor_y = center_y + move_dy
                        
                        # Check if cursor is close enough to target
                        dist_to_target = ((cursor_x - target_x)**2 + (cursor_y - target_y)**2)**0.5
                        print(f"Current position: ({cursor_x}, {cursor_y}), Target: ({target_x}, {target_y}), Distance: {dist_to_target:.1f}px")
                        
                        # if dist_to_target < 10:
                        #     time.sleep(0.2)  # Small delay before shooting
                        #     mouse.click(Button.left, 1)
                        #     print(f"SHOT! Distance to target: {dist_to_target:.1f}px")
                        # break

            #time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
