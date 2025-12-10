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


def get_head_position(box):
    x1, y1, x2, y2 = box.xyxy[0]
    box_height = y2 - y1
    head_x = (x1 + x2) / 2.0
    head_y = y1 + box_height * 0.1
    return int(head_x), int(head_y)


def move_mouse_smoothly(mouse, target_x, target_y, center_x, center_y, max_movement=100):
    offset_x = target_x - center_x
    offset_y = target_y - center_y
    
    distance = math.sqrt(offset_x * offset_x + offset_y * offset_y)
    
    if distance < 4:
        return 0, 0
    
    if distance <= max_movement * 3:
        max_movement = distance / 1.75
    
    if distance <= max_movement * 2:
        max_movement = distance / 2.5
    
    angle = math.atan2(offset_y, offset_x)
    move_x = math.cos(angle) * min(distance, max_movement)
    move_y = math.sin(angle) * min(distance, max_movement)
    
    mouse.move(move_x, move_y)
    return move_x, move_y


def filter_boxes_by_class(boxes, target_class):
    if target_class == 2:
        return boxes
    
    filtered = [b for b in boxes if int(b.cls[0]) == target_class]
    return filtered


def sort_boxes_by_predicted_position(boxes, last_head_x, last_head_y, last_move_x, last_move_y, center_x, center_y):
    predicted_x = last_head_x - last_move_x
    predicted_y = last_head_y - last_move_y
    
    def dist_sq(b):
        try:
            x1, y1, x2, y2 = b.xyxy[0]
        except Exception:
            return float('inf')
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dx = cx - predicted_x
        dy = cy - predicted_y
        return dx * dx + dy * dy
    
    return sorted(boxes, key=dist_sq)

def main():
    model_path = "best.pt"
    model = YOLO(model_path)
    mouse = Controller()
    target_class = get_target_class()

    center_x = SCREEN_WIDTH // 2
    center_y = SCREEN_HEIGHT // 2
    
    model_name = os.path.splitext(model_path)[0]
    log_file = f"{model_name}_fps_log.txt"

    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\nTracking '", end="")
    if target_class == 2:
        print("Everyone", end="")
    else:
        print(CLASS_NAMES[target_class], end="")
    print("'. Press Ctrl+C to stop.\n")

    frame_count = 0
    fps_time = time.time()
    aim_frame_counter = 0
    last_head_x = center_x
    last_head_y = center_y
    last_move_x = 0
    last_move_y = 0
    frames_since_detection = 0
    prev_target_x = center_x
    prev_target_y = center_y
    mouse_x = 960
    mouse_y = 540
    
    fps_history = []
    fps_window_time = time.time()

    try:
        while True:
            frame = capture_screen()
            results = model(frame, name=f"{time.time()}.jpg", verbose=False, save=False)

            frame_count += 1
            aim_frame_counter += 1
            frames_since_detection += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                print(f"FPS: {fps:.1f}", end="\r")
                fps_history.append(fps)
                frame_count = 0
                fps_time = time.time()

            if aim_frame_counter % 2 != 0:
                continue

            for result in results:
                boxes = result.boxes.cpu().numpy()
                boxes = filter_boxes_by_class(boxes, target_class)
                
                if len(boxes) > 0:
                    # if frames_since_detection < 5:
                    #     boxes = sort_boxes_by_predicted_position(boxes, last_head_x, last_head_y, last_move_x, last_move_y, center_x, center_y)
                    # else:
                    #     boxes = sort_boxes_by_center_distance(boxes, center_x, center_y)
                    boxes = sort_boxes_by_center_distance(boxes, center_x, center_y)
                    closest_box = boxes[0]
                    head_x, head_y = get_head_position(closest_box)
                    
                    target_shift_x = head_x - prev_target_x - last_move_x
                    target_shift_y = head_y - prev_target_y - last_move_y
                    target_shift = math.sqrt(target_shift_x * target_shift_x + target_shift_y * target_shift_y)
                    
                    if target_shift > 40:
                        move_x, move_y = move_mouse_smoothly(mouse, head_x, head_y, center_x, center_y, max_movement=50)
                        prev_target_x = head_x
                        prev_target_y = head_y
                    else:
                        move_x, move_y = move_mouse_smoothly(mouse, prev_target_x, prev_target_y, center_x, center_y, max_movement=50)
                    
                    last_move_x = move_x
                    last_move_y = move_y
                    last_head_x = head_x
                    last_head_y = head_y
                    frames_since_detection = 0


    except KeyboardInterrupt:
        with open(log_file, "a") as f:
            str_fps_history = ", ".join([f"{fps}" for fps in fps_history])
            f.write(f"{datetime.now()}: {str_fps_history}\n")


if __name__ == "__main__":
    main()
