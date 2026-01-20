import cv2
import time
import threading
import queue
import numpy as np
from model_handler import model, ids_to_detect, all_names, name_detect

# Khởi tạo Queue toàn cục
frame_queue = queue.Queue(maxsize=1)
actual_fps_value = 0.0
fps_lock = threading.Lock()
count_lock = threading.Lock()

# Bộ đếm và bộ nhớ
counts_by_class = {name: 0 for name in name_detect}
counted_ids = set()
track_history = {}

# Biến kiểm soát
cap = None
original_width = 0
original_height = 0
COUNTING_LINE_Y = 0
COUNT_DIRECTION = 'up'
LINE_POSITION_RATIO = 4/5
is_running = threading.Event()

def reset_counters_logic():
    global counts_by_class, counted_ids, track_history, actual_fps_value
    with count_lock:
        counts_by_class = {name: 0 for name in name_detect}
        counted_ids = set()
        track_history = {}
    with fps_lock:
        actual_fps_value = 0.0

def video_processing_loop():
    global counts_by_class, counted_ids, track_history, cap, actual_fps_value
    processing_prev_time = time.time()

    while True:
        is_running.wait()
        if cap is None or not cap.isOpened():
            time.sleep(0.5)
            continue

        current_time = time.time()
        if (current_time - processing_prev_time) > 0:
            fps = 1.0 / (current_time - processing_prev_time)
            with fps_lock:
                actual_fps_value = fps
        processing_prev_time = current_time

        returnFlag, frame = cap.read()
        if not returnFlag:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            reset_counters_logic()
            time.sleep(0.1)
            continue

        results_generator = model.track(
            frame, classes=ids_to_detect, persist=True,
            device='cuda', imgsz=640, half=True, verbose=False
        )

        results = results_generator[0]
        frame_to_display = results.orig_img.copy()
        cv2.line(frame_to_display, (0, COUNTING_LINE_Y), (original_width, COUNTING_LINE_Y), (0, 255, 0), 2)

        temp_counts = counts_by_class.copy()

        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            track_ids = results.boxes.id.cpu().numpy().astype(int)

            for box, cls_id, track_id in zip(boxes, class_ids, track_ids):
                x1, y1, x2, y2 = box
                class_name = all_names[cls_id]
                label = f"ID:{track_id} {class_name}"
                cv2.rectangle(frame_to_display, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame_to_display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                current_center_y = (y1 + y2) // 2
                if track_id in track_history:
                    previous_y = track_history[track_id]
                    is_counted = False
                    if COUNT_DIRECTION == 'down':
                        if previous_y < COUNTING_LINE_Y and current_center_y >= COUNTING_LINE_Y:
                            is_counted = True
                    elif COUNT_DIRECTION == 'up':
                        if previous_y > COUNTING_LINE_Y and current_center_y <= COUNTING_LINE_Y:
                            is_counted = True

                    if is_counted and track_id not in counted_ids:
                        if class_name in temp_counts:
                            temp_counts[class_name] += 1
                        counted_ids.add(track_id)
                track_history[track_id] = current_center_y

        with count_lock:
            counts_by_class.update(temp_counts)

        if not frame_queue.full():
            try: frame_queue.get_nowait()
            except queue.Empty: pass
            frame_queue.put(frame_to_display)