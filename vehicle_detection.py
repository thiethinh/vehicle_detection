import cv2
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import time
from ultralytics import YOLO
import numpy as np
import threading
import queue

# Khởi tạo Queue toàn cục để trao đổi khung hình đã xử lý
frame_queue = queue.Queue(maxsize=1)

# Biến FPS thực tế và Lock
actual_fps_value = 0.0
fps_lock = threading.Lock()

# classes to detect
name_detect = ['bus', 'car', 'motorbike', 'truck']

model = YOLO('11s_retrain.pt')
all_names = model.names

# 3. Lấy ID của các lớp đó
ids_to_detect = [id for id, name in all_names.items() if name in name_detect]
print(f'Các ID phương tiện sẽ được đếm: {ids_to_detect}')

# 4. Tạo BỘ ĐẾM và BỘ NHỚ (Dùng Lock để đồng bộ hóa truy cập)
counts_by_class = {name: 0 for name in name_detect}
counted_ids = set()
track_history = {}
count_lock = threading.Lock()  # Lock để bảo vệ các biến đếm/nhớ

# --- Biến Kiểm Soát Video và Đếm ---
cap = None
original_width = 0
original_height = 0
COUNTING_LINE_Y = 0
COUNT_DIRECTION = 'up'
LINE_POSITION_RATIO = 4/5
is_running = threading.Event()  # Cờ báo hiệu luồng xử lý có được phép chạy hay không


# --- Các Hàm Quản lý Video và Đếm ---

def reset_counters():
    """Thiết lập lại tất cả bộ đếm và bộ nhớ."""
    global counts_by_class, counted_ids, track_history, actual_fps_value
    with count_lock:
        counts_by_class = {name: 0 for name in name_detect}
        counted_ids = set()
        track_history = {}

    with fps_lock:
        actual_fps_value = 0.0  # Reset FPS khi video được reset

    # Cập nhật giao diện (sẽ được thực hiện trong luồng chính)
    if 'total_count_var' in globals():
        root.after(0, lambda: total_count_var.set("Tổng cộng: 0"))
        for name in name_detect:
            if name in count_vars:
                root.after(0, lambda n=name: count_vars[n].set(f"{n.capitalize()}: 0"))
        root.after(0, lambda: fps_var.set("FPS: --"))


def update_direction_button_text():
    """Cập nhật chữ trên nút chuyển chiều."""
    if COUNT_DIRECTION == 'down':
        direction_btn.config(text="Chiều đếm: TỪ TRÊN XUỐNG (Vạch 1/3)")
    else:
        direction_btn.config(text="Chiều đếm: TỪ DƯỚI LÊN (Vạch 4/5)")


def set_line_position(ratio):
    """Thiết lập vị trí vạch đếm và chiều đếm tương ứng."""
    global COUNTING_LINE_Y, LINE_POSITION_RATIO, COUNT_DIRECTION
    LINE_POSITION_RATIO = ratio

    if original_height > 0:
        COUNTING_LINE_Y = int(original_height * LINE_POSITION_RATIO)

    if ratio == 1 / 3:
        COUNT_DIRECTION = 'down'
    elif ratio == 4/5:
        COUNT_DIRECTION = 'up'

    if 'direction_btn' in globals():
        update_direction_button_text()


def load_video(video_path):
    """Mở video mới và khởi tạo lại các biến."""
    global cap, original_width, original_height

    if cap is not None:
        cap.release()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video tại đường dẫn: {video_path}")
        cap = None
        is_running.clear()
        return False

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    set_line_position(LINE_POSITION_RATIO)
    reset_counters()

    # Báo hiệu cho luồng xử lý rằng video đã sẵn sàng
    is_running.set()

    print(f"Đã tải video: {video_path}")
    return True


def open_file_dialog():
    """Mở hộp thoại chọn file và gọi load_video."""
    file_path = filedialog.askopenfilename(
        defaultextension=".mp4",
        filetypes=[("Video files", "*.mp4;*.avi;*.mov"), ("All files", "*.*")]
    )
    if file_path:
        load_video(file_path)


def toggle_direction():
    """Đổi vị trí vạch đếm và chiều đếm."""
    if LINE_POSITION_RATIO == 4 / 5:
        set_line_position(1 / 3)
    else:
        set_line_position(4 / 5)
    reset_counters()


# --- HÀM XỬ LÝ VIDEO TRONG LUỒNG RIÊNG (Worker Thread) ---

def video_processing_loop():
    """Vòng lặp xử lý chính (YOLO và Đếm) chạy trên luồng phụ."""
    global counts_by_class, counted_ids, track_history, cap, actual_fps_value

    # Biến tính FPS nội bộ cho luồng xử lý
    processing_prev_time = time.time()

    while True:
        is_running.wait()

        if cap is None or not cap.isOpened():
            time.sleep(0.5)
            continue

        # 1. TÍNH TOÁN FPS THỰC TẾ
        current_time = time.time()
        # Tính toán FPS dựa trên thời gian thực hiện Vòng lặp
        if (current_time - processing_prev_time) > 0:
            fps = 1.0 / (current_time - processing_prev_time)
            # Cập nhật FPS an toàn
            with fps_lock:
                actual_fps_value = fps

        processing_prev_time = current_time  # Cập nhật thời gian bắt đầu vòng lặp tiếp theo

        # 2. Đọc khung hình
        returnFlag, frame = cap.read()
        if not returnFlag:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            reset_counters()
            time.sleep(0.1)
            continue

        # === NƠI CHẠY MODEL YOLO VÀ ĐẾM XE (TÁC VỤ NẶNG) ===
        results_generator = model.track(
            frame,
            classes=ids_to_detect,
            persist=True,
            device='cuda',
            imgsz=640,
            half=True,
            verbose=False
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

                # LOGIC ĐẾM
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
                        cv2.line(frame_to_display, (0, COUNTING_LINE_Y), (original_width, COUNTING_LINE_Y), (0, 0, 255),
                                 3)

                track_history[track_id] = current_center_y

        # CẬP NHẬT BIẾN ĐẾM (Đồng bộ hóa an toàn)
        with count_lock:
            counts_by_class.update(temp_counts)

        # Gửi khung hình đã xử lý sang luồng chính (UI)
        if not frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            frame_queue.put(frame_to_display)

        # Không cần sleep ở đây nếu muốn chạy hết tốc độ (max FPS)


# --- VÒNG LẶP CẬP NHẬT GIAO DIỆN (Luồng chính Tkinter) ---

def update_frame():
    # 1. CẬP NHẬT FPS THỰC TẾ (Đọc an toàn từ luồng phụ)
    current_fps = 0.0
    with fps_lock:
        current_fps = actual_fps_value

    if current_fps > 0:
        fps_var.set(f"FPS: {current_fps:.1f}")
    else:
        # Chỉ hiển thị '--' khi chưa có video hoặc reset
        fps_var.set("FPS: --")

    frame_to_display = None

    # 2. LẤY KHUNG HÌNH TỪ QUEUE (Không chặn luồng chính)
    try:
        frame_to_display = frame_queue.get_nowait()
    except queue.Empty:
        pass

    if frame_to_display is None:
        if cap is None or not cap.isOpened():
            # Hiển thị khung đen chờ khi chưa có video
            view_width = video_frame.winfo_width()
            view_height = video_frame.winfo_height()

            frame_to_display = np.zeros((view_height, view_width, 3), dtype=np.uint8)

            text = "SELECT VIDEO AT SIDEBAR"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (view_width - text_size[0]) // 2
            text_y = (view_height + text_size[1]) // 2
            cv2.putText(frame_to_display, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Không cần làm gì nếu luồng phụ chưa gửi khung hình mới
            pass

    # Chỉ xử lý vẽ nếu có khung hình
    if frame_to_display is not None:

        # 3. CONVERT KHUNG HÌNH (CV2 -> TKINTER)
        view_width = video_frame.winfo_width()
        view_height = video_frame.winfo_height()

        h, w, _ = frame_to_display.shape
        scale = min(view_width / w, view_height / h)

        if scale > 0.01:
            new_width = int(w * scale)
            new_height = int(h * scale)
            frame_resized = cv2.resize(frame_to_display, (new_width, new_height))
        else:
            frame_resized = frame_to_display

        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        tk_image = ImageTk.PhotoImage(image=pil_image)

        # 4. CẬP NHẬT ẢNH LÊN LABEL
        video_label.config(image=tk_image)
        video_label.image = tk_image

    # 5. CẬP NHẬT BIẾN ĐẾM TKINTER (Đọc an toàn)
    with count_lock:
        total_count = sum(counts_by_class.values())
        total_count_var.set(f"Tổng cộng: {total_count}")

        for name, var in count_vars.items():
            count = counts_by_class.get(name, 0)
            var.set(f"{name.capitalize()}: {count}")

    # 6. LẶP LẠI (Tần suất cao để UI phản hồi nhanh)
    root.after(1, update_frame)


# --- PHẦN KHỞI TẠO TKINTER (Giữ nguyên) ---

root = tk.Tk()
root.title("Trình xem Video với Sidebar (YOLOv11 - Multithreading)")
root.geometry("1400x800")

main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

sidebar_frame = ttk.Frame(main_frame, width=300)
sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
sidebar_frame.pack_propagate(False)

video_frame = ttk.Frame(main_frame)
video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

fps_var = tk.StringVar(value="FPS: --")
total_count_var = tk.StringVar(value="Tổng cộng: 0")

count_vars = {}
for name in name_detect:
    count_vars[name] = tk.StringVar(value=f"{name.capitalize()}: 0")

title_label = ttk.Label(sidebar_frame, text="Bảng Điều Khiển", font=("Helvetica", 18, "bold"))
title_label.pack(pady=10)

open_btn = ttk.Button(sidebar_frame, text="📁 Chọn Video", command=open_file_dialog)
open_btn.pack(fill='x', padx=20, pady=(0, 5))

direction_btn = ttk.Button(sidebar_frame, text="🔄 Đang khởi tạo...", command=toggle_direction)
direction_btn.pack(fill='x', padx=20, pady=(5, 10))

separator_controls = ttk.Separator(sidebar_frame, orient='horizontal')
separator_controls.pack(fill='x', padx=20, pady=10)

fps_label = ttk.Label(sidebar_frame, textvariable=fps_var, font=("Helvetica", 14, "bold"), foreground="green")
fps_label.pack(anchor="w", padx=20, pady=10)

separator_count = ttk.Separator(sidebar_frame, orient='horizontal')
separator_count.pack(fill='x', padx=20, pady=10)

total_label = ttk.Label(sidebar_frame, textvariable=total_count_var, font=("Helvetica", 16, "bold"))
total_label.pack(anchor="w", padx=20, pady=10)

for name in name_detect:
    var = count_vars[name]
    label = ttk.Label(sidebar_frame, textvariable=var, font=("Helvetica", 12))
    label.pack(anchor="w", padx=20, pady=5)

video_label = ttk.Label(video_frame)
video_label.pack(fill=tk.BOTH, expand=True)

# --- BẮT ĐẦU ỨNG DỤNG ---
print("Đang khởi tạo giao diện và luồng xử lý...")

set_line_position(LINE_POSITION_RATIO)

video_thread = threading.Thread(target=video_processing_loop, daemon=True)
video_thread.start()

root.after(100, update_frame)

root.mainloop()

# --- DỌN DẸP ---
print("Đóng ứng dụng...")
is_running.clear()
if cap is not None:
    cap.release()
cv2.destroyAllWindows()
