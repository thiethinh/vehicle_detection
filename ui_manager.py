import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import video_processor as vp
from model_handler import name_detect


class VehicleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Detection System")
        self.root.geometry("1400x800")
        self.setup_ui()
        self.set_line_position(vp.LINE_POSITION_RATIO)

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.sidebar = ttk.Frame(main_frame, width=300)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.sidebar.pack_propagate(False)

        self.video_frame = ttk.Frame(main_frame)
        self.video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fps_var = tk.StringVar(value="FPS: --")
        self.total_count_var = tk.StringVar(value="Tổng cộng: 0")
        self.count_vars = {name: tk.StringVar(value=f"{name.capitalize()}: 0") for name in name_detect}

        ttk.Label(self.sidebar, text="Bảng Điều Khiển", font=("Helvetica", 18, "bold")).pack(pady=10)
        ttk.Button(self.sidebar, text="Chọn Video", command=self.open_file_dialog).pack(fill='x', padx=20, pady=5)
        self.direction_btn = ttk.Button(self.sidebar, text="Đang khởi tạo", command=self.toggle_direction)
        self.direction_btn.pack(fill='x', padx=20, pady=10)

        ttk.Label(self.sidebar, textvariable=self.fps_var, font=("Helvetica", 14, "bold"), foreground="green").pack(
            padx=20, pady=10)
        ttk.Label(self.sidebar, textvariable=self.total_count_var, font=("Helvetica", 16, "bold")).pack(padx=20,
                                                                                                        pady=10)

        for name in name_detect:
            ttk.Label(self.sidebar, textvariable=self.count_vars[name], font=("Helvetica", 12)).pack(anchor="w",
                                                                                                     padx=20, pady=5)

        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

    def set_line_position(self, ratio):
        vp.LINE_POSITION_RATIO = ratio
        if vp.original_height > 0:
            vp.COUNTING_LINE_Y = int(vp.original_height * ratio)
        vp.COUNT_DIRECTION = 'down' if ratio == 1 / 3 else 'up'
        text = "TỪ TRÊN XUỐNG (Vạch 1/3)" if vp.COUNT_DIRECTION == 'down' else "TỪ DƯỚI LÊN (Vạch 4/5)"
        self.direction_btn.config(text=f"Chiều đếm: {text}")

    def open_file_dialog(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if file_path:
            if vp.cap is not None: vp.cap.release()
            vp.cap = cv2.VideoCapture(file_path)
            vp.original_width = int(vp.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vp.original_height = int(vp.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.set_line_position(vp.LINE_POSITION_RATIO)
            vp.reset_counters_logic()
            vp.is_running.set()

    def toggle_direction(self):
        new_ratio = 1 / 3 if vp.LINE_POSITION_RATIO == 4 / 5 else 4 / 5
        self.set_line_position(new_ratio)
        vp.reset_counters_logic()

    def update_frame(self):
        with vp.fps_lock:
            self.fps_var.set(f"FPS: {vp.actual_fps_value:.1f}" if vp.actual_fps_value > 0 else "FPS: --")

        try:
            frame = vp.frame_queue.get_nowait()
            view_w, view_h = self.video_frame.winfo_width(), self.video_frame.winfo_height()
            h, w, _ = frame.shape
            scale = min(view_w / w, view_h / h)
            if scale > 0.01:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.video_label.config(image=img)
            self.video_label.image = img
        except:
            pass

        with vp.count_lock:
            self.total_count_var.set(f"Tổng cộng: {sum(vp.counts_by_class.values())}")
            for name, var in self.count_vars.items():
                var.set(f"{name.capitalize()}: {vp.counts_by_class[name]}")

        self.root.after(1, self.update_frame)