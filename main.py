import tkinter as tk
import threading
import cv2
import video_processor as vp
from ui_manager import VehicleApp

if __name__ == "__main__":
    root = tk.Tk()
    app = VehicleApp(root)

    # Khởi chạy luồng xử lý video
    video_thread = threading.Thread(target=vp.video_processing_loop, daemon=True)
    video_thread.start()

    # Bắt đầu vòng lặp cập nhật giao diện
    root.after(100, app.update_frame)

    try:
        root.mainloop()
    finally:
        print("Đóng ứng dụng")
        vp.is_running.clear()
        if vp.cap is not None:
            vp.cap.release()
        cv2.destroyAllWindows()