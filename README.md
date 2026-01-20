project/
│
├── main.py                    # Entry point
│
├── state.py                   # Toàn bộ biến global & lock
│
├── detector.py                # YOLO + class detect
│
├── video_logic.py             # Load video, reset, direction
│
├── video_worker.py            # Thread xử lý YOLO + đếm
│
└── ui.py                      # Tkinter UI + update_frame