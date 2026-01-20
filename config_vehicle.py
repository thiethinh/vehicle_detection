import threading
import queue

# Cấu hình Model và Classes
MODEL_PATH = '11s_retrain.pt'
NAME_DETECT = ['bus', 'car', 'motorbike', 'truck']

# Chia sẻ tài nguyên giữa các luồng
frame_queue = queue.Queue(maxsize=1)
count_lock = threading.Lock()
fps_lock = threading.Lock()
is_running = threading.Event()

# Các biến trạng thái đếm (Shared)
counts_by_class = {name: 0 for name in NAME_DETECT}
counted_ids = set()
track_history = {}
actual_fps_value = 0.0

# Thông số video
video_info = {
    'cap': None,
    'original_width': 0,
    'original_height': 0,
    'counting_line_y': 0,
    'direction': 'up',
    'ratio': 4/5
}