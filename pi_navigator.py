import cv2
import numpy as np
import time
import threading
import pyttsx3
import argparse
import os


# --- LAYERED ENVIRONMENT DETECTORS (for surfaces YOLO cannot classify) ---

def detect_wall(frame_bgr):
    """
    Detects wall-like flat surfaces using texture analysis.
    Walls have low pixel variance (smooth, uniform) and cover large frame areas.
    """
    h, w = frame_bgr.shape[:2]
    strip = frame_bgr[int(h*0.3):int(h*0.9), int(w*0.2):int(w*0.8)]
    gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    if np.mean(gray_strip) < 20:
        return None
    _, std_dev = cv2.meanStdDev(gray_strip)
    global_std = std_dev[0][0]
    cell_h = max(1, strip.shape[0] // 4)
    cell_w = max(1, strip.shape[1] // 4)
    uniform_cells = 0
    for i in range(4):
        for j in range(4):
            cell = gray_strip[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            if cell.size == 0:
                continue
            _, cell_std = cv2.meanStdDev(cell)
            if cell_std[0][0] < 30:
                uniform_cells += 1
    uniform_ratio = uniform_cells / 16.0
    if uniform_ratio > 0.70 and global_std < 35:
        return "VERY_CLOSE"
    elif uniform_ratio > 0.50 and global_std < 50:
        return "CLOSE"
    return None


def detect_vegetation(frame_bgr):
    """
    Detects trees, bushes, and grass using HSV green-range masking.
    """
    h, w = frame_bgr.shape[:2]
    cx1, cx2 = int(w*0.2), int(w*0.8)
    cy1, cy2 = int(h*0.1), int(h*0.9)
    center = frame_bgr[cy1:cy2, cx1:cx2]
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.count_nonzero(mask) / float(mask.size)
    if green_ratio > 0.50:
        return "CLOSE"
    elif green_ratio > 0.20:
        return "MEDIUM"
    return None


def detect_environment(frame_bgr):
    """
    Multi-layered fallback detector. Priority: wall > vegetation.
    Returns (label, dist_level) or None.
    """
    wall_dist = detect_wall(frame_bgr)
    if wall_dist:
        return ("wall", wall_dist)
    veg_dist = detect_vegetation(frame_bgr)
    if veg_dist:
        return ("tree", veg_dist)
    return None

# --- HARDWARE ABSTRACTION LAYER ---
try:
    from picamera2 import Picamera2
    HAS_PICAMERA = True
except ImportError:
    HAS_PICAMERA = False

# --- CONFIGURATION ---
PRIORITIES = {
    "person": 1, "car": 1, "bus": 1, "truck": 1, "bicycle": 1, "motorcycle": 1, "dog": 1, "cat": 1,
    "bench": 2, "chair": 2, "couch": 2, "potted plant": 2, "stairs": 2, "door": 2, "backpack": 2, "umbrella": 2,
    "handbag": 2, "suitcase": 2, "dining table": 2, "bed": 2, "toilet": 2, "tv": 2, "laptop": 2, "mouse": 2,
    "remote": 2, "keyboard": 2, "cell phone": 2, "microwave": 2, "oven": 2, "toaster": 2, "sink": 2, 
    "refrigerator": 2, "book": 2, "clock": 2, "vase": 2, "scissors": 2, "teddy bear": 2, "hair drier": 2, "toothbrush": 2
}

class NavigatorPi:
    def __init__(self, frequency=1.0, display=False, pc_test=False):
        self.frequency = frequency
        self.display = display
        self.pc_test = pc_test or (not HAS_PICAMERA)
        
        # Initialize TTS Engine
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 160)
            self.has_tts = True
        except Exception as e:
            print(f"[WARN] TTS failed to initialize: {e}. Falling back to terminal output.")
            self.has_tts = False
        
        # Load Model (Prioritize OpenVINO for RPi 5, otherwise standard YOLOv8n)
        from ultralytics import YOLO
        ov_model = "yolov8n_openvino_model"
        if os.path.exists(ov_model) and not self.pc_test:
            print(f"[INFO] Using ACCELERATED OpenVINO engine: {ov_model}")
            self.model = YOLO(ov_model, task="detect")
        else:
            engine_name = "STANDARD YOLOv8-nano" if not self.pc_test else "PC EMULATION"
            print(f"[INFO] Using {engine_name} engine (yolov8n.pt)...")
            self.model = YOLO("yolov8n.pt")
        
        # Initialize Camera
        if self.pc_test:
            print("[INFO] Initializing PC Webcam (cv2.VideoCapture)...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise ConnectionError("Could not open PC Webcam. Please check connection.")
        else:
            print("[INFO] Initializing Picamera2...")
            self.picam2 = Picamera2()
            self.picam2.configure(self.picam2.create_video_configuration(main={"size": (640, 480)}))
            self.picam2.start()
        
        self.last_speech_time = 0
        self.lock = threading.Lock()
        self.running = True

    def speak(self, text):
        if not self.has_tts:
            print(f"[VOICE SIM] {text}")
            return
            
        def _say():
            with self.lock:
                self.engine.say(text)
                self.engine.runAndWait()
        threading.Thread(target=_say).start()

    def run(self):
        mode_str = "PC EMULATION" if self.pc_test else "HARDWARE"
        print(f"[INFO] Starting Navigator Loop | Mode: {mode_str} | Interval: {self.frequency}s")
        try:
            while self.running:
                # Capture frame
                if self.pc_test:
                    ret, frame = self.cap.read()
                    if not ret: break
                else:
                    frame = self.picam2.capture_array()
                
                h, w = frame.shape[:2]
                current_time = time.time()
                
                # YOLO Inference
                results = self.model(frame, verbose=False, conf=0.45)
                objs = []
                
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cls = int(box.cls[0])
                        label = self.model.names[cls]
                        area = ((x2 - x1) * (y2 - y1)) / (w * h)
                        cx = (x1 + x2) / 2
                        zone = "center" if w/3 < cx < 2*w/3 else ("left" if cx < w/3 else "right")
                        priority = PRIORITIES.get(label, 3)
                        objs.append({"label": label, "zone": zone, "area": area, "prio": priority, "coords": (int(x1), int(y1), int(x2), int(y2))})

                # Speech Logic
                if current_time - self.last_speech_time >= self.frequency:
                    msg, speech = "PATH CLEAR", "The path ahead is clear."
                    objs_sorted = sorted(objs, key=lambda x: (x['prio'], -x['area']))
                    
                    if objs:
                        primary = objs_sorted[0]
                        label = primary['label']
                        zone = primary['zone']
                        
                        if primary['area'] > 0.4:
                            speech = f"Stop immediately. A {label} is directly in front of you."
                        elif zone == "center":
                            left_clear = len([o for o in objs if o['zone'] == 'left']) == 0
                            dir_hint = "Move left." if left_clear else "Move right."
                            speech = f"A {label} is blocking the center. {dir_hint}"
                        else:
                            speech = f"I see a {label} on your {zone}."
                    else:
                        # No YOLO detections — try environment heuristics
                        env = detect_environment(frame)
                        if env:
                            env_label, env_dist = env
                            if env_dist == "VERY_CLOSE":
                                speech = f"Warning! {env_label} directly ahead, very close. Stop."
                            elif env_dist == "CLOSE":
                                speech = f"Warning! {env_label} ahead, close."
                            else:
                                speech = f"{env_label.capitalize()} nearby ahead."
                    print(f"[NAV] {speech}")
                    self.speak(speech)
                    self.last_speech_time = current_time

                # Display Logic
                if self.display:
                    for o in objs:
                        x1, y1, x2, y2 = o['coords']
                        color = (0, 255, 0) if o['area'] < 0.4 else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, o['label'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    title = f"SEEING WITH SOUND - {mode_str}"
                    cv2.imshow(title, frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        except Exception as e:
            print(f"[ERROR] {e}")
        finally:
            if self.pc_test:
                self.cap.release()
            else:
                self.picam2.stop()
                self.picam2.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEEING WITH SOUND - RPi 5 Navigator")
    parser.add_argument("--freq", type=float, default=1.0, help="Scan frequency in seconds")
    parser.add_argument("--display", action="store_true", help="Show video window")
    parser.add_argument("--pc-test", action="store_true", help="Force PC Emulation mode (uses Webcam)")
    args = parser.parse_args()

    nav = NavigatorPi(frequency=args.freq, display=args.display, pc_test=args.pc_test)
    nav.run()
