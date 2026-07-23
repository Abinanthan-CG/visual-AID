import streamlit as st
import cv2
import numpy as np
import time
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
from streamlit_autorefresh import st_autorefresh
from ultralytics import YOLO
import av

# --- HIGH CONTRAST ACCESSIBILITY UI CONFIG ---
st.set_page_config(page_title="VisionAid", page_icon="👁️", layout="centered")

st.markdown("""
    <style>
    /* High Contrast Accessibility Theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    
    .main { background-color: #000000 !important; color: #FFFFFF !important; font-family: 'Inter', sans-serif; }
    /* Hide top menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp { max-width: 100%; padding: 0; background-color: #000000 !important; }
    
    p, li, span, div { font-size: 1.1rem !important; color: #FFFFFF !important; }
    
    .stExpander {
        background: #0D1117 !important;
        border: 2px solid #1A73E8 !important;
        border-radius: 10px;
        margin: 10px;
    }
    
    .stExpander summary { color: #FFFFFF !important; font-size: 1.2rem !important; }
    
    .element-container img { border-radius: 8px; width: 100% !important; border: 2px solid #1A73E8; }
    
    /* Status Panel underneath camera */
    .status-panel {
        background: #0D1117;
        padding: 20px;
        border: 3px solid #1A73E8;
        border-radius: 12px;
        text-align: center;
        margin-top: 15px;
    }
    
    .status-text {
        font-size: 20px !important;
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    .status-clear { color: #888888 !important; } /* FAR mapped color */
    
    /* Distance Colors for Text */
    .dist-very-close { color: #DB4437 !important; }
    .dist-close { color: #F4B400 !important; }
    .dist-medium { color: #0F9D58 !important; }
    .dist-far { color: #888888 !important; }
    
    /* Caregiver text */
    .caregiver-subtitle {
        color: #888888 !important;
        font-size: 1rem !important;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


# --- CONFIGURATIONS & STATE ---
LANGUAGES = {
    "English": {
        "code": "en-US",
        "warning": "Warning",
        "ahead": "ahead",
        "left": "on your left",
        "right": "on your right",
        "very_close": "very close",
        "close": "close",
        "nearby": "nearby",
        "clear": "Path appears clear",
        "dark": "Environment too dark",
        "started": "VisionAid navigation started",
        "obstacle": "Obstacle ahead",
        "move_left": "Move left",
        "move_right": "Move right",
        "both_blocked": "Proceed with caution",
    },
    "Tamil": {
        "code": "ta-IN",
        "warning": "எச்சரிக்கை",
        "ahead": "முன்னால்",
        "left": "இடதுபுறம்",
        "right": "வலதுபுறம்",
        "very_close": "மிக அருகில்",
        "close": "அருகில்",
        "nearby": "அண்மையில்",
        "clear": "பாதை தெளிவாக உள்ளது",
        "dark": "சூழல் மிகவும் இருட்டாக உள்ளது",
        "started": "விஷன்எய்ட் தொடங்கியது",
        "obstacle": "தடை முன்னால்",
        "move_left": "இடதுபுறம் செல்லுங்கள்",
        "move_right": "வலதுபுறம் செல்லுங்கள்",
        "both_blocked": "கவனமாக செல்லுங்கள்",
    },
    "Hindi": {
        "code": "hi-IN",
        "warning": "चेतावनी",
        "ahead": "आगे",
        "left": "बाईं तरफ",
        "right": "दाईं तरफ",
        "very_close": "बहुत पास",
        "close": "पास",
        "nearby": "नज़दीक",
        "clear": "रास्ता साफ है",
        "dark": "वातावरण बहुत अंधेरा है",
        "started": "VisionAid शुरू हो गया",
        "obstacle": "बाधा आगे",
        "move_left": "बाईं तरफ जाएं",
        "move_right": "दाईं तरफ जाएं",
        "both_blocked": "सावधानी से आगे बढ़ें",
    }
}

OBJECT_TRANSLATIONS = {
    "Tamil": {
        "person": "நபர்", "car": "கார்", "truck": "லாரி",
        "bus": "பேருந்து", "motorcycle": "மோட்டார் சைக்கிள்",
        "bicycle": "சைக்கிள்", "chair": "நாற்காலி",
        "dining table": "மேசை", "bottle": "பாட்டில்",
        "dog": "நாய்", "cat": "பூனை", "door": "கதவு",
        "bed": "படுக்கை", "toilet": "கழிவறை",
        "tv": "தொலைக்காட்சி", "laptop": "மடிக்கணினி",
        "cell phone": "கைப்பேசி", "book": "புத்தகம்",
        "clock": "கடிகாரம்", "cup": "கோப்பை",
        "traffic light": "போக்குவரத்து விளக்கு",
        "fire hydrant": "தீயணைப்பு குழாய்",
        "stop sign": "நிறுத்த அடையாளம்",
        "bench": "இருக்கை", "backpack": "பை",
        "umbrella": "குடை", "handbag": "கைப்பை",
        "suitcase": "பெட்டி", "sports ball": "பந்து",
        "couch": "சோபா", "potted plant": "தாவரம்",
        "sink": "கழுவுதொட்டி", "refrigerator": "குளிர்சாதனப்பெட்டி",
        "scissors": "கத்தரிக்கோல்", "vase": "பூச்சட்டி",
        # Environment labels (non-COCO, detected via CV heuristics)
        "wall": "சுவர்", "tree": "மரம்", "obstacle": "தடை",
    },
    "Hindi": {
        "person": "व्यक्ति", "car": "कार", "truck": "ट्रक",
        "bus": "बस", "motorcycle": "मोटरसाइकिल",
        "bicycle": "साइकिल", "chair": "कुर्सी",
        "dining table": "मेज़", "bottle": "बोतल",
        "dog": "कुत्ता", "cat": "बिल्ली", "door": "दरवाज़ा",
        "bed": "बिस्तर", "toilet": "शौचालय",
        "tv": "टीवी", "laptop": "लैपटॉप",
        "cell phone": "मोबाइल फ़ोन", "book": "किताब",
        "clock": "घड़ी", "cup": "कप",
        "traffic light": "ट्रैफ़िक लाइट",
        "fire hydrant": "अग्निशमन यंत्र",
        "stop sign": "रुकने का संकेत",
        "bench": "बेंच", "backpack": "बैग",
        "umbrella": "छाता", "handbag": "पर्स",
        "suitcase": "सूटकेस", "sports ball": "गेंद",
        "couch": "सोफ़ा", "potted plant": "पौधा",
        "sink": "नल", "refrigerator": "फ्रिज",
        "scissors": "कैंची", "vase": "फूलदान",
        # Environment labels (non-COCO, detected via CV heuristics)
        "wall": "दीवार", "tree": "पेड़", "obstacle": "बाधा",
    }
}

PRIORITY_OBJECTS = {"person", "car", "truck", "bus", "motorcycle", "bicycle", "chair", "stairs", "door"}

if "last_spoken" not in st.session_state: st.session_state.last_spoken = ""
if "last_speak_time" not in st.session_state: st.session_state.last_speak_time = 0
if "last_dark_warn" not in st.session_state: st.session_state.last_dark_warn = 0
if "frame_count" not in st.session_state: st.session_state.frame_count = 0
if "det_count" not in st.session_state: st.session_state.det_count = 0
if "ui_msg" not in st.session_state: st.session_state.ui_msg = "START NAVIGATION to begin"
if "ui_msg_class" not in st.session_state: st.session_state.ui_msg_class = "status-clear"
if "last_dir_key" not in st.session_state: st.session_state.last_dir_key = ""  # direction+label fingerprint
if "last_dir_time" not in st.session_state: st.session_state.last_dir_time = 0

# --- INJECT HAPTIC ARM BUTTON & POLLING LOOP ---
components.html(f"""
    <div id="haptic-container" style="text-align: center; margin-bottom: 10px;">
        <button id="arm-btn" style="
            background-color: #DB4437; color: white; border: none; 
            padding: 15px 30px; font-size: 1.2rem; font-weight: bold; 
            border-radius: 8px; cursor: pointer; width: 100%; max-width: 400px;
            font-family: 'Inter', sans-serif;">
            TAP TO ENABLE HAPTICS
        </button>
        <div id="armed-indicator" style="
            display: none; color: #0F9D58; font-size: 1.2rem; 
            font-weight: bold; font-family: 'Inter', sans-serif;
            padding: 10px;">
            📳 Haptics ON
        </div>
    </div>
    <script>
    window.hapticsArmed = false;
    window.currentVibrationPattern = [0];
    window.lastVibrationPattern = [0];
    window.lastVibrationTime = 0;
    document.getElementById('arm-btn').addEventListener('click', function() {{
        window.hapticsArmed = true;
        this.style.display = 'none';
        document.getElementById('armed-indicator').style.display = 'block';
        if(navigator.vibrate) navigator.vibrate(50);
    }});
    setInterval(function() {{
        if(!window.hapticsArmed || !navigator.vibrate) return;
        let now = Date.now();
        if (JSON.stringify(window.currentVibrationPattern) !== JSON.stringify(window.lastVibrationPattern) || (now - window.lastVibrationTime) > 2000) {{
             if (window.currentVibrationPattern.length > 0 && window.currentVibrationPattern[0] !== 0) {{
                 navigator.vibrate(window.currentVibrationPattern);
                 window.lastVibrationPattern = [...window.currentVibrationPattern];
                 window.lastVibrationTime = now;
             }}
        }}
    }}, 300);
    </script>
""", height=100)

# Sidebar Config
st.sidebar.title("Accessibility Settings")
selected_lang_name = st.sidebar.selectbox("Language / மொழி / भाषा", options=list(LANGUAGES.keys()))
lang_cfg = LANGUAGES[selected_lang_name]

# Stats
st.sidebar.markdown("### About This View")
st.sidebar.markdown("""
<div style="color: #bbb; font-size: 0.95rem;">
This screen is for caregivers and developers.<br>
Real-time navigation is currently running at optimized latency.
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
stats_placeholder = st.sidebar.empty()

# --- ML MODELS ---
@st.cache_resource(show_spinner="Loading YOLOv8n model...")
def load_yolo():
    return YOLO("yolov8n.pt")

try:
    MODEL = load_yolo()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()
    
# --- HELPER FUNCTIONS ---
def translate_label(label, language):
    if language == "English": return label
    return OBJECT_TRANSLATIONS.get(language, {}).get(label, label)

def trigger_voice_and_haptic(text, dist_level="FAR", dir_key=""):
    """Speak text with smart multi-layer debouncing to prevent audio spam."""
    # Guard: never speak empty or whitespace-only strings
    if not text or not text.strip():
        return
    # Sanitize: remove apostrophes that would break JS string literals
    safe_text = text.replace("'", "").replace('"', '')

    now = time.time() * 1000

    # Layer 1 — Global floor: minimum 700ms between ANY speech call
    MIN_ANY_MS = 700
    if (now - st.session_state.last_speak_time) < MIN_ANY_MS:
        return

    # Layer 2 — Direction-key debounce: suppress if same direction+label within 2.5s
    # This stops "car on your right, close" / "car on your right, nearby" spam
    DIR_DEBOUNCE_MS = 2500
    if dir_key and dir_key == st.session_state.last_dir_key:
        if (now - st.session_state.last_dir_time) < DIR_DEBOUNCE_MS:
            return

    # Layer 3 — Exact text debounce
    DEBOUNCE_MS = 2000
    URGENT_DEBOUNCE_MS = 800
    current_debounce = URGENT_DEBOUNCE_MS if dist_level == "VERY_CLOSE" else DEBOUNCE_MS
    if safe_text == st.session_state.last_spoken and (now - st.session_state.last_speak_time) < current_debounce:
        return

    # Commit
    st.session_state.last_spoken = safe_text
    st.session_state.last_speak_time = now
    if dir_key:
        st.session_state.last_dir_key = dir_key
        st.session_state.last_dir_time = now

    # Haptic Pattern for JS
    vib_pattern = "[150]"
    if dist_level == "VERY_CLOSE": vib_pattern = "[100, 50, 100, 50, 100]"
    elif dist_level == "CLOSE": vib_pattern = "[200, 100, 200]"
    elif dist_level == "MEDIUM": vib_pattern = "[300]"

    components.html(f"""
        <script>
        if (window.parent.window.currentVibrationPattern !== undefined) {{
            window.parent.window.currentVibrationPattern = {vib_pattern};
        }}
        var msg = new SpeechSynthesisUtterance('{safe_text}');
        msg.lang = '{lang_cfg["code"]}';
        msg.rate = 1.0;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(msg);
        </script>
    """, height=0)

# --- LAYERED ENVIRONMENT DETECTORS (for surfaces YOLO cannot classify) ---

def detect_wall(frame_bgr):
    """
    Detects wall-like flat surfaces using texture analysis.
    Walls have low pixel variance (smooth, uniform) and cover large frame areas.
    Runs in ~2ms on CPU.
    """
    h, w = frame_bgr.shape[:2]
    # Sample the bottom-center of the frame where walls block the path
    strip = frame_bgr[int(h*0.3):int(h*0.9), int(w*0.2):int(w*0.8)]
    gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)

    # Skip if too dark (not a wall, just low light)
    mean_brightness = np.mean(gray_strip)
    if mean_brightness < 20:
        return None

    # Global texture check: walls have low std-dev overall
    _, std_dev = cv2.meanStdDev(gray_strip)
    global_std = std_dev[0][0]

    # Fine-grained: count how many 4x4 grid cells are near-uniform
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
    Runs in ~3ms on CPU.
    """
    h, w = frame_bgr.shape[:2]
    # Center zone
    cx1, cx2 = int(w*0.2), int(w*0.8)
    cy1, cy2 = int(h*0.1), int(h*0.9)
    center = frame_bgr[cy1:cy2, cx1:cx2]

    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    # Green hue range (covers grass, leaves, bushes)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    green_ratio = np.count_nonzero(mask) / float(mask.size)

    if green_ratio > 0.50:
        return "CLOSE"
    elif green_ratio > 0.20:
        return "MEDIUM"
    return None


def detect_obstacle_contour(frame_bgr):
    """
    Generic obstacle detection using contour area in the center zone.
    Catches unknown objects that YOLO missed. Runs in <5ms on CPU.
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    edges = cv2.Canny(blurred, 30, 100)

    cx1, cx2 = int(w*0.25), int(w*0.75)
    cy1, cy2 = int(h*0.15), int(h*0.85)
    center_edges = edges[cy1:cy2, cx1:cx2]

    contours, _ = cv2.findContours(center_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_area = (cx2-cx1) * (cy2-cy1)
    total_contour_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 500)
    ratio = total_contour_area / center_area

    if ratio > 0.45:   return "VERY_CLOSE"
    elif ratio > 0.25: return "CLOSE"
    return None


def detect_environment(frame_bgr):
    """
    Multi-layered fallback detector for surfaces YOLO cannot classify.
    Priority order: wall (most dangerous) > obstacle (unknown) > vegetation.
    Returns (label_en, dist_level) or None.
    """
    wall_dist = detect_wall(frame_bgr)
    if wall_dist:
        return ("wall", wall_dist)

    obs_dist = detect_obstacle_contour(frame_bgr)
    if obs_dist:
        return ("obstacle", obs_dist)

    veg_dist = detect_vegetation(frame_bgr)
    if veg_dist:
        return ("tree", veg_dist)

    return None

def letterbox(image_bgr, target_size=320): # [FIX 3] Target 320
    h, w = image_bgr.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image_bgr, (new_w, new_h))
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_x, pad_y = (target_size - new_w) // 2, (target_size - new_h) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    return canvas, scale, pad_x, pad_y

def unletterbox_bbox(x1, y1, x2, y2, scale, pad_x, pad_y):
    return (x1-pad_x)/scale, (y1-pad_y)/scale, (x2-pad_x)/scale, (y2-pad_y)/scale

def estimate_distance(bbox_height, frame_height):
    ratio = bbox_height / frame_height
    if ratio > 0.60:   return "VERY_CLOSE"
    elif ratio > 0.40: return "CLOSE"
    elif ratio > 0.20: return "MEDIUM"
    return "FAR"

def classify_direction(bbox_center_x, frame_width):
    ratio = bbox_center_x / frame_width
    if ratio < 0.35: return "LEFT"
    elif ratio > 0.65: return "RIGHT"
    return "CENTER"

def get_dist_class(dist_lvl):
    if dist_lvl == "VERY_CLOSE": return "dist-very-close"
    if dist_lvl == "CLOSE": return "dist-close"
    if dist_lvl == "MEDIUM": return "dist-medium"
    return "dist-far"

def get_dir_icon(direction):
    if direction == "LEFT": return "◀"
    elif direction == "RIGHT": return "▶"
    return "▲"

# --- WEBRTC PROCESSOR ---
class VideoProcessor:
    def __init__(self):
        self.latest_announce = ""
        self.latest_dist = "FAR"
        self.latest_dir_key = ""   # direction+label fingerprint for spam filter
        self.total_frames = 0
        self.total_dets = 0
        self.last_inference_time = 0
        self.last_results = []
        self.empty_count = 0

    def recv(self, frame):
        frame_bgr = frame.to_ndarray(format="bgr24")
        orig_h, orig_w = frame_bgr.shape[:2]
        self.total_frames += 1
        
        now = time.time()
        INFERENCE_INTERVAL = 0.4 # [FIX 4] 400ms interval
        
        if now - self.last_inference_time >= INFERENCE_INTERVAL:
            # [FIX 3] Run inference at 320
            lb_img, scale, pad_x, pad_y = letterbox(frame_bgr, target_size=320)
            results = MODEL(lb_img, verbose=False, conf=0.50, imgsz=320)[0]
            
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                ux1, uy1, ux2, uy2 = unletterbox_bbox(x1, y1, x2, y2, scale, pad_x, pad_y)
                cls_id = int(box.cls[0])
                label_en = MODEL.names[cls_id]
                dist = estimate_distance(uy2-uy1, orig_h)
                direction = classify_direction((ux1+ux2)/2, orig_w)
                
                detections.append({
                    "label": translate_label(label_en, selected_lang_name),
                    "label_en": label_en,
                    "dist": dist,
                    "dir": direction,
                    "bbox": (int(ux1), int(uy1), int(ux2), int(uy2)),
                    "rank": (0 if label_en in PRIORITY_OBJECTS else 1, 
                             0 if dist=="VERY_CLOSE" else (1 if dist=="CLOSE" else 2))
                })
            
            self.last_results = detections
            self.last_inference_time = now
            self.total_dets += len(detections)
        
        # UI logic using latest results (persistent display)
        annotated_frame = frame_bgr.copy()
        if len(self.last_results) > 0:
            self.empty_count = 0
            self.last_results.sort(key=lambda x: x["rank"])
            primary = self.last_results[0]
            plabel, pdist, pdir = primary["label"], primary["dist"], primary["dir"]
            
            px1, py1, px2, py2 = primary["bbox"]
            cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 0, 255), 4)
            cv2.putText(annotated_frame, f"{plabel.upper()} {pdist}", (px1, py1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            # Formulate announce string with directional move hints
            move_hint = ""
            if pdir == "CENTER" and pdist in ("VERY_CLOSE", "CLOSE"):
                # Check if adjacent zones are clear to suggest a safer direction
                left_blocked  = any(d["dir"] == "LEFT"  for d in self.last_results[1:])
                right_blocked = any(d["dir"] == "RIGHT" for d in self.last_results[1:])
                if not left_blocked:
                    move_hint = f". {lang_cfg['move_left']}"
                elif not right_blocked:
                    move_hint = f". {lang_cfg['move_right']}"
                else:
                    move_hint = f". {lang_cfg['both_blocked']}"

            if pdist == "VERY_CLOSE" and pdir == "CENTER":
                self.latest_announce = f"{lang_cfg['warning']}! {plabel} {lang_cfg['ahead']}, {lang_cfg['very_close']}{move_hint}"
            elif pdist == "CLOSE" and pdir == "CENTER":
                self.latest_announce = f"{lang_cfg['warning']}! {plabel} {lang_cfg['ahead']}, {lang_cfg['close']}{move_hint}"
            elif pdist == "FAR" and pdir == "CENTER":
                self.latest_announce = f"{plabel} {lang_cfg['ahead']}"
            else:
                dir_str  = lang_cfg["left"] if pdir == "LEFT" else (lang_cfg["right"] if pdir == "RIGHT" else lang_cfg["ahead"])
                dist_str = lang_cfg["very_close"] if pdist == "VERY_CLOSE" else (lang_cfg["close"] if pdist == "CLOSE" else (lang_cfg["nearby"] if pdist == "MEDIUM" else ""))
                self.latest_announce = f"{plabel} {dir_str}, {dist_str}".strip(", ")
            self.latest_dist = pdist
            self.latest_dir_key = f"{primary['label_en']}_{pdir}"  # fingerprint for spam filter
            
        else: # No YOLO detections — run layered environment detector
            env = detect_environment(frame_bgr)
            if env:
                self.empty_count = 0
                env_label_en, env_dist = env
                env_label = translate_label(env_label_en, selected_lang_name)
                if env_dist == "VERY_CLOSE":
                    dist_str = lang_cfg["very_close"]
                    self.latest_announce = f"{lang_cfg['warning']}! {env_label} {lang_cfg['ahead']}, {dist_str}"
                elif env_dist == "CLOSE":
                    dist_str = lang_cfg["close"]
                    self.latest_announce = f"{lang_cfg['warning']}! {env_label} {lang_cfg['ahead']}, {dist_str}"
                else:  # MEDIUM — vegetation typically detected at range
                    self.latest_announce = f"{env_label} {lang_cfg['ahead']}, {lang_cfg['nearby']}"
                self.latest_dist = env_dist
            else:
                self.empty_count += 1
                if self.empty_count >= 10:
                    self.latest_announce = lang_cfg["clear"]
                    self.latest_dist = "FAR"
            
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- MAIN UI ---
st.markdown("<h1>VisionAid</h1>", unsafe_allow_html=True)
st.markdown("<p class='caregiver-subtitle'>👁 CAREGIVER VIEW — Optimized Performance</p>", unsafe_allow_html=True)

webrtc_ctx = webrtc_streamer(
    key="visionaid",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
    async_processing=False,
)

if webrtc_ctx.state.playing:
    st_autorefresh(interval=500, key="voice_trigger_loop") # [FIX 4] 500ms polling
    if webrtc_ctx.video_processor:
        proc = webrtc_ctx.video_processor
        if proc.latest_announce:
            st.session_state.frame_count = proc.total_frames
            st.session_state.det_count = proc.total_dets
            trigger_voice_and_haptic(proc.latest_announce, proc.latest_dist, proc.latest_dir_key)
            st.session_state.ui_msg = proc.latest_announce.upper()
            st.session_state.ui_msg_class = get_dist_class(proc.latest_dist)

stats_placeholder.markdown(f"**Frames:** {st.session_state.frame_count} | **Detections:** {st.session_state.det_count}")

st.markdown(f"""
    <div class="status-panel">
        <div class="status-text {st.session_state.ui_msg_class}">
            {st.session_state.ui_msg}<br>
            <span style='font-size: 0.85rem; color: #888888; font-weight: 400;'>Voice Engine Active</span>
        </div>
    </div>
""", unsafe_allow_html=True)
