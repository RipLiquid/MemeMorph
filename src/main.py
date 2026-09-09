import cv2
import math
import mediapipe as mp
import numpy as np
import time

from pathlib import Path
from PIL import Image, ImageSequence

from pose_tracker import PoseTracker, draw_pose
from hand_tracker import HandTracker, draw_hands


# ============================================================
# DISPLAY / PERFORMANCE
# ============================================================

OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080

# Keep the final UI at 1080p, but run AI inference on a much smaller frame.
# MediaPipe landmarks are normalized, so they still map correctly to 1080p.
DETECTION_WIDTH = 640
DETECTION_HEIGHT = 360

# Independent detector cadences.
# Face stays responsive, hands run at roughly half rate, and pose runs less often
# because it is only needed for larger body gestures such as Absolute Cinema.
FACE_DETECTION_EVERY_N_FRAMES = 1
HAND_DETECTION_EVERY_N_FRAMES = 2
POSE_DETECTION_EVERY_N_FRAMES = 4

# Drawing all 478 face points every frame is expensive.
# When landmark drawing is enabled, only every Nth point is drawn.
FACE_LANDMARK_DRAW_STEP = 6

PANEL_X = 24
PANEL_Y = 24
PANEL_WIDTH = 520
PANEL_HEIGHT = 1032

FONT = cv2.FONT_HERSHEY_DUPLEX
MANUAL_REACTION_SECONDS = 3.0


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FACE_MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"
POSE_MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"
HAND_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

REACTION_DIR = PROJECT_ROOT / "assets" / "reactions"


# ============================================================
# REACTIONS
# ============================================================

REACTIONS = {
    "speed": {
        "label": "Speed Reverse Smile",
        "file": "speed_reverse_smile.gif",
        "overlay_scale": 1.45,
        "hold": 0.18,
        "release": 0.28,
    },
    "eyebrow": {
        "label": "Eyebrow Raise",
        "file": "rock_eyebrow.gif",
        "overlay_scale": 1.40,
        "hold": 0.28,
        "release": 0.30,
    },
    "surprised": {
        "label": "Surprised",
        "file": "surprised_pikachu.gif",
        "overlay_scale": 1.40,
        "hold": 0.18,
        "release": 0.28,
    },
    "sad": {
        "label": "Sad / Cry",
        "file": "sad_cry.gif",
        "overlay_scale": 1.40,
        "hold": 0.28,
        "release": 0.35,
    },
    "side_eye": {
        "label": "Side Eye",
        "file": "side_eye.gif",
        "overlay_scale": 1.40,
        "hold": 0.24,
        "release": 0.32,
    },
    "jerry_laugh": {
        "label": "Jerry Laugh",
        "file": "mocking_tom_jerry.gif",
        "overlay_scale": 1.45,
        "hold": 0.22,
        "release": 0.32,
    },
    "jerry_point": {
        "label": "Jerry Point & Laugh",
        "file": "laughing.gif",
        "overlay_scale": 1.55,
        "hold": 0.20,
        "release": 0.38,
    },
    "facepalm": {
        "label": "Facepalm",
        "file": "hands_on_head.gif",
        "overlay_scale": 1.55,
        "hold": 0.20,
        "release": 0.42,
    },
    "thumbs_up": {
        "label": "Thumbs Up",
        "file": "thumbs_up.gif",
        "overlay_scale": 1.55,
        "hold": 0.22,
        "release": 0.40,
    },
    "absolute_cinema": {
        "label": "Absolute Cinema",
        "file": "absolute_cinema.gif",
        "overlay_scale": 1.80,
        "hold": 0.28,
        "release": 0.48,
    },
}

# Manual test keys.
MANUAL_KEYS = {
    ord("1"): "speed",
    ord("2"): "eyebrow",
    ord("3"): "surprised",
    ord("4"): "sad",
    ord("5"): "side_eye",
    ord("6"): "jerry_laugh",
    ord("7"): "jerry_point",
    ord("8"): "facepalm",
    ord("9"): "thumbs_up",
    ord("0"): "absolute_cinema",
}


# ============================================================
# MEDIAPIPE FACE BLENDSHAPE ORDER
# ============================================================

BLENDSHAPE_NAMES = [
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
]


# ============================================================
# GIF PLAYER
# ============================================================

class GifPlayer:
    def __init__(self, path):
        self.frames = []
        self.durations = []
        self.total_duration = 0.0
        self.started_at = None
        self.load(path)

    def load(self, path):
        if not path.exists():
            print(f"[MISSING] {path.name}")
            return

        gif = Image.open(path)

        for source_frame in ImageSequence.Iterator(gif):
            rgba = source_frame.convert("RGBA")
            bgra = cv2.cvtColor(np.array(rgba), cv2.COLOR_RGBA2BGRA)

            duration_ms = source_frame.info.get(
                "duration",
                gif.info.get("duration", 80),
            )
            duration_ms = max(int(duration_ms or 80), 20)

            self.frames.append(bgra)
            self.durations.append(duration_ms / 1000.0)

        self.total_duration = sum(self.durations)

        print(
            f"[LOADED] {path.name:<28} "
            f"{len(self.frames):>3} frame(s)"
        )

    def available(self):
        return bool(self.frames) and self.total_duration > 0

    def start(self, now=None):
        self.started_at = now if now is not None else time.perf_counter()

    def stop(self):
        self.started_at = None

    def get_frame(self, now=None):
        if not self.available():
            return None

        now = now if now is not None else time.perf_counter()

        if self.started_at is None:
            self.start(now)

        elapsed = (now - self.started_at) % self.total_duration
        accumulated = 0.0

        for index, duration in enumerate(self.durations):
            accumulated += duration
            if elapsed <= accumulated:
                return self.frames[index]

        return self.frames[-1]


# ============================================================
# BASIC HELPERS
# ============================================================

def average(a, b):
    return (a + b) / 2.0


def get_value(values, name):
    return values.get(name, 0.0)


def ema(previous, new_value, alpha=0.18):
    """Exponential moving average for stable inference timing numbers."""
    if previous <= 0.0:
        return new_value
    return previous * (1.0 - alpha) + new_value * alpha


def point_xy(landmark):
    return np.array([landmark.x, landmark.y], dtype=np.float32)


def distance_landmarks(a, b):
    return float(np.linalg.norm(point_xy(a) - point_xy(b)))


def distance_points(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def joint_angle(a, b, c):
    ba = point_xy(a) - point_xy(b)
    bc = point_xy(c) - point_xy(b)

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator <= 1e-8:
        return 180.0

    cosine = float(np.dot(ba, bc) / denominator)
    cosine = np.clip(cosine, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


# ============================================================
# CAMERA / OUTPUT
# ============================================================

def make_16_9_1080p(frame):
    height, width = frame.shape[:2]
    target_ratio = 16 / 9
    current_ratio = width / height

    if current_ratio > target_ratio:
        new_width = int(height * target_ratio)
        start_x = (width - new_width) // 2
        frame = frame[:, start_x:start_x + new_width]

    elif current_ratio < target_ratio:
        new_height = int(width / target_ratio)
        start_y = (height - new_height) // 2
        frame = frame[start_y:start_y + new_height, :]

    # Avoid an expensive full-HD resize if the camera already delivered 1080p.
    if frame.shape[1] == OUTPUT_WIDTH and frame.shape[0] == OUTPUT_HEIGHT:
        return frame

    return cv2.resize(
        frame,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )


# ============================================================
# FACE GEOMETRY / GIF OVERLAY
# ============================================================

def get_face_bounds_normalized(face_landmarks):
    xs = [landmark.x for landmark in face_landmarks]
    ys = [landmark.y for landmark in face_landmarks]

    return min(xs), min(ys), max(xs), max(ys)


def get_face_box_pixels(face_landmarks, frame_width, frame_height):
    x1, y1, x2, y2 = get_face_bounds_normalized(face_landmarks)

    return (
        int(x1 * frame_width),
        int(y1 * frame_height),
        int(x2 * frame_width),
        int(y2 * frame_height),
    )


def overlay_reaction(frame, gif_frame, face_landmarks, scale):
    """
    Preserve the GIF's aspect ratio and center it on the head.
    """
    if gif_frame is None or not face_landmarks:
        return frame

    frame_height, frame_width = frame.shape[:2]

    face_x1, face_y1, face_x2, face_y2 = get_face_box_pixels(
        face_landmarks,
        frame_width,
        frame_height,
    )

    face_height = max(face_y2 - face_y1, 1)
    face_center_x = (face_x1 + face_x2) // 2
    face_center_y = (face_y1 + face_y2) // 2

    gif_height, gif_width = gif_frame.shape[:2]

    target_height = int(face_height * scale)
    target_width = int(target_height * gif_width / max(gif_height, 1))

    # Prevent oversized overlays.
    max_width = int(frame_width * 0.55)
    max_height = int(frame_height * 0.72)

    if target_width > max_width:
        ratio = max_width / target_width
        target_width = int(target_width * ratio)
        target_height = int(target_height * ratio)

    if target_height > max_height:
        ratio = max_height / target_height
        target_width = int(target_width * ratio)
        target_height = int(target_height * ratio)

    center_y = int(face_center_y - face_height * 0.08)

    x1 = int(face_center_x - target_width / 2)
    y1 = int(center_y - target_height / 2)
    x2 = x1 + target_width
    y2 = y1 + target_height

    clipped_x1 = max(0, x1)
    clipped_y1 = max(0, y1)
    clipped_x2 = min(frame_width, x2)
    clipped_y2 = min(frame_height, y2)

    if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
        return frame

    resized = cv2.resize(
        gif_frame,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    )

    source_x1 = clipped_x1 - x1
    source_y1 = clipped_y1 - y1
    source_x2 = source_x1 + (clipped_x2 - clipped_x1)
    source_y2 = source_y1 + (clipped_y2 - clipped_y1)

    resized = resized[source_y1:source_y2, source_x1:source_x2]

    gif_bgr = resized[:, :, :3]
    alpha = (resized[:, :, 3].astype(np.float32) / 255.0)[:, :, None]

    roi = frame[clipped_y1:clipped_y2, clipped_x1:clipped_x2]
    blended = gif_bgr * alpha + roi * (1.0 - alpha)

    frame[clipped_y1:clipped_y2, clipped_x1:clipped_x2] = (
        blended.astype(np.uint8)
    )

    return frame


# ============================================================
# FACE ANALYSIS
# ============================================================

def analyze_face(values):
    blink = average(
        get_value(values, "eyeBlinkLeft"),
        get_value(values, "eyeBlinkRight"),
    )

    squint = average(
        get_value(values, "eyeSquintLeft"),
        get_value(values, "eyeSquintRight"),
    )

    wide = average(
        get_value(values, "eyeWideLeft"),
        get_value(values, "eyeWideRight"),
    )

    brow_down = average(
        get_value(values, "browDownLeft"),
        get_value(values, "browDownRight"),
    )

    brow_outer_left = get_value(values, "browOuterUpLeft")
    brow_outer_right = get_value(values, "browOuterUpRight")

    mouth_smile = average(
        get_value(values, "mouthSmileLeft"),
        get_value(values, "mouthSmileRight"),
    )

    mouth_frown = average(
        get_value(values, "mouthFrownLeft"),
        get_value(values, "mouthFrownRight"),
    )

    mouth_press = average(
        get_value(values, "mouthPressLeft"),
        get_value(values, "mouthPressRight"),
    )

    mouth_roll = average(
        get_value(values, "mouthRollLower"),
        get_value(values, "mouthRollUpper"),
    )

    look_left = average(
        get_value(values, "eyeLookOutLeft"),
        get_value(values, "eyeLookInRight"),
    )

    look_right = average(
        get_value(values, "eyeLookInLeft"),
        get_value(values, "eyeLookOutRight"),
    )

    return {
        "blink": blink,
        "squint": squint,
        "wide": wide,
        "brow_down": brow_down,
        "brow_inner_up": get_value(values, "browInnerUp"),
        "brow_outer_left": brow_outer_left,
        "brow_outer_right": brow_outer_right,
        "brow_asymmetry": abs(brow_outer_left - brow_outer_right),
        "jaw_open": get_value(values, "jawOpen"),
        "mouth_smile": mouth_smile,
        "mouth_frown": mouth_frown,
        "mouth_pucker": get_value(values, "mouthPucker"),
        "mouth_funnel": get_value(values, "mouthFunnel"),
        "mouth_press": mouth_press,
        "mouth_roll": mouth_roll,
        "look_left": look_left,
        "look_right": look_right,
        "side_eye": max(look_left, look_right),
    }


# ============================================================
# HAND ANALYSIS
# ============================================================

def hand_palm_center(landmarks):
    indices = [0, 5, 9, 13, 17]

    return (
        sum(landmarks[index].x for index in indices) / len(indices),
        sum(landmarks[index].y for index in indices) / len(indices),
    )


def hand_scale(landmarks):
    # Wrist -> middle-finger MCP.
    return max(distance_landmarks(landmarks[0], landmarks[9]), 0.02)


def finger_extended(landmarks, mcp_index, pip_index, tip_index):
    """
    More orientation tolerant than only comparing y-values.
    """
    angle = joint_angle(
        landmarks[mcp_index],
        landmarks[pip_index],
        landmarks[tip_index],
    )

    wrist = landmarks[0]
    tip_distance = distance_landmarks(wrist, landmarks[tip_index])
    pip_distance = distance_landmarks(wrist, landmarks[pip_index])

    return angle > 145 and tip_distance > pip_distance * 1.10


def non_thumb_extended_count(landmarks):
    fingers = [
        (5, 6, 8),
        (9, 10, 12),
        (13, 14, 16),
        (17, 18, 20),
    ]

    return sum(
        1
        for mcp, pip, tip in fingers
        if finger_extended(landmarks, mcp, pip, tip)
    )


def non_thumb_folded_count(landmarks):
    return 4 - non_thumb_extended_count(landmarks)


def thumb_extended(landmarks):
    wrist = landmarks[0]
    thumb_ip = landmarks[3]
    thumb_tip = landmarks[4]

    return (
        distance_landmarks(wrist, thumb_tip)
        > distance_landmarks(wrist, thumb_ip) * 1.10
    )


def is_open_hand(landmarks):
    return non_thumb_extended_count(landmarks) >= 3


def is_thumb_point(landmarks):
    """
    Jerry point gesture:
      - thumb extended
      - thumb points mostly sideways
      - other fingers mostly folded
    """
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    scale = hand_scale(landmarks)

    dx = thumb_tip.x - wrist.x
    dy = thumb_tip.y - wrist.y

    mostly_horizontal = abs(dx) > abs(dy) * 1.20
    enough_extension = abs(dx) > scale * 0.75

    return (
        thumb_extended(landmarks)
        and mostly_horizontal
        and enough_extension
        and non_thumb_folded_count(landmarks) >= 3
    )


def is_thumbs_up(landmarks):
    """
    Thumbs-up gesture:
      - thumb extended
      - thumb points upward
      - other fingers mostly folded
    """
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    scale = hand_scale(landmarks)

    dx = thumb_tip.x - wrist.x
    dy = thumb_tip.y - wrist.y

    mostly_vertical = abs(dy) > abs(dx) * 0.90
    pointing_up = dy < -(scale * 0.65)

    return (
        thumb_extended(landmarks)
        and mostly_vertical
        and pointing_up
        and non_thumb_folded_count(landmarks) >= 3
    )


def analyze_hands(hand_result):
    data = {
        "count": 0,
        "open_hands": 0,
        "thumb_point": False,
        "thumbs_up": False,
        "palms": [],
    }

    if not hand_result or not hand_result.hand_landmarks:
        return data

    data["count"] = len(hand_result.hand_landmarks)

    for landmarks in hand_result.hand_landmarks:
        center = hand_palm_center(landmarks)
        open_hand = is_open_hand(landmarks)

        data["palms"].append(
            {
                "x": center[0],
                "y": center[1],
                "open": open_hand,
            }
        )

        if open_hand:
            data["open_hands"] += 1

        if is_thumb_point(landmarks):
            data["thumb_point"] = True

        if is_thumbs_up(landmarks):
            data["thumbs_up"] = True

    return data


# ============================================================
# BODY / HAND REACTIONS
# ============================================================

def detect_facepalm(face_landmarks, hand_data):
    """
    Open palm overlapping the upper part of the face.
    """
    if not face_landmarks or not hand_data["palms"]:
        return False

    face_x1, face_y1, face_x2, face_y2 = get_face_bounds_normalized(
        face_landmarks
    )

    face_width = max(face_x2 - face_x1, 0.01)
    face_height = max(face_y2 - face_y1, 0.01)

    target = (
        (face_x1 + face_x2) / 2.0,
        face_y1 + face_height * 0.28,
    )

    max_distance = max(face_width, face_height) * 0.62

    for palm in hand_data["palms"]:
        # Facepalm should be an open-ish palm, not just any fist near the face.
        if not palm["open"]:
            continue

        inside_x = (
            face_x1 - face_width * 0.28
            <= palm["x"]
            <= face_x2 + face_width * 0.28
        )

        inside_y = (
            face_y1 - face_height * 0.30
            <= palm["y"]
            <= face_y1 + face_height * 0.72
        )

        close_enough = (
            distance_points(
                (palm["x"], palm["y"]),
                target,
            )
            < max_distance
        )

        if inside_x and inside_y and close_enough:
            return True

    return False


def detect_absolute_cinema(pose_result, hand_data):
    """
    Absolute Cinema:
      - two hands visible
      - both open
      - both around shoulder height or higher
      - hands spread wider than shoulders
    """
    if (
        not pose_result
        or not pose_result.pose_landmarks
        or hand_data["count"] < 2
        or hand_data["open_hands"] < 2
    ):
        return False

    pose = pose_result.pose_landmarks[0]

    left_shoulder = pose[11]
    right_shoulder = pose[12]

    shoulder_y = average(left_shoulder.y, right_shoulder.y)
    shoulder_width = max(
        abs(left_shoulder.x - right_shoulder.x),
        0.08,
    )

    palms = hand_data["palms"]

    raised_count = sum(
        1
        for palm in palms
        if palm["y"] < shoulder_y + 0.06
    )

    palm_xs = [palm["x"] for palm in palms]
    hand_spread = max(palm_xs) - min(palm_xs)

    return (
        raised_count >= 2
        and hand_spread > shoulder_width * 1.25
    )


# ============================================================
# FACE REACTION CONDITIONS
# ============================================================

def speed_condition(face):
    """
    Tuned around the readings you showed earlier:
      eyeBlink ~0.70
      eyeSquint ~0.63
      browDown ~0.50+
      mouth movement is subtler
    """
    eyes = (
        face["blink"] > 0.50
        or face["squint"] > 0.42
    )

    brows = face["brow_down"] > 0.36

    mouth = (
        face["mouth_pucker"] > 0.050
        or face["mouth_funnel"] > 0.10
        or face["mouth_press"] > 0.14
        or face["mouth_roll"] > 0.14
        or face["mouth_frown"] > 0.12
    )

    return eyes and brows and mouth


def eyebrow_condition(face):
    return (
        face["brow_asymmetry"] > 0.18
        and max(
            face["brow_outer_left"],
            face["brow_outer_right"],
        ) > 0.24
        and face["jaw_open"] < 0.28
        and face["blink"] < 0.55
    )


def surprised_condition(face):
    return (
        face["wide"] > 0.22
        and face["jaw_open"] > 0.30
    )


def sad_condition(face):
    return (
        face["brow_inner_up"] > 0.22
        and face["mouth_frown"] > 0.15
        and face["mouth_smile"] < 0.20
        and face["jaw_open"] < 0.32
    )


def side_eye_condition(face):
    return (
        face["side_eye"] > 0.34
        and face["blink"] < 0.42
        and face["jaw_open"] < 0.30
    )


def jerry_laugh_condition(face):
    return (
        face["mouth_smile"] > 0.35
        and face["jaw_open"] > 0.22
        and (
            face["squint"] > 0.20
            or face["blink"] > 0.25
        )
    )


def jerry_point_condition(face, hand_data):
    """
    Requested Jerry feature:
    sideways thumb point + laughing/smirking face.
    """
    laughing_face = (
        (
            face["mouth_smile"] > 0.18
            or face["jaw_open"] > 0.20
        )
        and (
            face["squint"] > 0.14
            or face["blink"] > 0.22
        )
    )

    return hand_data["thumb_point"] and laughing_face


def thumbs_up_condition(hand_data):
    return hand_data["thumbs_up"]


# ============================================================
# REACTION ENGINE
# ============================================================

def choose_reaction(
    face,
    face_landmarks,
    pose_result,
    hand_data,
):
    # Most specific gestures first.

    if detect_absolute_cinema(pose_result, hand_data):
        return "absolute_cinema"

    if detect_facepalm(face_landmarks, hand_data):
        return "facepalm"

    if jerry_point_condition(face, hand_data):
        return "jerry_point"

    if thumbs_up_condition(hand_data):
        return "thumbs_up"

    if speed_condition(face):
        return "speed"

    if eyebrow_condition(face):
        return "eyebrow"

    if surprised_condition(face):
        return "surprised"

    if sad_condition(face):
        return "sad"

    if side_eye_condition(face):
        return "side_eye"

    if jerry_laugh_condition(face):
        return "jerry_laugh"

    return None


# ============================================================
# DEBUG PANEL
# ============================================================

def draw_transparent_panel(frame):
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (PANEL_X, PANEL_Y),
        (
            PANEL_X + PANEL_WIDTH,
            PANEL_Y + PANEL_HEIGHT,
        ),
        (10, 12, 16),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.78,
        frame,
        0.22,
        0,
        frame,
    )


def draw_text(
    frame,
    text,
    x,
    y,
    size=0.55,
    thickness=1,
    color=(235, 235, 235),
):
    cv2.putText(
        frame,
        text,
        (x, y),
        FONT,
        size,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_section_title(frame, title, y):
    x = PANEL_X + 30

    draw_text(
        frame,
        title.upper(),
        x,
        y,
        size=0.44,
        color=(170, 178, 190),
    )

    cv2.line(
        frame,
        (x, y + 10),
        (
            PANEL_X + PANEL_WIDTH - 28,
            y + 10,
        ),
        (75, 80, 90),
        1,
        cv2.LINE_AA,
    )


def draw_status_badge(frame, label, value, x, y, good):
    width = 138
    height = 54

    fill = (
        (38, 75, 48)
        if good
        else (55, 55, 62)
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        fill,
        -1,
    )

    draw_text(
        frame,
        label,
        x + 12,
        y + 19,
        size=0.36,
        color=(180, 185, 195),
    )

    draw_text(
        frame,
        value,
        x + 12,
        y + 43,
        size=0.52,
        color=(
            (235, 255, 239)
            if good
            else (225, 225, 230)
        ),
    )


def draw_metric(frame, label, value, y):
    x = PANEL_X + 30

    draw_text(
        frame,
        label,
        x,
        y,
        size=0.40,
        color=(205, 208, 214),
    )

    draw_text(
        frame,
        f"{value:.2f}",
        x + 385,
        y,
        size=0.40,
        color=(235, 235, 238),
    )

    bar_x = x + 120
    bar_y = y - 9
    bar_width = 245
    bar_height = 8

    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (55, 58, 65),
        -1,
    )

    amount = int(
        bar_width
        * np.clip(value, 0.0, 1.0)
    )

    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + amount, bar_y + bar_height),
        (205, 205, 210),
        -1,
    )


def reaction_label(name):
    if name is None:
        return "-"

    return REACTIONS[name]["label"]


def draw_debug_panel(
    frame,
    face_detected,
    pose_detected,
    hand_data,
    detected_reaction,
    candidate_reaction,
    display_reaction,
    face,
    gesture_debug,
    fps,
    performance,
):
    draw_transparent_panel(frame)

    left = PANEL_X + 30

    # Header
    draw_text(
        frame,
        "MemeMorph",
        left,
        PANEL_Y + 50,
        size=1.02,
        thickness=2,
        color=(255, 255, 255),
    )

    draw_text(
        frame,
        "REAL-TIME REACTION ENGINE",
        left,
        PANEL_Y + 78,
        size=0.38,
        color=(160, 168, 180),
    )

    draw_text(
        frame,
        (
            f"{fps:4.1f} FPS | "
            f"F {performance['face_ms']:.1f}  "
            f"H {performance['hand_ms']:.1f}  "
            f"P {performance['pose_ms']:.1f} ms"
        ),
        left,
        PANEL_Y + 104,
        size=0.36,
        color=(160, 168, 180),
    )

    # Tracking
    draw_section_title(
        frame,
        "Tracking",
        PANEL_Y + 142,
    )

    badge_y = PANEL_Y + 164

    draw_status_badge(
        frame,
        "FACE",
        "ON" if face_detected else "OFF",
        left,
        badge_y,
        face_detected,
    )

    draw_status_badge(
        frame,
        "POSE",
        "ON" if pose_detected else "OFF",
        left + 150,
        badge_y,
        pose_detected,
    )

    draw_status_badge(
        frame,
        "HANDS",
        str(hand_data["count"]),
        left + 300,
        badge_y,
        hand_data["count"] > 0,
    )

    # Reaction
    draw_section_title(
        frame,
        "Reaction",
        PANEL_Y + 252,
    )

    rows = [
        ("Detected", reaction_label(detected_reaction)),
        ("Candidate", reaction_label(candidate_reaction)),
        ("Active", reaction_label(display_reaction)),
    ]

    row_y = PANEL_Y + 284

    for label, value in rows:
        draw_text(
            frame,
            label,
            left,
            row_y,
            size=0.40,
            color=(160, 168, 180),
        )

        draw_text(
            frame,
            value,
            left + 105,
            row_y,
            size=0.46,
            color=(
                (255, 255, 255)
                if value != "-"
                else (150, 155, 165)
            ),
        )

        row_y += 32

    # Gesture states
    draw_section_title(
        frame,
        "Gestures",
        PANEL_Y + 394,
    )

    gestures = [
        ("Thumb Point", gesture_debug["thumb_point"]),
        ("Thumbs Up", gesture_debug["thumbs_up"]),
        ("Facepalm", gesture_debug["facepalm"]),
        ("Cinema Hands", gesture_debug["cinema"]),
    ]

    gesture_y = PANEL_Y + 425

    for label, active in gestures:
        draw_text(
            frame,
            label,
            left,
            gesture_y,
            size=0.42,
            color=(200, 204, 210),
        )

        draw_text(
            frame,
            "YES" if active else "NO",
            left + 390,
            gesture_y,
            size=0.42,
            color=(
                (100, 230, 140)
                if active
                else (145, 150, 160)
            ),
        )

        gesture_y += 28

    # Face metrics
    draw_section_title(
        frame,
        "Face Metrics",
        PANEL_Y + 556,
    )

    metrics = [
        ("Blink", face["blink"]),
        ("Squint", face["squint"]),
        ("Wide", face["wide"]),
        ("Jaw", face["jaw_open"]),
        ("Smile", face["mouth_smile"]),
        ("Frown", face["mouth_frown"]),
        ("Brow Down", face["brow_down"]),
        ("Brow Inner", face["brow_inner_up"]),
        ("Brow Asym", face["brow_asymmetry"]),
        ("Side Eye", face["side_eye"]),
    ]

    metric_y = PANEL_Y + 592

    for label, value in metrics:
        draw_metric(
            frame,
            label,
            value,
            metric_y,
        )
        metric_y += 31

    # Controls
    draw_section_title(
        frame,
        "Controls",
        PANEL_Y + 922,
    )

    draw_text(
        frame,
        "D  Panel   L  Landmarks   Q  Quit",
        left,
        PANEL_Y + 958,
        size=0.40,
        color=(200, 204, 210),
    )

    draw_text(
        frame,
        "1-9, 0  Manual reaction test",
        left,
        PANEL_Y + 985,
        size=0.40,
        color=(200, 204, 210),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    cv2.setUseOptimized(True)

    required_models = [
        ("Face Landmarker", FACE_MODEL_PATH),
        ("Pose Landmarker", POSE_MODEL_PATH),
        ("Hand Landmarker", HAND_MODEL_PATH),
    ]

    for model_name, model_path in required_models:
        if not model_path.exists():
            print()
            print(f"ERROR: {model_name} model missing.")
            print(f"Expected: {model_path}")
            print()
            return

    # --------------------------------------------------------
    # Load reaction media
    # --------------------------------------------------------

    reaction_players = {}

    print()
    print("MemeMorph reaction assets")
    print("-------------------------")

    for reaction_name, config in REACTIONS.items():
        reaction_players[reaction_name] = GifPlayer(
            REACTION_DIR / config["file"]
        )

    # --------------------------------------------------------
    # MediaPipe Face Landmarker
    # --------------------------------------------------------

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(FACE_MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    pose_tracker = PoseTracker(POSE_MODEL_PATH)
    hand_tracker = HandTracker(HAND_MODEL_PATH)

    # --------------------------------------------------------
    # Webcam
    # --------------------------------------------------------

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera = cv2.VideoCapture(0)

    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        OUTPUT_WIDTH,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        OUTPUT_HEIGHT,
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        30,
    )

    # Some backends ignore this, but when supported it reduces camera latency.
    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        pose_tracker.close()
        hand_tracker.close()
        return

    cv2.namedWindow(
        "MemeMorph",
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        "MemeMorph",
        OUTPUT_WIDTH,
        OUTPUT_HEIGHT,
    )

    print()
    print("MemeMorph 1080p - Performance Build")
    print("----------------------------------")
    print("Output: 1920x1080")
    print("Inference: 640x360")
    print("Face: every frame | Hands: every 2 | Pose: every 4")
    print()
    print("1 = Speed")
    print("2 = Eyebrow")
    print("3 = Surprised")
    print("4 = Sad")
    print("5 = Side Eye")
    print("6 = Jerry Laugh")
    print("7 = Jerry Point + Laugh")
    print("8 = Facepalm")
    print("9 = Thumbs Up")
    print("0 = Absolute Cinema")
    print("D = Toggle debug")
    print("Q = Quit")
    print()

    # --------------------------------------------------------
    # Runtime state
    # --------------------------------------------------------

    debug_mode = True

    candidate_reaction = None
    candidate_since = None

    active_reaction = None
    active_last_seen = 0.0

    manual_reaction = None
    manual_until = 0.0

    previous_display_reaction = None

    frame_counter = 0
    cached_face_result = None
    cached_pose_result = None
    cached_hand_result = None

    last_timestamp_ms = 0

    draw_landmarks = False

    performance = {
        "face_ms": 0.0,
        "hand_ms": 0.0,
        "pose_ms": 0.0,
    }

    fps = 0.0
    fps_last_time = time.perf_counter()
    fps_frame_count = 0

    try:
        with FaceLandmarker.create_from_options(
            face_options
        ) as face_landmarker:

            while True:
                now = time.perf_counter()
                frame_counter += 1

                success, frame = camera.read()

                if not success:
                    print("ERROR: Failed to read webcam frame.")
                    break

                # Selfie mirror.
                frame = cv2.flip(frame, 1)

                # Final 1920x1080 frame.
                frame = make_16_9_1080p(frame)

                # Smaller frame for inference.
                detection_frame = cv2.resize(
                    frame,
                    (DETECTION_WIDTH, DETECTION_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )

                rgb_frame = cv2.cvtColor(
                    detection_frame,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = int(
                    time.perf_counter() * 1000
                )

                timestamp_ms = max(
                    timestamp_ms,
                    last_timestamp_ms + 1,
                )

                last_timestamp_ms = timestamp_ms

                # ------------------------------------------------
                # Face detector
                # ------------------------------------------------

                should_run_face = (
                    cached_face_result is None
                    or frame_counter % FACE_DETECTION_EVERY_N_FRAMES == 0
                )

                if should_run_face:
                    started = time.perf_counter()
                    cached_face_result = face_landmarker.detect_for_video(
                        mp_image,
                        timestamp_ms,
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    performance["face_ms"] = ema(
                        performance["face_ms"],
                        elapsed_ms,
                    )

                # ------------------------------------------------
                # Hands - higher cadence than pose
                # ------------------------------------------------

                should_run_hands = (
                    cached_hand_result is None
                    or frame_counter % HAND_DETECTION_EVERY_N_FRAMES == 0
                )

                if should_run_hands:
                    started = time.perf_counter()
                    cached_hand_result = hand_tracker.detect(
                        mp_image,
                        timestamp_ms,
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    performance["hand_ms"] = ema(
                        performance["hand_ms"],
                        elapsed_ms,
                    )

                # ------------------------------------------------
                # Pose - lower cadence is enough for body gestures
                # ------------------------------------------------

                should_run_pose = (
                    cached_pose_result is None
                    or frame_counter % POSE_DETECTION_EVERY_N_FRAMES == 0
                )

                if should_run_pose:
                    started = time.perf_counter()
                    cached_pose_result = pose_tracker.detect(
                        mp_image,
                        timestamp_ms,
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    performance["pose_ms"] = ema(
                        performance["pose_ms"],
                        elapsed_ms,
                    )

                face_result = cached_face_result
                hand_result = cached_hand_result
                pose_result = cached_pose_result

                # ------------------------------------------------
                # Blendshapes
                # ------------------------------------------------

                blendshape_values = {}

                if face_result.face_blendshapes:
                    for blendshape in face_result.face_blendshapes[0]:
                        index = blendshape.index

                        if 0 <= index < len(BLENDSHAPE_NAMES):
                            blendshape_values[
                                BLENDSHAPE_NAMES[index]
                            ] = blendshape.score

                face_landmarks = None

                if face_result.face_landmarks:
                    face_landmarks = face_result.face_landmarks[0]

                face_data = analyze_face(
                    blendshape_values
                )

                hand_data = analyze_hands(
                    hand_result
                )

                # ------------------------------------------------
                # Current gesture state
                # ------------------------------------------------

                facepalm_now = detect_facepalm(
                    face_landmarks,
                    hand_data,
                )

                cinema_now = detect_absolute_cinema(
                    pose_result,
                    hand_data,
                )

                gesture_debug = {
                    "thumb_point": hand_data["thumb_point"],
                    "thumbs_up": hand_data["thumbs_up"],
                    "facepalm": facepalm_now,
                    "cinema": cinema_now,
                }

                # ------------------------------------------------
                # Raw reaction selection
                # ------------------------------------------------

                detected_reaction = choose_reaction(
                    face_data,
                    face_landmarks,
                    pose_result,
                    hand_data,
                )

                # ------------------------------------------------
                # Stable reaction state
                # ------------------------------------------------

                if active_reaction is not None:
                    if detected_reaction == active_reaction:
                        active_last_seen = now

                    else:
                        release_time = REACTIONS[
                            active_reaction
                        ]["release"]

                        if now - active_last_seen > release_time:
                            active_reaction = None
                            candidate_reaction = None
                            candidate_since = None

                if active_reaction is None:
                    if detected_reaction is None:
                        candidate_reaction = None
                        candidate_since = None

                    elif detected_reaction != candidate_reaction:
                        candidate_reaction = detected_reaction
                        candidate_since = now

                    else:
                        required_hold = REACTIONS[
                            candidate_reaction
                        ]["hold"]

                        if (
                            candidate_since is not None
                            and now - candidate_since >= required_hold
                        ):
                            active_reaction = candidate_reaction
                            active_last_seen = now
                            candidate_reaction = None
                            candidate_since = None

                # ------------------------------------------------
                # Manual test override
                # ------------------------------------------------

                if now >= manual_until:
                    manual_reaction = None

                display_reaction = (
                    manual_reaction
                    if manual_reaction
                    else active_reaction
                )

                # ------------------------------------------------
                # Restart GIF when reaction changes
                # ------------------------------------------------

                if display_reaction != previous_display_reaction:
                    if previous_display_reaction in reaction_players:
                        reaction_players[
                            previous_display_reaction
                        ].stop()

                    if display_reaction in reaction_players:
                        reaction_players[
                            display_reaction
                        ].start(now)

                    previous_display_reaction = display_reaction

                # ------------------------------------------------
                # Debug landmarks
                # ------------------------------------------------

                face_detected = face_landmarks is not None

                pose_detected = bool(
                    pose_result
                    and pose_result.pose_landmarks
                )

                if debug_mode and draw_landmarks:
                    if face_landmarks:
                        for landmark in face_landmarks[::FACE_LANDMARK_DRAW_STEP]:
                            x = int(landmark.x * OUTPUT_WIDTH)
                            y = int(landmark.y * OUTPUT_HEIGHT)

                            cv2.circle(
                                frame,
                                (x, y),
                                1,
                                (65, 230, 95),
                                -1,
                                cv2.LINE_AA,
                            )

                    if pose_result is not None:
                        draw_pose(frame, pose_result)

                    if hand_result is not None:
                        draw_hands(frame, hand_result)

                # ------------------------------------------------
                # Reaction media
                # ------------------------------------------------

                if display_reaction and face_landmarks:
                    player = reaction_players[display_reaction]

                    if player.available():
                        frame = overlay_reaction(
                            frame,
                            player.get_frame(now),
                            face_landmarks,
                            REACTIONS[
                                display_reaction
                            ]["overlay_scale"],
                        )

                # ------------------------------------------------
                # FPS
                # ------------------------------------------------

                fps_frame_count += 1
                fps_elapsed = now - fps_last_time

                if fps_elapsed >= 0.5:
                    fps = fps_frame_count / fps_elapsed
                    fps_frame_count = 0
                    fps_last_time = now

                # ------------------------------------------------
                # Side panel
                # ------------------------------------------------

                if debug_mode:
                    draw_debug_panel(
                        frame,
                        face_detected,
                        pose_detected,
                        hand_data,
                        detected_reaction,
                        candidate_reaction,
                        display_reaction,
                        face_data,
                        gesture_debug,
                        fps,
                        performance,
                    )

                cv2.imshow(
                    "MemeMorph",
                    frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key == ord("d"):
                    debug_mode = not debug_mode

                if key == ord("l"):
                    draw_landmarks = not draw_landmarks

                if key in MANUAL_KEYS:
                    manual_reaction = MANUAL_KEYS[key]
                    manual_until = (
                        time.perf_counter()
                        + MANUAL_REACTION_SECONDS
                    )

                    reaction_players[
                        manual_reaction
                    ].start(
                        time.perf_counter()
                    )

    finally:
        pose_tracker.close()
        hand_tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
