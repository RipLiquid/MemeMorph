import cv2
import mediapipe as mp
import numpy as np
import time

from pathlib import Path
from PIL import Image, ImageSequence


# ============================================================
# MEMEMORPH SETTINGS
# ============================================================

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# How long the expression must be held before triggering.
# This helps prevent ordinary blinks from immediately activating
# the reaction.
SPEED_HOLD_SECONDS = 0.20

# Keeps the reaction active briefly if detection drops for
# a couple of frames.
SPEED_RELEASE_GRACE = 0.20


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "face_landmarker.task"
)

SPEED_GIF_PATH = (
    PROJECT_ROOT
    / "assets"
    / "reactions"
    / "speed_reverse_smile.gif"
)


# ============================================================
# MEDIAPIPE FACE BLENDSHAPES
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
    """
    Loads an animated GIF into memory and returns the
    correct animation frame based on elapsed time.
    """

    def __init__(self, path):
        self.frames = []
        self.durations = []

        self.total_duration = 0.0
        self.started_at = None

        self.load(path)

    def load(self, path):
        if not path.exists():
            print()
            print("WARNING: Speed GIF was not found.")
            print(f"Expected location: {path}")
            print("Press G will not display anything until it is added.")
            print()
            return

        gif = Image.open(path)

        for frame in ImageSequence.Iterator(gif):
            rgba = frame.convert("RGBA")

            frame_array = np.array(rgba)

            # Pillow gives RGBA.
            # OpenCV expects BGRA.
            bgra = cv2.cvtColor(
                frame_array,
                cv2.COLOR_RGBA2BGRA
            )

            duration_ms = frame.info.get(
                "duration",
                gif.info.get("duration", 80)
            )

            # Prevent bad GIF metadata from creating
            # zero-length frames.
            duration_ms = max(
                duration_ms,
                20
            )

            self.frames.append(bgra)

            self.durations.append(
                duration_ms / 1000.0
            )

        self.total_duration = sum(
            self.durations
        )

        print(
            f"Loaded reaction GIF: "
            f"{len(self.frames)} frames"
        )

    def available(self):
        return len(self.frames) > 0

    def start(self, now=None):
        if now is None:
            now = time.perf_counter()

        self.started_at = now

    def stop(self):
        self.started_at = None

    def get_frame(self, now=None):
        if not self.available():
            return None

        if now is None:
            now = time.perf_counter()

        if self.started_at is None:
            self.start(now)

        elapsed = (
            now - self.started_at
        ) % self.total_duration

        accumulated_time = 0.0

        for index, duration in enumerate(
            self.durations
        ):
            accumulated_time += duration

            if elapsed <= accumulated_time:
                return self.frames[index]

        return self.frames[-1]


# ============================================================
# GENERAL HELPERS
# ============================================================

def get_value(values, name):
    return values.get(name, 0.0)


def average(a, b):
    return (a + b) / 2.0


# ============================================================
# CAMERA ASPECT RATIO
# ============================================================

def make_16_9(frame):
    """
    Crops the webcam image to 16:9 without stretching it,
    then resizes it to 1280x720.
    """

    height, width = frame.shape[:2]

    target_ratio = 16 / 9
    current_ratio = width / height

    if current_ratio > target_ratio:
        # Camera image is too wide.
        new_width = int(
            height * target_ratio
        )

        x_start = (
            width - new_width
        ) // 2

        frame = frame[
            :,
            x_start:x_start + new_width
        ]

    elif current_ratio < target_ratio:
        # Camera image is too tall.
        new_height = int(
            width / target_ratio
        )

        y_start = (
            height - new_height
        ) // 2

        frame = frame[
            y_start:y_start + new_height,
            :
        ]

    return cv2.resize(
        frame,
        (
            WINDOW_WIDTH,
            WINDOW_HEIGHT
        )
    )


# ============================================================
# FACE BOUNDING BOX
# ============================================================

def get_face_box(
    landmarks,
    frame_width,
    frame_height
):
    """
    Creates a box around the detected face.

    Extra padding is added so the reaction GIF covers
    the entire head instead of only the facial landmarks.
    """

    xs = [
        landmark.x * frame_width
        for landmark in landmarks
    ]

    ys = [
        landmark.y * frame_height
        for landmark in landmarks
    ]

    x1 = int(min(xs))
    x2 = int(max(xs))

    y1 = int(min(ys))
    y2 = int(max(ys))

    face_width = x2 - x1
    face_height = y2 - y1

    # Horizontal padding.
    x_padding = int(
        face_width * 0.30
    )

    # More padding above the face to include hair/head.
    y_padding_top = int(
        face_height * 0.40
    )

    y_padding_bottom = int(
        face_height * 0.18
    )

    x1 -= x_padding
    x2 += x_padding

    y1 -= y_padding_top
    y2 += y_padding_bottom

    # Keep the box inside the webcam image.
    x1 = max(
        0,
        x1
    )

    y1 = max(
        0,
        y1
    )

    x2 = min(
        frame_width,
        x2
    )

    y2 = min(
        frame_height,
        y2
    )

    return x1, y1, x2, y2


# ============================================================
# GIF OVERLAY
# ============================================================

def overlay_gif(
    frame,
    gif_frame,
    face_box
):
    """
    Resizes the current GIF frame and alpha-blends it
    over the detected face.
    """

    if gif_frame is None:
        return frame

    x1, y1, x2, y2 = face_box

    target_width = x2 - x1
    target_height = y2 - y1

    if (
        target_width <= 0
        or
        target_height <= 0
    ):
        return frame

    resized = cv2.resize(
        gif_frame,
        (
            target_width,
            target_height
        ),
        interpolation=cv2.INTER_LINEAR
    )

    gif_bgr = resized[:, :, :3]

    alpha = (
        resized[:, :, 3]
        .astype(np.float32)
        / 255.0
    )

    alpha = alpha[:, :, None]

    roi = frame[
        y1:y2,
        x1:x2
    ]

    blended = (
        gif_bgr * alpha
        +
        roi * (1.0 - alpha)
    )

    frame[
        y1:y2,
        x1:x2
    ] = blended.astype(
        np.uint8
    )

    return frame


# ============================================================
# FACIAL EXPRESSION ANALYSIS
# ============================================================

def analyze_expression(values):
    """
    Converts MediaPipe's raw blendshape values into
    simplified facial-expression measurements.
    """

    # --------------------------------------------------------
    # EYES
    # --------------------------------------------------------

    blink_left = get_value(
        values,
        "eyeBlinkLeft"
    )

    blink_right = get_value(
        values,
        "eyeBlinkRight"
    )

    squint_left = get_value(
        values,
        "eyeSquintLeft"
    )

    squint_right = get_value(
        values,
        "eyeSquintRight"
    )

    wide_left = get_value(
        values,
        "eyeWideLeft"
    )

    wide_right = get_value(
        values,
        "eyeWideRight"
    )

    blink_average = average(
        blink_left,
        blink_right
    )

    squint_average = average(
        squint_left,
        squint_right
    )

    wide_average = average(
        wide_left,
        wide_right
    )

    # --------------------------------------------------------
    # EYEBROWS
    # --------------------------------------------------------

    brow_down_left = get_value(
        values,
        "browDownLeft"
    )

    brow_down_right = get_value(
        values,
        "browDownRight"
    )

    brow_down_average = average(
        brow_down_left,
        brow_down_right
    )

    brow_inner_up = get_value(
        values,
        "browInnerUp"
    )

    # --------------------------------------------------------
    # MOUTH / LIPS
    # --------------------------------------------------------

    jaw_open = get_value(
        values,
        "jawOpen"
    )

    mouth_pucker = get_value(
        values,
        "mouthPucker"
    )

    mouth_funnel = get_value(
        values,
        "mouthFunnel"
    )

    mouth_press = average(
        get_value(
            values,
            "mouthPressLeft"
        ),
        get_value(
            values,
            "mouthPressRight"
        )
    )

    mouth_roll = average(
        get_value(
            values,
            "mouthRollLower"
        ),
        get_value(
            values,
            "mouthRollUpper"
        )
    )

    mouth_stretch = average(
        get_value(
            values,
            "mouthStretchLeft"
        ),
        get_value(
            values,
            "mouthStretchRight"
        )
    )

    # --------------------------------------------------------
    # READABLE EYE STATE
    # --------------------------------------------------------

    if blink_average > 0.55:
        eye_state = "CLOSED"

    elif squint_average > 0.45:
        eye_state = "SQUINT"

    elif wide_average > 0.35:
        eye_state = "WIDE"

    else:
        eye_state = "OPEN"

    # --------------------------------------------------------
    # READABLE EYEBROW STATE
    # --------------------------------------------------------

    if brow_down_average > 0.35:
        brow_state = "DOWN"

    elif brow_inner_up > 0.35:
        brow_state = "UP"

    else:
        brow_state = "NEUTRAL"

    # ========================================================
    # SPEED REACTION DETECTOR
    # ========================================================

    # IMPORTANT:
    #
    # The Speed expression now accepts:
    #
    #     eyes CLOSED
    #          OR
    #     eyes SQUINTED
    #
    # It does NOT require both at the same time.
    #
    # Eyebrow and lip checks are still required to
    # prevent a normal blink from triggering the reaction.

    eyes_match = (
        blink_average > 0.50
        or
        squint_average > 0.45
    )

    brows_match = (
        brow_down_average > 0.40
    )

    lips_match = (
        mouth_pucker > 0.07
        or
        mouth_funnel > 0.12
        or
        mouth_press > 0.20
        or
        mouth_roll > 0.20
    )

    speed_candidate = (
        eyes_match
        and
        brows_match
        and
        lips_match
    )

    return {
        "eye_state": eye_state,
        "brow_state": brow_state,

        "blink": blink_average,
        "squint": squint_average,
        "wide": wide_average,

        "brow_down": brow_down_average,
        "brow_inner_up": brow_inner_up,

        "jaw_open": jaw_open,

        "mouth_pucker": mouth_pucker,
        "mouth_funnel": mouth_funnel,
        "mouth_press": mouth_press,
        "mouth_roll": mouth_roll,
        "mouth_stretch": mouth_stretch,

        "eyes_match": eyes_match,
        "brows_match": brows_match,
        "lips_match": lips_match,

        "speed_candidate": speed_candidate,
    }


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    # --------------------------------------------------------
    # Required MediaPipe model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        print()
        print("ERROR: MediaPipe model was not found.")
        print(f"Expected location: {MODEL_PATH}")
        print()

        return

    # --------------------------------------------------------
    # Load reaction GIF
    # --------------------------------------------------------

    speed_gif = GifPlayer(
        SPEED_GIF_PATH
    )

    # --------------------------------------------------------
    # MediaPipe setup
    # --------------------------------------------------------

    BaseOptions = (
        mp.tasks.BaseOptions
    )

    FaceLandmarker = (
        mp.tasks.vision.FaceLandmarker
    )

    FaceLandmarkerOptions = (
        mp.tasks.vision.FaceLandmarkerOptions
    )

    RunningMode = (
        mp.tasks.vision.RunningMode
    )

    options = FaceLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=str(
                MODEL_PATH
            )
        ),

        running_mode=RunningMode.VIDEO,

        num_faces=1,

        output_face_blendshapes=True,

        output_facial_transformation_matrixes=True,

        min_face_detection_confidence=0.5,

        min_face_presence_confidence=0.5,

        min_tracking_confidence=0.5,
    )

    # --------------------------------------------------------
    # Webcam setup
    # --------------------------------------------------------

    camera = cv2.VideoCapture(0)

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        WINDOW_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        WINDOW_HEIGHT
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        30
    )

    if not camera.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        return

    # --------------------------------------------------------
    # OpenCV window
    # --------------------------------------------------------

    cv2.namedWindow(
        "MemeMorph",
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        "MemeMorph",
        WINDOW_WIDTH,
        WINDOW_HEIGHT
    )

    print()
    print("==============================")
    print("          MemeMorph")
    print("==============================")
    print()
    print("Face tracking: ON")
    print("Expression tracking: ON")
    print("Resolution: 1280x720")
    print()
    print("CONTROLS")
    print("------------------------------")
    print("Q = Quit")
    print("D = Toggle debug view")
    print("G = Manually test Speed GIF")
    print()

    # --------------------------------------------------------
    # Runtime state
    # --------------------------------------------------------

    debug_mode = True

    speed_candidate_since = None

    speed_active = False

    previous_overlay_active = False

    last_speed_match = 0.0

    manual_gif_until = 0.0

    # --------------------------------------------------------
    # Start MediaPipe detector
    # --------------------------------------------------------

    with FaceLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:

            now = time.perf_counter()

            # ------------------------------------------------
            # Camera frame
            # ------------------------------------------------

            success, frame = (
                camera.read()
            )

            if not success:

                print(
                    "ERROR: Could not read webcam frame."
                )

                break

            # Mirror like a selfie camera.
            frame = cv2.flip(
                frame,
                1
            )

            # Convert to landscape 16:9.
            frame = make_16_9(
                frame
            )

            # ------------------------------------------------
            # Convert frame for MediaPipe
            # ------------------------------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(
                time.perf_counter()
                * 1000
            )

            result = (
                landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )
            )

            # ------------------------------------------------
            # Extract facial blendshapes
            # ------------------------------------------------

            blendshape_values = {}

            if result.face_blendshapes:

                for blendshape in (
                    result.face_blendshapes[0]
                ):

                    index = blendshape.index

                    if (
                        0
                        <= index
                        < len(BLENDSHAPE_NAMES)
                    ):

                        name = (
                            BLENDSHAPE_NAMES[
                                index
                            ]
                        )

                        blendshape_values[
                            name
                        ] = blendshape.score

            # ------------------------------------------------
            # Analyze expression
            # ------------------------------------------------

            expression = analyze_expression(
                blendshape_values
            )

            # ------------------------------------------------
            # SPEED REACTION SMOOTHING
            # ------------------------------------------------

            if expression[
                "speed_candidate"
            ]:

                last_speed_match = now

                if speed_candidate_since is None:
                    speed_candidate_since = now

                held_for = (
                    now
                    - speed_candidate_since
                )

                if (
                    held_for
                    >= SPEED_HOLD_SECONDS
                ):
                    speed_active = True

            else:

                speed_candidate_since = None

                if (
                    now
                    - last_speed_match
                    > SPEED_RELEASE_GRACE
                ):
                    speed_active = False

            # ------------------------------------------------
            # Manual GIF testing
            # ------------------------------------------------

            manual_active = (
                now < manual_gif_until
            )

            overlay_active = (
                speed_active
                or
                manual_active
            )

            # Restart the GIF whenever the reaction begins.
            if (
                overlay_active
                and
                not previous_overlay_active
            ):
                speed_gif.start(now)

            # Stop animation when the reaction ends.
            if (
                not overlay_active
                and
                previous_overlay_active
            ):
                speed_gif.stop()

            previous_overlay_active = (
                overlay_active
            )

            # ------------------------------------------------
            # Face detection
            # ------------------------------------------------

            face_detected = False

            if result.face_landmarks:

                face_detected = True

                landmarks = (
                    result.face_landmarks[0]
                )

                height, width = (
                    frame.shape[:2]
                )

                face_box = get_face_box(
                    landmarks,
                    width,
                    height
                )

                # --------------------------------------------
                # Reaction GIF
                # --------------------------------------------

                if (
                    overlay_active
                    and
                    speed_gif.available()
                ):

                    gif_frame = (
                        speed_gif.get_frame(
                            now
                        )
                    )

                    frame = overlay_gif(
                        frame,
                        gif_frame,
                        face_box
                    )

                # --------------------------------------------
                # Debug landmarks
                # --------------------------------------------

                elif debug_mode:

                    for landmark in landmarks:

                        x = int(
                            landmark.x
                            * width
                        )

                        y = int(
                            landmark.y
                            * height
                        )

                        cv2.circle(
                            frame,
                            (x, y),
                            1,
                            (0, 255, 0),
                            -1
                        )

            # =================================================
            # DEBUG PANEL
            # =================================================

            if debug_mode:

                overlay = frame.copy()

                cv2.rectangle(
                    overlay,
                    (20, 20),
                    (450, 690),
                    (0, 0, 0),
                    -1
                )

                frame = cv2.addWeighted(
                    overlay,
                    0.60,
                    frame,
                    0.40,
                    0
                )

                # --------------------------------------------
                # Title
                # --------------------------------------------

                cv2.putText(
                    frame,
                    "MemeMorph",
                    (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (255, 255, 255),
                    2
                )

                # --------------------------------------------
                # Face status
                # --------------------------------------------

                if face_detected:

                    face_text = (
                        "FACE DETECTED"
                    )

                    face_color = (
                        0,
                        255,
                        0
                    )

                else:

                    face_text = (
                        "NO FACE"
                    )

                    face_color = (
                        0,
                        0,
                        255
                    )

                cv2.putText(
                    frame,
                    face_text,
                    (40, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    face_color,
                    2
                )

                # --------------------------------------------
                # Speed state
                # --------------------------------------------

                if speed_active:

                    reaction_text = (
                        "SPEED: DETECTED!"
                    )

                    reaction_color = (
                        0,
                        255,
                        255
                    )

                elif expression[
                    "speed_candidate"
                ]:

                    reaction_text = (
                        "SPEED: HOLD..."
                    )

                    reaction_color = (
                        0,
                        200,
                        255
                    )

                else:

                    reaction_text = (
                        "SPEED: READY"
                    )

                    reaction_color = (
                        255,
                        255,
                        255
                    )

                cv2.putText(
                    frame,
                    reaction_text,
                    (40, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    reaction_color,
                    2
                )

                # --------------------------------------------
                # Main detected states
                # --------------------------------------------

                cv2.putText(
                    frame,
                    (
                        "EYES: "
                        + expression[
                            "eye_state"
                        ]
                    ),
                    (40, 185),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    (
                        "BROWS: "
                        + expression[
                            "brow_state"
                        ]
                    ),
                    (40, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2
                )

                # --------------------------------------------
                # Component matching
                # --------------------------------------------

                eyes_status = (
                    "YES"
                    if expression["eyes_match"]
                    else "NO"
                )

                brows_status = (
                    "YES"
                    if expression["brows_match"]
                    else "NO"
                )

                lips_status = (
                    "YES"
                    if expression["lips_match"]
                    else "NO"
                )

                cv2.putText(
                    frame,
                    f"Eyes Match: {eyes_status}",
                    (40, 260),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1
                )

                cv2.putText(
                    frame,
                    f"Brows Match: {brows_status}",
                    (40, 290),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1
                )

                cv2.putText(
                    frame,
                    f"Lips Match: {lips_status}",
                    (40, 320),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1
                )

                # --------------------------------------------
                # Raw values
                # --------------------------------------------

                debug_values = [
                    (
                        "eyeBlink",
                        expression["blink"]
                    ),
                    (
                        "eyeSquint",
                        expression["squint"]
                    ),
                    (
                        "eyeWide",
                        expression["wide"]
                    ),
                    (
                        "browDown",
                        expression["brow_down"]
                    ),
                    (
                        "browInnerUp",
                        expression["brow_inner_up"]
                    ),
                    (
                        "jawOpen",
                        expression["jaw_open"]
                    ),
                    (
                        "mouthPucker",
                        expression["mouth_pucker"]
                    ),
                    (
                        "mouthFunnel",
                        expression["mouth_funnel"]
                    ),
                    (
                        "mouthPress",
                        expression["mouth_press"]
                    ),
                    (
                        "mouthRoll",
                        expression["mouth_roll"]
                    ),
                    (
                        "mouthStretch",
                        expression["mouth_stretch"]
                    ),
                ]

                y_position = 365

                for name, value in debug_values:

                    cv2.putText(
                        frame,
                        f"{name}: {value:.2f}",
                        (
                            40,
                            y_position
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (255, 255, 255),
                        1
                    )

                    y_position += 27

            # =================================================
            # SHOW WINDOW
            # =================================================

            cv2.imshow(
                "MemeMorph",
                frame
            )

            # =================================================
            # KEYBOARD CONTROLS
            # =================================================

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # Quit.
            if key == ord("q"):
                break

            # Toggle debugging.
            elif key == ord("d"):

                debug_mode = (
                    not debug_mode
                )

            # Manually play the reaction GIF.
            elif key == ord("g"):

                manual_gif_until = (
                    time.perf_counter()
                    + 3.0
                )

    # ========================================================
    # CLEANUP
    # ========================================================

    camera.release()

    cv2.destroyAllWindows()


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()