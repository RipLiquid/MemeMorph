from app_config import load_local_config


# ============================================================
# REACTION METADATA / PRIORITY
# ============================================================

REACTION_METADATA = {
    "speed": {
        "label": "Speed Reverse Smile",
        "file": "speed_reverse_smile.gif",
    },
    "eyebrow": {
        "label": "Eyebrow Raise",
        "file": "rock_eyebrow.gif",
    },
    "surprised": {
        "label": "Surprised",
        "file": "surprised_pikachu.gif",
    },
    "sad": {
        "label": "Sad / Cry",
        "file": "sad_cry.gif",
    },
    "side_eye": {
        "label": "Side Eye",
        "file": "side_eye.gif",
    },
    "jerry_laugh": {
        "label": "Jerry Laugh",
        "file": "mocking_tom_jerry.gif",
    },
    "jerry_point": {
        "label": "Jerry Point & Laugh",
        "file": "laughing.gif",
    },
    "facepalm": {
        "label": "Facepalm",
        "file": "hands_on_head.gif",
    },
    "thumbs_up": {
        "label": "Thumbs Up",
        "file": "thumbs_up.gif",
    },
    "absolute_cinema": {
        "label": "Absolute Cinema",
        "file": "absolute_cinema.gif",
    },
}


# Highest-priority reactions come first.
REACTION_PRIORITY = [
    "absolute_cinema",
    "facepalm",
    "jerry_point",
    "thumbs_up",
    "speed",
    "eyebrow",
    "surprised",
    "sad",
    "side_eye",
    "jerry_laugh",
]


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
# CONFIGURATION
# ============================================================

_RUNTIME_CONFIG = load_local_config()
REACTIONS = {}


def build_reactions(config):
    reactions = {}

    for name, metadata in REACTION_METADATA.items():
        settings = config["reactions"][name]

        reactions[name] = {
            **metadata,
            "overlay_scale": float(
                settings["overlay_scale"]
            ),
            "hold": float(
                settings["hold"]
            ),
            "release": float(
                settings["release"]
            ),
        }

    return reactions


def configure_reaction_engine(config):
    """
    Apply the final local/Azure configuration.

    REACTIONS is mutated in place rather than reassigned so modules that
    imported the dictionary keep seeing the current values.
    """
    _RUNTIME_CONFIG.clear()
    _RUNTIME_CONFIG.update(config)

    REACTIONS.clear()
    REACTIONS.update(
        build_reactions(
            _RUNTIME_CONFIG
        )
    )


def reaction_setting(
    reaction_name,
    setting_name,
):
    return _RUNTIME_CONFIG[
        "reactions"
    ][reaction_name][setting_name]


def reaction_enabled(reaction_name):
    return bool(
        reaction_setting(
            reaction_name,
            "enabled",
        )
    )


configure_reaction_engine(
    _RUNTIME_CONFIG.copy()
)


# ============================================================
# SMALL GEOMETRY HELPERS
# ============================================================

def _average(a, b):
    return (a + b) / 2.0


def _distance_points(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]

    return (
        dx * dx
        + dy * dy
    ) ** 0.5


def _get_face_bounds_normalized(
    face_landmarks,
):
    xs = [
        landmark.x
        for landmark in face_landmarks
    ]

    ys = [
        landmark.y
        for landmark in face_landmarks
    ]

    return (
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    )


# ============================================================
# BODY / HAND REACTIONS
# ============================================================

def detect_facepalm(
    face_landmarks,
    hand_data,
):
    """
    Open palm overlapping the upper part of the face.
    """
    if (
        not face_landmarks
        or not hand_data["palms"]
    ):
        return False

    (
        face_x1,
        face_y1,
        face_x2,
        face_y2,
    ) = _get_face_bounds_normalized(
        face_landmarks
    )

    face_width = max(
        face_x2 - face_x1,
        0.01,
    )

    face_height = max(
        face_y2 - face_y1,
        0.01,
    )

    target = (
        (face_x1 + face_x2) / 2.0,
        face_y1
        + face_height * 0.28,
    )

    max_distance = max(
        face_width,
        face_height,
    ) * 0.62

    for palm in hand_data["palms"]:
        if not palm["open"]:
            continue

        inside_x = (
            face_x1
            - face_width * 0.28
            <= palm["x"]
            <= face_x2
            + face_width * 0.28
        )

        inside_y = (
            face_y1
            - face_height * 0.30
            <= palm["y"]
            <= face_y1
            + face_height * 0.72
        )

        close_enough = (
            _distance_points(
                (
                    palm["x"],
                    palm["y"],
                ),
                target,
            )
            < max_distance
        )

        if (
            inside_x
            and inside_y
            and close_enough
        ):
            return True

    return False


def detect_absolute_cinema(
    pose_result,
    hand_data,
):
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

    shoulder_y = _average(
        left_shoulder.y,
        right_shoulder.y,
    )

    shoulder_width = max(
        abs(
            left_shoulder.x
            - right_shoulder.x
        ),
        0.08,
    )

    palms = hand_data["palms"]

    raised_count = sum(
        1
        for palm in palms
        if palm["y"]
        < shoulder_y
        + float(
            reaction_setting(
                "absolute_cinema",
                "shoulder_height_margin",
            )
        )
    )

    palm_xs = [
        palm["x"]
        for palm in palms
    ]

    hand_spread = (
        max(palm_xs)
        - min(palm_xs)
    )

    return (
        raised_count >= 2
        and hand_spread
        > shoulder_width
        * float(
            reaction_setting(
                "absolute_cinema",
                "hand_spread_multiplier",
            )
        )
    )


# ============================================================
# FACE REACTION CONDITIONS
# ============================================================

def speed_condition(
    face,
    face_delta,
):
    eye_change = (
        face_delta["squint"]
        > float(
            reaction_setting(
                "speed",
                "squint_delta_min",
            )
        )
        or
        face_delta["blink"]
        > float(
            reaction_setting(
                "speed",
                "blink_delta_min",
            )
        )
    )

    brow_change = (
        face_delta["brow_down"]
        > float(
            reaction_setting(
                "speed",
                "brow_down_delta_min",
            )
        )
    )

    mouth_closed = (
        face["jaw_open"]
        < float(
            reaction_setting(
                "speed",
                "jaw_open_max",
            )
        )
    )

    return (
        eye_change
        and brow_change
        and mouth_closed
    )


def eyebrow_condition(
    face,
    face_delta,
):
    return (
        face_delta["brow_asymmetry"]
        > float(
            reaction_setting(
                "eyebrow",
                "brow_asymmetry_delta_min",
            )
        )
        and face["brow_asymmetry"]
        > float(
            reaction_setting(
                "eyebrow",
                "brow_asymmetry_raw_min",
            )
        )
        and face["jaw_open"]
        < float(
            reaction_setting(
                "eyebrow",
                "jaw_open_max",
            )
        )
    )


def surprised_condition(face):
    return (
        face["wide"]
        > float(
            reaction_setting(
                "surprised",
                "wide_min",
            )
        )
        and face["jaw_open"]
        > float(
            reaction_setting(
                "surprised",
                "jaw_open_min",
            )
        )
    )


def sad_condition(face):
    return (
        face["brow_inner_up"]
        > float(
            reaction_setting(
                "sad",
                "brow_inner_up_min",
            )
        )
        and face["mouth_frown"]
        > float(
            reaction_setting(
                "sad",
                "mouth_frown_min",
            )
        )
        and face["mouth_smile"]
        < float(
            reaction_setting(
                "sad",
                "mouth_smile_max",
            )
        )
        and face["jaw_open"]
        < float(
            reaction_setting(
                "sad",
                "jaw_open_max",
            )
        )
    )


def side_eye_condition(face):
    return (
        face["side_eye"]
        > float(
            reaction_setting(
                "side_eye",
                "side_eye_min",
            )
        )
        and face["blink"]
        < float(
            reaction_setting(
                "side_eye",
                "blink_max",
            )
        )
        and face["jaw_open"]
        < float(
            reaction_setting(
                "side_eye",
                "jaw_open_max",
            )
        )
    )


def jerry_laugh_condition(face):
    return (
        face["mouth_smile"]
        > float(
            reaction_setting(
                "jerry_laugh",
                "mouth_smile_min",
            )
        )
        and face["jaw_open"]
        > float(
            reaction_setting(
                "jerry_laugh",
                "jaw_open_min",
            )
        )
        and (
            face["squint"]
            > float(
                reaction_setting(
                    "jerry_laugh",
                    "squint_min",
                )
            )
            or face["blink"]
            > float(
                reaction_setting(
                    "jerry_laugh",
                    "blink_min",
                )
            )
        )
    )


def jerry_point_condition(
    face,
    hand_data,
):
    laughing_face = (
        (
            face["mouth_smile"]
            > float(
                reaction_setting(
                    "jerry_point",
                    "mouth_smile_min",
                )
            )
            or face["jaw_open"]
            > float(
                reaction_setting(
                    "jerry_point",
                    "jaw_open_min",
                )
            )
        )
        and (
            face["squint"]
            > float(
                reaction_setting(
                    "jerry_point",
                    "squint_min",
                )
            )
            or face["blink"]
            > float(
                reaction_setting(
                    "jerry_point",
                    "blink_min",
                )
            )
        )
    )

    return (
        hand_data["thumb_point"]
        and laughing_face
    )


def thumbs_up_condition(hand_data):
    return hand_data["thumbs_up"]


# ============================================================
# DETECTOR REGISTRY
# ============================================================

def _detect_absolute_cinema(context):
    return detect_absolute_cinema(
        context["pose_result"],
        context["hand_data"],
    )


def _detect_facepalm(context):
    return detect_facepalm(
        context["face_landmarks"],
        context["hand_data"],
    )


def _detect_jerry_point(context):
    return jerry_point_condition(
        context["face"],
        context["hand_data"],
    )


def _detect_thumbs_up(context):
    return thumbs_up_condition(
        context["hand_data"]
    )


def _detect_speed(context):
    return speed_condition(
        context["face"],
        context["face_delta"],
    )


def _detect_eyebrow(context):
    return eyebrow_condition(
        context["face"],
        context["face_delta"],
    )


def _detect_surprised(context):
    return surprised_condition(
        context["face"]
    )


def _detect_sad(context):
    return sad_condition(
        context["face"]
    )


def _detect_side_eye(context):
    return side_eye_condition(
        context["face"]
    )


def _detect_jerry_laugh(context):
    return jerry_laugh_condition(
        context["face"]
    )


REACTION_DETECTORS = {
    "absolute_cinema":
        _detect_absolute_cinema,
    "facepalm":
        _detect_facepalm,
    "jerry_point":
        _detect_jerry_point,
    "thumbs_up":
        _detect_thumbs_up,
    "speed":
        _detect_speed,
    "eyebrow":
        _detect_eyebrow,
    "surprised":
        _detect_surprised,
    "sad":
        _detect_sad,
    "side_eye":
        _detect_side_eye,
    "jerry_laugh":
        _detect_jerry_laugh,
}


def choose_reaction(
    face,
    face_delta,
    face_landmarks,
    pose_result,
    hand_data,
):
    """
    Evaluate registered reaction detectors by priority.

    Adding a future reaction now only requires:
      1. config/reaction_defaults.json settings
      2. REACTION_METADATA entry
      3. detector function + REACTION_DETECTORS entry
      4. a position in REACTION_PRIORITY
      5. its GIF/image asset
    """
    context = {
        "face": face,
        "face_delta": face_delta,
        "face_landmarks":
            face_landmarks,
        "pose_result":
            pose_result,
        "hand_data":
            hand_data,
    }

    for reaction_name in REACTION_PRIORITY:
        if not reaction_enabled(
            reaction_name
        ):
            continue

        detector = REACTION_DETECTORS.get(
            reaction_name
        )

        if (
            detector is not None
            and detector(context)
        ):
            return reaction_name

    return None
