import sys
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


# Make src/ importable when running:
#   python -m unittest discover -s tests -v
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import main as mm
import reaction_engine as re


def make_face(**overrides):
    """
    Build a complete face-metric dictionary with safe neutral defaults.
    """
    face = {
        "blink": 0.05,
        "squint": 0.05,
        "wide": 0.05,
        "brow_down": 0.05,
        "brow_inner_up": 0.05,
        "brow_outer_left": 0.05,
        "brow_outer_right": 0.05,
        "brow_asymmetry": 0.00,
        "jaw_open": 0.05,
        "mouth_smile": 0.05,
        "mouth_frown": 0.05,
        "mouth_pucker": 0.05,
        "mouth_funnel": 0.05,
        "mouth_press": 0.05,
        "mouth_roll": 0.05,
        "look_left": 0.05,
        "look_right": 0.05,
        "side_eye": 0.05,
    }

    face.update(overrides)
    return face


def make_delta(**overrides):
    """
    Build a complete calibrated-delta dictionary.
    """
    delta = {
        key: 0.0
        for key in mm.CALIBRATION_KEYS
    }

    delta.update(overrides)
    return delta


def landmark(x=0.0, y=0.0):
    return SimpleNamespace(
        x=float(x),
        y=float(y),
    )


def make_pose_result(
    left_shoulder=(0.40, 0.50),
    right_shoulder=(0.60, 0.50),
):
    """
    Create a minimal fake MediaPipe pose result.
    Only shoulder indexes 11 and 12 matter for Absolute Cinema.
    """
    pose = [
        landmark(0.50, 0.50)
        for _ in range(33)
    ]

    pose[11] = landmark(*left_shoulder)
    pose[12] = landmark(*right_shoulder)

    return SimpleNamespace(
        pose_landmarks=[pose]
    )


def make_face_landmarks():
    """
    Create a simple normalized face rectangle.
    """
    return [
        landmark(0.40, 0.30),
        landmark(0.60, 0.30),
        landmark(0.60, 0.60),
        landmark(0.40, 0.60),
        landmark(0.50, 0.45),
    ]


def make_hand_landmarks_for_thumbs_up():
    """
    Synthetic 21-point hand:
      - thumb points upward
      - index/middle/ring/pinky are folded
    """
    points = [
        landmark(0.50, 0.60)
        for _ in range(21)
    ]

    # Wrist
    points[0] = landmark(0.50, 0.60)

    # Thumb chain
    points[2] = landmark(0.50, 0.54)
    points[3] = landmark(0.50, 0.50)
    points[4] = landmark(0.50, 0.38)

    # Middle MCP establishes palm scale.
    points[9] = landmark(0.50, 0.50)

    # Keep non-thumb fingertips close to their PIP joints,
    # causing finger_extended() to return False.
    for mcp, pip, tip in [
        (5, 6, 8),
        (9, 10, 12),
        (13, 14, 16),
        (17, 18, 20),
    ]:
        points[mcp] = landmark(0.50, 0.50)
        points[pip] = landmark(0.51, 0.52)
        points[tip] = landmark(0.51, 0.52)

    return points


def make_hand_landmarks_for_thumb_point():
    """
    Synthetic 21-point hand:
      - thumb points sideways
      - other fingers are folded
    """
    points = [
        landmark(0.50, 0.60)
        for _ in range(21)
    ]

    points[0] = landmark(0.50, 0.60)

    # Thumb extends strongly to the right.
    points[2] = landmark(0.56, 0.59)
    points[3] = landmark(0.62, 0.59)
    points[4] = landmark(0.72, 0.59)

    # Palm scale reference.
    points[9] = landmark(0.50, 0.50)

    for mcp, pip, tip in [
        (5, 6, 8),
        (9, 10, 12),
        (13, 14, 16),
        (17, 18, 20),
    ]:
        points[mcp] = landmark(0.50, 0.50)
        points[pip] = landmark(0.51, 0.52)
        points[tip] = landmark(0.51, 0.52)

    return points


class TestCalibration(unittest.TestCase):

    def test_neutral_baseline_uses_median(self):
        samples = []

        for value in [0.10, 0.11, 0.12, 0.90, 0.13]:
            sample = make_face(
                squint=value,
                brow_down=value,
            )

            samples.append(sample)

        baseline = mm.build_neutral_baseline(
            samples
        )

        # The outlier 0.90 must not drag the baseline upward.
        self.assertAlmostEqual(
            baseline["squint"],
            0.12,
            places=6,
        )

        self.assertAlmostEqual(
            baseline["brow_down"],
            0.12,
            places=6,
        )

    def test_calculate_face_delta(self):
        baseline = make_face(
            squint=0.20,
            brow_down=0.10,
            jaw_open=0.05,
        )

        current = make_face(
            squint=0.55,
            brow_down=0.30,
            jaw_open=0.07,
        )

        delta = mm.calculate_face_delta(
            current,
            baseline,
        )

        self.assertAlmostEqual(
            delta["squint"],
            0.35,
            places=6,
        )

        self.assertAlmostEqual(
            delta["brow_down"],
            0.20,
            places=6,
        )

        self.assertAlmostEqual(
            delta["jaw_open"],
            0.02,
            places=6,
        )

    def test_delta_without_baseline_is_zero(self):
        delta = mm.calculate_face_delta(
            make_face(squint=0.90),
            None,
        )

        self.assertTrue(
            all(
                value == 0.0
                for value in delta.values()
            )
        )


class TestFaceReactions(unittest.TestCase):

    def test_speed_detects_reverse_smile(self):
        face = make_face(
            jaw_open=0.02,
        )

        delta = make_delta(
            squint=0.30,
            brow_down=0.35,
        )

        self.assertTrue(
            re.speed_condition(
                face,
                delta,
            )
        )

    def test_speed_rejects_neutral_face(self):
        face = make_face(
            jaw_open=0.02,
        )

        delta = make_delta(
            squint=0.02,
            blink=0.03,
            brow_down=0.03,
        )

        self.assertFalse(
            re.speed_condition(
                face,
                delta,
            )
        )

    def test_eyebrow_raise_detected(self):
        face = make_face(
            brow_asymmetry=0.24,
            jaw_open=0.02,
        )

        delta = make_delta(
            brow_asymmetry=0.20,
        )

        self.assertTrue(
            re.eyebrow_condition(
                face,
                delta,
            )
        )

    def test_eyebrow_raise_rejects_small_asymmetry(self):
        face = make_face(
            brow_asymmetry=0.08,
            jaw_open=0.02,
        )

        delta = make_delta(
            brow_asymmetry=0.05,
        )

        self.assertFalse(
            re.eyebrow_condition(
                face,
                delta,
            )
        )

    def test_surprised_detected(self):
        face = make_face(
            wide=0.45,
            jaw_open=0.55,
        )

        self.assertTrue(
            re.surprised_condition(face)
        )

    def test_surprised_requires_wide_eyes_and_open_jaw(self):
        not_wide = make_face(
            wide=0.10,
            jaw_open=0.60,
        )

        mouth_closed = make_face(
            wide=0.50,
            jaw_open=0.10,
        )

        self.assertFalse(
            re.surprised_condition(
                not_wide
            )
        )

        self.assertFalse(
            re.surprised_condition(
                mouth_closed
            )
        )

    def test_sad_detected(self):
        face = make_face(
            brow_inner_up=0.40,
            mouth_frown=0.35,
            mouth_smile=0.05,
            jaw_open=0.10,
        )

        self.assertTrue(
            re.sad_condition(face)
        )

    def test_sad_rejects_smile(self):
        face = make_face(
            brow_inner_up=0.40,
            mouth_frown=0.35,
            mouth_smile=0.60,
            jaw_open=0.10,
        )

        self.assertFalse(
            re.sad_condition(face)
        )

    def test_side_eye_detected(self):
        face = make_face(
            side_eye=0.60,
            blink=0.10,
            jaw_open=0.05,
        )

        self.assertTrue(
            re.side_eye_condition(face)
        )

    def test_side_eye_rejects_closed_eyes(self):
        face = make_face(
            side_eye=0.60,
            blink=0.80,
            jaw_open=0.05,
        )

        self.assertFalse(
            re.side_eye_condition(face)
        )

    def test_jerry_laugh_detected(self):
        face = make_face(
            mouth_smile=0.60,
            jaw_open=0.45,
            squint=0.40,
        )

        self.assertTrue(
            re.jerry_laugh_condition(
                face
            )
        )

    def test_jerry_laugh_rejects_closed_mouth(self):
        face = make_face(
            mouth_smile=0.60,
            jaw_open=0.10,
            squint=0.50,
        )

        self.assertFalse(
            re.jerry_laugh_condition(
                face
            )
        )


class TestHandReactions(unittest.TestCase):

    def test_thumb_point_geometry(self):
        hand = (
            make_hand_landmarks_for_thumb_point()
        )

        self.assertTrue(
            mm.is_thumb_point(hand)
        )

    def test_thumbs_up_geometry(self):
        hand = (
            make_hand_landmarks_for_thumbs_up()
        )

        self.assertTrue(
            mm.is_thumbs_up(hand)
        )

    def test_jerry_point_requires_thumb_and_laugh(self):
        face = make_face(
            mouth_smile=0.45,
            jaw_open=0.30,
            squint=0.30,
        )

        hand_data = {
            "thumb_point": True,
        }

        self.assertTrue(
            re.jerry_point_condition(
                face,
                hand_data,
            )
        )

        hand_data["thumb_point"] = False

        self.assertFalse(
            re.jerry_point_condition(
                face,
                hand_data,
            )
        )

    def test_thumbs_up_condition(self):
        self.assertTrue(
            re.thumbs_up_condition(
                {"thumbs_up": True}
            )
        )

        self.assertFalse(
            re.thumbs_up_condition(
                {"thumbs_up": False}
            )
        )


class TestBodyReactions(unittest.TestCase):

    def test_facepalm_detected(self):
        face_landmarks = (
            make_face_landmarks()
        )

        hand_data = {
            "palms": [
                {
                    "x": 0.50,
                    "y": 0.385,
                    "open": True,
                }
            ]
        }

        self.assertTrue(
            re.detect_facepalm(
                face_landmarks,
                hand_data,
            )
        )

    def test_facepalm_rejects_closed_hand(self):
        face_landmarks = (
            make_face_landmarks()
        )

        hand_data = {
            "palms": [
                {
                    "x": 0.50,
                    "y": 0.385,
                    "open": False,
                }
            ]
        }

        self.assertFalse(
            re.detect_facepalm(
                face_landmarks,
                hand_data,
            )
        )

    def test_absolute_cinema_detected(self):
        pose_result = (
            make_pose_result()
        )

        hand_data = {
            "count": 2,
            "open_hands": 2,
            "palms": [
                {
                    "x": 0.20,
                    "y": 0.38,
                    "open": True,
                },
                {
                    "x": 0.80,
                    "y": 0.38,
                    "open": True,
                },
            ],
        }

        self.assertTrue(
            re.detect_absolute_cinema(
                pose_result,
                hand_data,
            )
        )

    def test_absolute_cinema_requires_two_open_hands(self):
        pose_result = (
            make_pose_result()
        )

        hand_data = {
            "count": 2,
            "open_hands": 1,
            "palms": [
                {
                    "x": 0.20,
                    "y": 0.38,
                    "open": True,
                },
                {
                    "x": 0.80,
                    "y": 0.38,
                    "open": False,
                },
            ],
        }

        self.assertFalse(
            re.detect_absolute_cinema(
                pose_result,
                hand_data,
            )
        )


class TestReactionPriority(unittest.TestCase):

    def setUp(self):
        self.face = make_face()
        self.delta = make_delta()

        self.hand_data = {
            "count": 0,
            "open_hands": 0,
            "thumb_point": False,
            "thumbs_up": False,
            "palms": [],
        }

    def test_absolute_cinema_has_highest_priority(self):
        """
        Even if every other detector also reports True,
        Absolute Cinema must win.
        """
        with (
            patch.object(
                re,
                "detect_absolute_cinema",
                return_value=True,
            ),
            patch.object(
                re,
                "detect_facepalm",
                return_value=True,
            ),
            patch.object(
                re,
                "jerry_point_condition",
                return_value=True,
            ),
            patch.object(
                re,
                "thumbs_up_condition",
                return_value=True,
            ),
            patch.object(
                re,
                "speed_condition",
                return_value=True,
            ),
            patch.object(
                re,
                "eyebrow_condition",
                return_value=True,
            ),
            patch.object(
                re,
                "surprised_condition",
                return_value=True,
            ),
        ):
            reaction = re.choose_reaction(
                self.face,
                self.delta,
                None,
                None,
                self.hand_data,
            )

        self.assertEqual(
            reaction,
            "absolute_cinema",
        )

    def test_facepalm_beats_face_only_reactions(self):
        with (
            patch.object(
                re,
                "detect_absolute_cinema",
                return_value=False,
            ),
            patch.object(
                re,
                "detect_facepalm",
                return_value=True,
            ),
            patch.object(
                re,
                "speed_condition",
                return_value=True,
            ),
            patch.object(
                re,
                "eyebrow_condition",
                return_value=True,
            ),
            patch.object(
                re,
                "surprised_condition",
                return_value=True,
            ),
        ):
            reaction = re.choose_reaction(
                self.face,
                self.delta,
                None,
                None,
                self.hand_data,
            )

        self.assertEqual(
            reaction,
            "facepalm",
        )

    def test_jerry_point_beats_thumbs_up_and_face_reactions(self):
        with (
            patch.object(
                re,
                "detect_absolute_cinema",
                return_value=False,
            ),
            patch.object(
                re,
                "detect_facepalm",
                return_value=False,
            ),
            patch.object(
                re,
                "jerry_point_condition",
                return_value=True,
            ),
            patch.object(
                re,
                "thumbs_up_condition",
                return_value=True,
            ),
            patch.object(
                re,
                "speed_condition",
                return_value=True,
            ),
        ):
            reaction = re.choose_reaction(
                self.face,
                self.delta,
                None,
                None,
                self.hand_data,
            )

        self.assertEqual(
            reaction,
            "jerry_point",
        )

    def test_no_match_returns_none(self):
        reaction = re.choose_reaction(
            self.face,
            self.delta,
            None,
            None,
            self.hand_data,
        )

        self.assertIsNone(
            reaction
        )


class TestReactionArchitecture(unittest.TestCase):

    def test_every_priority_reaction_has_detector(self):
        self.assertEqual(
            set(re.REACTION_PRIORITY),
            set(re.REACTION_DETECTORS),
        )

    def test_every_reaction_has_metadata_and_runtime_config(self):
        self.assertEqual(
            set(re.REACTION_METADATA),
            set(re.REACTIONS),
        )

        self.assertEqual(
            set(re.REACTION_PRIORITY),
            set(re.REACTIONS),
        )

    def test_disabled_reaction_is_skipped(self):
        original = mm.load_local_config()
        modified = copy.deepcopy(original)

        modified["reactions"]["speed"]["enabled"] = False

        re.configure_reaction_engine(
            modified
        )

        try:
            face = make_face(
                jaw_open=0.02,
            )

            delta = make_delta(
                squint=0.40,
                brow_down=0.40,
            )

            reaction = re.choose_reaction(
                face,
                delta,
                None,
                None,
                {
                    "count": 0,
                    "open_hands": 0,
                    "thumb_point": False,
                    "thumbs_up": False,
                    "palms": [],
                },
            )

            self.assertNotEqual(
                reaction,
                "speed",
            )

        finally:
            re.configure_reaction_engine(
                original
            )


if __name__ == "__main__":
    unittest.main()
