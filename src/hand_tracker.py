import cv2
import mediapipe as mp


# ============================================================
# HAND CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [

    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (0, 17),
]


class HandTracker:

    def __init__(self, model_path):

        BaseOptions = mp.tasks.BaseOptions

        HandLandmarker = (
            mp.tasks.vision.HandLandmarker
        )

        HandLandmarkerOptions = (
            mp.tasks.vision.HandLandmarkerOptions
        )

        RunningMode = (
            mp.tasks.vision.RunningMode
        )

        options = HandLandmarkerOptions(

            base_options=BaseOptions(
                model_asset_path=str(model_path)
            ),

            running_mode=RunningMode.VIDEO,

            num_hands=2,

            min_hand_detection_confidence=0.5,

            min_hand_presence_confidence=0.5,

            min_tracking_confidence=0.5,
        )

        self.landmarker = (
            HandLandmarker.create_from_options(
                options
            )
        )

    def detect(
        self,
        mp_image,
        timestamp_ms
    ):

        return self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

    def close(self):
        self.landmarker.close()


def draw_hands(
    frame,
    hand_result
):

    if not hand_result.hand_landmarks:
        return 0

    height, width = frame.shape[:2]

    hand_count = 0

    for hand_index, landmarks in enumerate(
        hand_result.hand_landmarks
    ):

        hand_count += 1

        # --------------------------------------------
        # Draw connections
        # --------------------------------------------

        for start_index, end_index in HAND_CONNECTIONS:

            start = landmarks[
                start_index
            ]

            end = landmarks[
                end_index
            ]

            start_point = (
                int(start.x * width),
                int(start.y * height)
            )

            end_point = (
                int(end.x * width),
                int(end.y * height)
            )

            cv2.line(
                frame,
                start_point,
                end_point,
                (0, 165, 255),
                2
            )

        # --------------------------------------------
        # Draw landmarks
        # --------------------------------------------

        for landmark in landmarks:

            point = (
                int(landmark.x * width),
                int(landmark.y * height)
            )

            cv2.circle(
                frame,
                point,
                4,
                (0, 255, 255),
                -1
            )

        # --------------------------------------------
        # Left / Right label
        # --------------------------------------------

        if (
            hand_index
            < len(hand_result.handedness)
        ):

            handedness = (
                hand_result.handedness[
                    hand_index
                ]
            )

            if handedness:

                label = (
                    handedness[0]
                    .category_name
                )

                wrist = landmarks[0]

                x = int(
                    wrist.x * width
                )

                y = int(
                    wrist.y * height
                )

                cv2.putText(
                    frame,
                    label,
                    (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

    return hand_count