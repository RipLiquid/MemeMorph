import cv2
import mediapipe as mp


# MediaPipe pose indexes that matter for arms
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_ELBOW = 13
RIGHT_ELBOW = 14

LEFT_WRIST = 15
RIGHT_WRIST = 16


# Connections we want to draw
ARM_CONNECTIONS = [
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),

    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),

    (LEFT_SHOULDER, RIGHT_SHOULDER),
]


class PoseTracker:

    def __init__(self, model_path):

        BaseOptions = mp.tasks.BaseOptions

        PoseLandmarker = (
            mp.tasks.vision.PoseLandmarker
        )

        PoseLandmarkerOptions = (
            mp.tasks.vision.PoseLandmarkerOptions
        )

        RunningMode = (
            mp.tasks.vision.RunningMode
        )

        options = PoseLandmarkerOptions(

            base_options=BaseOptions(
                model_asset_path=str(model_path)
            ),

            running_mode=RunningMode.VIDEO,

            num_poses=1,

            min_pose_detection_confidence=0.5,

            min_pose_presence_confidence=0.5,

            min_tracking_confidence=0.5,

            output_segmentation_masks=False,
        )

        self.landmarker = (
            PoseLandmarker.create_from_options(
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


def landmark_visible(
    landmark,
    threshold=0.5
):

    visibility = getattr(
        landmark,
        "visibility",
        None
    )

    presence = getattr(
        landmark,
        "presence",
        None
    )

    if (
        visibility is not None
        and visibility < threshold
    ):
        return False

    if (
        presence is not None
        and presence < threshold
    ):
        return False

    return True


def draw_pose(
    frame,
    pose_result
):

    if not pose_result.pose_landmarks:
        return False

    landmarks = (
        pose_result.pose_landmarks[0]
    )

    height, width = frame.shape[:2]

    # --------------------------------------------
    # Draw shoulder / elbow / wrist connections
    # --------------------------------------------

    for start_index, end_index in ARM_CONNECTIONS:

        start = landmarks[start_index]
        end = landmarks[end_index]

        if not (
            landmark_visible(start)
            and
            landmark_visible(end)
        ):
            continue

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
            (255, 0, 255),
            3
        )

    # --------------------------------------------
    # Draw important joints
    # --------------------------------------------

    important_landmarks = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_ELBOW,
        RIGHT_ELBOW,
        LEFT_WRIST,
        RIGHT_WRIST,
    ]

    for index in important_landmarks:

        landmark = landmarks[index]

        if not landmark_visible(
            landmark
        ):
            continue

        point = (
            int(landmark.x * width),
            int(landmark.y * height)
        )

        cv2.circle(
            frame,
            point,
            7,
            (255, 0, 255),
            -1
        )

    return True