import cv2
import mediapipe as mp

from config import (
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    POSE_MODEL_COMPLEXITY,
    JUMP_THRESHOLD,
    JUMP_RESET_OFFSET,
    ARM_STRAIGHT_THRESHOLD,
    ARM_BENT_THRESHOLD,
    SHOULDER_DIFF_THRESHOLD,
    HEAD_FORWARD_THRESHOLD,
)

from utils.geo import calculate_angle
from display.draw_text import draw_text


class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=POSE_MODEL_COMPLEXITY,
            enable_segmentation=False,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )

        self.jump_count = 0
        self.is_jumping = False
        self.baseline_hip_y = None

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

            landmarks = results.pose_landmarks.landmark
            self.analyze_pose(frame, landmarks)

        return frame

    def analyze_pose(self, frame, landmarks):
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]

        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]

        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]

        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]

        self.detect_jump(frame, left_hip, right_hip)
        self.detect_hand_raise(frame, left_wrist, right_wrist, left_shoulder, right_shoulder)
        self.analyze_arm_angles(
            frame,
            left_shoulder,
            left_elbow,
            left_wrist,
            right_shoulder,
            right_elbow,
            right_wrist
        )
        self.analyze_posture(frame, left_shoulder, right_shoulder, left_ear, right_ear)

    def detect_jump(self, frame, left_hip, right_hip):
        hip_center_y = (left_hip.y + right_hip.y) / 2

        if self.baseline_hip_y is None:
            self.baseline_hip_y = hip_center_y

        if hip_center_y < self.baseline_hip_y - JUMP_THRESHOLD and not self.is_jumping:
            self.is_jumping = True

        if hip_center_y > self.baseline_hip_y - JUMP_RESET_OFFSET and self.is_jumping:
            self.jump_count += 1
            self.is_jumping = False

        draw_text(
            frame,
            f"Jumps: {self.jump_count}",
            (30, 360),
            color=(255, 0, 255),
            scale=0.9
        )

        if self.is_jumping:
            draw_text(
                frame,
                "Jumping!",
                (30, 400),
                color=(0, 255, 255),
                scale=0.9
            )

    def detect_hand_raise(self, frame, left_wrist, right_wrist, left_shoulder, right_shoulder):
        if right_wrist.y < right_shoulder.y and left_wrist.y < left_shoulder.y:
            draw_text(frame, "BOTH HANDS UP", (60, 60), color=(0, 255, 0), scale=1)
        elif left_wrist.y < left_shoulder.y:
            draw_text(frame, "LEFT HAND UP", (60, 60), color=(0, 255, 0), scale=1)
        elif right_wrist.y < right_shoulder.y:
            draw_text(frame, "RIGHT HAND UP", (60, 60), color=(0, 255, 0), scale=1)

    def analyze_arm_angles(
        self,
        frame,
        left_shoulder,
        left_elbow,
        left_wrist,
        right_shoulder,
        right_elbow,
        right_wrist
    ):
        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        if left_arm_angle > ARM_STRAIGHT_THRESHOLD:
            draw_text(frame, "Left arm straight", (30, 270), color=(0, 255, 0))
        elif left_arm_angle < ARM_BENT_THRESHOLD:
            draw_text(frame, "Left arm bent", (30, 270), color=(0, 255, 255))

        if right_arm_angle > ARM_STRAIGHT_THRESHOLD:
            draw_text(frame, "Right arm straight", (30, 310), color=(0, 255, 0))
        elif right_arm_angle < ARM_BENT_THRESHOLD:
            draw_text(frame, "Right arm bent", (30, 310), color=(0, 255, 255))

    def analyze_posture(self, frame, left_shoulder, right_shoulder, left_ear, right_ear):
        ear_center_x = (left_ear.x + right_ear.x) / 2
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2

        shoulder_diff_y = abs(left_shoulder.y - right_shoulder.y)
        head_forward_diff = abs(ear_center_x - shoulder_center_x)

        posture_status = "Good posture"

        if shoulder_diff_y > SHOULDER_DIFF_THRESHOLD:
            posture_status = "Uneven shoulders"
        elif head_forward_diff > HEAD_FORWARD_THRESHOLD:
            posture_status = "Head shifted forward/side"

        color = (0, 255, 0) if posture_status == "Good posture" else (0, 0, 255)

        draw_text(
            frame,
            f"Posture: {posture_status}",
            (30, 470),
            color=color
        )

    def close(self):
        self.pose.close()