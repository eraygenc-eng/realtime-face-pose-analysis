import cv2
import mediapipe as mp

from config import (
    MAX_NUM_FACES,
    REFINE_LANDMARKS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    EYE_CLOSED_THRESHOLD,
)

from utils.geo import calculate_distance
from display.draw_text import draw_text


class FaceAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils

        self.draw_spec = self.mp_draw.DrawingSpec(
            thickness=1,
            circle_radius=1
        )

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=MAX_NUM_FACES,
            refine_landmarks=REFINE_LANDMARKS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.draw_spec,
                    connection_drawing_spec=self.draw_spec
                )

                landmarks = face_landmarks.landmark
                self.analyze_eyes(frame, landmarks)

        return frame

    def analyze_eyes(self, frame, landmarks):
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]

        right_eye_top = landmarks[386]
        right_eye_bottom = landmarks[374]

        left_eye_distance = calculate_distance(left_eye_top, left_eye_bottom)
        right_eye_distance = calculate_distance(right_eye_top, right_eye_bottom)

        avg_eye_distance = (left_eye_distance + right_eye_distance) / 2

        if avg_eye_distance < EYE_CLOSED_THRESHOLD:
            eye_status = "Eyes closed"
            color = (0, 0, 255)
        else:
            eye_status = "Eyes open"
            color = (0, 255, 0)

        draw_text(
            frame,
            eye_status,
            (400, 100),
            color=color,
            scale=1,
            thickness=2
        )

        draw_text(
            frame,
            f"Eye distance: {avg_eye_distance:.4f}",
            (400, 140),
            color=(0, 255, 0),
            scale=0.7,
            thickness=2
        )

    def close(self):
        self.face_mesh.close()