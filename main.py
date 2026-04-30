import cv2

from config import CAMERA_INDEX
from analyzers.pose_analyzer import PoseAnalyzer
from analyzers.face_analyzer import FaceAnalyzer


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    pose_analyzer = PoseAnalyzer()
    face_analyzer = FaceAnalyzer()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Kamera görüntüsü alınamadı.")
            break

        # Pose analizi
        frame = pose_analyzer.process_frame(frame)

        # Face analizi
        frame = face_analyzer.process_frame(frame)

        cv2.imshow("MediaPipe Realtime Analysis", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    pose_analyzer.close()
    face_analyzer.close()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()