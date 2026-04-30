import cv2


def draw_text(
    frame,
    text,
    position,
    color=(0, 255, 0),
    scale=0.8,
    thickness=2,
    font=cv2.FONT_HERSHEY_SIMPLEX
):
    cv2.putText(
        frame,
        text,
        position,
        font,
        scale,
        color,
        thickness
    )