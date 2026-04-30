import math


def calculate_angle(a, b, c):
    """
    Calculates the angle at point b using three landmark points: a, b, c.
    Example: shoulder - elbow - wrist gives the elbow angle.
    """

    ab_x = a.x - b.x
    ab_y = a.y - b.y

    cb_x = c.x - b.x
    cb_y = c.y - b.y

    dot_product = ab_x * cb_x + ab_y * cb_y

    ab_length = math.sqrt(ab_x**2 + ab_y**2)
    cb_length = math.sqrt(cb_x**2 + cb_y**2)

    cos_angle = dot_product / (ab_length * cb_length)
    cos_angle = max(-1, min(1, cos_angle))

    angle = math.degrees(math.acos(cos_angle))

    return angle


def calculate_distance(p1, p2):
    """
    Calculates the distance between two landmark points.
    Used for eye open/closed detection.
    """

    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)