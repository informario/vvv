import numpy as np
import cv2 as cv

from config import BLUR_SIGMA, EDGE_GAIN, EDGE_HI, EDGE_LO, GRAY_W, MIN_COMPONENT


def edge_image(frame):
    """Build the grayscale plus hysteresis edge image used by LK."""
    gray0 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray0, 3)
    gray = cv.bilateralFilter(gray, 5, 30, 3)
    gray = cv.GaussianBlur(gray, (0, 0), BLUR_SIGMA)

    gx = cv.Scharr(gray, cv.CV_32F, 1, 0)
    gy = cv.Scharr(gray, cv.CV_32F, 0, 1)
    magnitude = cv.magnitude(gx, gy) / 16.0

    weak = (magnitude >= EDGE_LO).astype(np.uint8)
    count, labels, stats, _ = cv.connectedComponentsWithStats(weak, connectivity=8)
    accepted = np.zeros(count, dtype=bool)
    accepted[np.unique(labels[magnitude >= EDGE_HI])] = True
    accepted &= stats[:, cv.CC_STAT_AREA] >= MIN_COMPONENT
    accepted[0] = False

    edge = np.where(
        accepted[labels],
        np.clip((magnitude - EDGE_LO) * EDGE_GAIN, 0, 255),
        0,
    ).astype(np.uint8)
    edge = cv.GaussianBlur(edge, (3, 3), 0)
    smooth_gray = cv.GaussianBlur(gray0, (0, 0), 1.0)
    return cv.addWeighted(smooth_gray, GRAY_W, edge, 1 - GRAY_W, 0)


def rotation_matrix(height, width, angle, canvas_size):
    """Return a rotation around the source center translated onto a square canvas."""
    matrix = cv.getRotationMatrix2D(
        (width / 2, height / 2), np.degrees(angle), 1.0
    )
    matrix[0, 2] += (canvas_size - width) / 2
    matrix[1, 2] += (canvas_size - height) / 2
    return matrix


def prepare_canvas(frame_shape):
    """Return the diagonal canvas size and the source frame dimensions."""
    height, width = frame_shape[:2]
    return int(np.ceil(np.hypot(width, height))), height, width


def rotated_frame(frame, tracking_image, angle, canvas_size, margin):
    """Rotate the frame and tracking image, and calculate its valid area."""
    height, width = frame.shape[:2]
    matrix = rotation_matrix(height, width, angle, canvas_size)
    rotated = cv.warpAffine(frame, matrix, (canvas_size, canvas_size))
    rotated_tracking = cv.warpAffine(
        tracking_image, matrix, (canvas_size, canvas_size)
    )

    source_mask = np.full((height, width), 255, np.uint8)
    erosion_kernel = np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)
    valid = cv.warpAffine(
        source_mask,
        matrix,
        (canvas_size, canvas_size),
        flags=cv.INTER_NEAREST,
    )
    valid = cv.erode(valid, erosion_kernel)
    return rotated, rotated_tracking, valid

