import numpy as np
import cv2 as cv


def estimate_roll(old_points, new_points):
    """Estimate frame-to-frame rotation in radians using a RANSAC similarity."""
    if len(old_points) < 8:
        return 0.0

    matrix, _ = cv.estimateAffinePartial2D(
        old_points,
        new_points,
        method=cv.RANSAC,
        ransacReprojThreshold=2.0,
    )
    if matrix is None:
        return 0.0
    return float(np.arctan2(matrix[1, 0], matrix[0, 0]))


def accumulate_roll(angle, delta, leak):
    return angle * leak + delta


def remove_roll_component(old_points, new_points, angle, width, height):
    """Remove the rotational displacement while keeping points in original coordinates."""
    if len(new_points) == 0:
        return old_points, new_points

    center = (width / 2, height / 2)
    matrix = cv.getRotationMatrix2D(center, np.degrees(angle), 1.0)
    corrected_new = cv.transform(new_points[None, :, :], matrix)[0]
    return old_points, corrected_new
