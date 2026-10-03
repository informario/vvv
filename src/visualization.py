import numpy as np
import cv2 as cv

from config import HEAT_ALPHA, HEAT_DRAW, LINE_GAIN, SHOW_ROLL
from foe import heat_centroid


def draw_flow(image, old_points, new_points):
    for old, new in zip(old_points, new_points):
        end = old + (new - old) * LINE_GAIN
        cv.line(
            image,
            (int(old[0]), int(old[1])),
            (int(end[0]), int(end[1])),
            (0, 255, 0),
            1,
        )
        cv.circle(image, (int(new[0]), int(new[1])), 2, (0, 0, 255), -1)


def draw_heatmap(image, heat, grid_x, grid_y):
    if heat.max() <= 0:
        return None

    height, width = image.shape[:2]
    normalized = cv.resize(
        heat / heat.max(), (width, height), interpolation=cv.INTER_CUBIC
    )
    color = cv.applyColorMap(
        np.clip(normalized * 255, 0, 255).astype(np.uint8), cv.COLORMAP_JET
    )
    visible = normalized > HEAT_DRAW
    # Blend only selected pixels; hidden heat does not require a full-frame blend.
    image[visible] = (
        image[visible].astype(np.float32) * (1 - HEAT_ALPHA)
        + color[visible].astype(np.float32) * HEAT_ALPHA
    ).astype(np.uint8)

    # HEAT_DRAW is also the FOE cutoff: hidden heat cannot influence its position.
    foe = heat_centroid(heat, grid_x, grid_y, HEAT_DRAW)
    if foe is None:
        return None

    x, y = foe
    cv.drawMarker(image, (int(x), int(y)), (255, 255, 255), cv.MARKER_CROSS, 20, 2)
    cv.putText(
        image,
        f"FOE: ({x - width / 2:+.0f}, {y - height / 2:+.0f}) px from center",
        (5, 32),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    return foe


def draw_roll(image, angle):
    if not SHOW_ROLL:
        return

    cv.putText(
        image,
        f"roll: {np.degrees(angle):+.1f} deg",
        (5, 15),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
