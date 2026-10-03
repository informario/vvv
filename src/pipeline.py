import cv2 as cv
import numpy as np

from config import (
    EDGE_MARGIN,
    HEAT_DRAW,
    HEAT_GRID,
    HEAT_SMOOTH,
    ROLL_LEAK,
    SCALE,
    SHOW_GRADIENT,
    SHOW_ROLL,
    LIVE_CAPTURE,
)
from foe import heat_centroid, make_grid, update_heatmap, vote_heatmap
from live_overlay import LiveOverlay
from preprocessing import edge_image, prepare_canvas, rotated_frame, rotation_matrix
from roll import accumulate_roll, estimate_roll, remove_roll_component
from tracking import as_feature_array, refill_features, track_points
from visualization import draw_flow, draw_heatmap, draw_roll


def _open_camera():
    capture = cv.VideoCapture(10, cv.CAP_V4L2)
    if not capture.isOpened():
        raise RuntimeError("Cannot open camera")
    return capture


def run():
    capture = _open_camera()
    overlay = None
    try:
        if LIVE_CAPTURE:
            overlay = LiveOverlay()
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Can't read first frame")

        frame = cv.resize(frame, None, fx=SCALE, fy=SCALE)
        canvas_size, height, width = prepare_canvas(frame.shape)
        raw_valid = np.zeros((height, width), dtype=np.uint8)
        raw_valid[EDGE_MARGIN:height - EDGE_MARGIN, EDGE_MARGIN:width - EDGE_MARGIN] = 255

        previous_edge = edge_image(frame)
        initial_rotation = rotation_matrix(height, width, 0.0, canvas_size)
        previous_rotated_edge = cv.warpAffine(
            previous_edge, initial_rotation, (canvas_size, canvas_size)
        )
        raw_points = None
        rotated_points = None
        original_flow_points = None
        angle = 0.0
        heat = np.zeros((HEAT_GRID, HEAT_GRID), np.float32)
        if SHOW_ROLL:
            grid_x, grid_y, grid_points = make_grid(canvas_size)
        else:
            grid_x, grid_y, grid_points = make_grid(width, height)

        while True:
            ok, frame = capture.read()
            if not ok:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame = cv.resize(frame, None, fx=SCALE, fy=SCALE)
            current_edge = edge_image(frame)

            # Stage 1: estimate and accumulate roll from the raw frames.
            raw_points = refill_features(previous_edge, raw_points, raw_valid)
            delta = 0.0
            if raw_points is not None and len(raw_points):
                old, new = track_points(previous_edge, current_edge, raw_points, raw_valid)
                delta = estimate_roll(old, new)
                raw_points = as_feature_array(new)
            angle = accumulate_roll(angle, delta, ROLL_LEAK)

            # Stage 2 and 3: either display the derotated video, or keep the
            # original video and remove only the rotational flow component.
            flow_old = np.empty((0, 2), dtype=np.float32)
            flow_new = np.empty((0, 2), dtype=np.float32)
            if SHOW_ROLL:
                rotated, rotated_edge, valid = rotated_frame(
                    frame, current_edge, angle, canvas_size, EDGE_MARGIN
                )
                visual = rotated.copy()
                display_edge = rotated_edge
                rotated_points = refill_features(
                    previous_rotated_edge, rotated_points, valid
                )
                if rotated_points is not None and len(rotated_points):
                    old, new = track_points(
                        previous_rotated_edge, rotated_edge, rotated_points, valid
                    )
                    flow_old, flow_new = old, new
                    draw_flow(visual, old, new)
                    rotated_points = as_feature_array(new)
                    heat = update_heatmap(
                        heat, vote_heatmap(old, new, grid_points), HEAT_SMOOTH
                    )
            else:
                visual = frame.copy()
                display_edge = current_edge
                original_flow_points = refill_features(
                    previous_edge, original_flow_points, raw_valid
                )
                if original_flow_points is not None and len(original_flow_points):
                    old, new = track_points(
                        previous_edge, current_edge, original_flow_points, raw_valid
                    )
                    old, new = remove_roll_component(
                        old, new, delta, width, height
                    )
                    flow_old, flow_new = old, new
                    draw_flow(visual, old, new)
                    original_flow_points = as_feature_array(new)
                    heat = update_heatmap(
                        heat, vote_heatmap(old, new, grid_points), HEAT_SMOOTH
                    )

            if LIVE_CAPTURE:
                foe = heat_centroid(heat, grid_x, grid_y, HEAT_DRAW)
                source_size = (visual.shape[1], visual.shape[0])
                overlay.update_data(foe, flow_old, flow_new, source_size)
            else:
                draw_heatmap(visual, heat, grid_x, grid_y)
                draw_roll(visual, angle)
                cv.imshow("derotated flow", visual)
                if SHOW_GRADIENT:
                    cv.imshow("tracking image", display_edge)
            if cv.waitKey(1) == ord("q"):
                break

            previous_edge = current_edge
            if SHOW_ROLL:
                previous_rotated_edge = rotated_edge
    finally:
        if capture is not None:
            capture.release()
        if overlay is not None:
            overlay.close_overlay()
        cv.destroyAllWindows()


