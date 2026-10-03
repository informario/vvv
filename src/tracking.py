import cv2 as cv

from config import FB_THRESH, LK_PARAMS, MAX_POINTS, MIN_DIST, MIN_POINTS


def refill_features(image, points, valid_mask):
    """Top up Shi-Tomasi features while preserving existing points."""
    count = 0 if points is None else len(points)
    if count >= MIN_POINTS:
        return points

    mask = valid_mask.copy()
    if count:
        for x, y in points.reshape(-1, 2):
            cv.circle(mask, (int(x), int(y)), MIN_DIST, 0, -1)

    new_points = cv.goodFeaturesToTrack(
        image,
        maxCorners=MAX_POINTS - count,
        qualityLevel=0.05,
        minDistance=MIN_DIST,
        blockSize=7,
        mask=mask,
    )
    if new_points is None:
        return points

    new_points = new_points.astype("float32")
    return new_points if count == 0 else cv.vconcat([points, new_points])


def track_points(previous_image, image, points, valid_mask):
    """Track LK points with forward-backward validation and border rejection."""
    next_points, status, _ = cv.calcOpticalFlowPyrLK(
        previous_image, image, points, None, **LK_PARAMS
    )
    if next_points is None or status is None:
        empty = points.reshape(-1, 2)[:0]
        return empty, empty.copy()

    returned_points, return_status, _ = cv.calcOpticalFlowPyrLK(
        image, previous_image, next_points, None, **LK_PARAMS
    )
    if returned_points is None or return_status is None:
        empty = points.reshape(-1, 2)[:0]
        return empty, empty.copy()

    fb_error = abs(points - returned_points).reshape(-1, 2).max(axis=1)
    good = (
        (status.ravel() == 1)
        & (return_status.ravel() == 1)
        & (fb_error < FB_THRESH)
    )

    old = points[good].reshape(-1, 2)
    new = next_points[good].reshape(-1, 2)
    height, width = valid_mask.shape
    x = new[:, 0].astype(int)
    y = new[:, 1].astype(int)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    valid = inside.copy()
    valid[inside] = valid_mask[y[inside], x[inside]] > 0
    return old[valid], new[valid]


def as_feature_array(points):
    return points.reshape(-1, 1, 2).astype("float32")
