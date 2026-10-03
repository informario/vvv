import numpy as np

from config import FOE_MIN_FLOW, FOE_MIN_PTS, HEAT_GRID, HEAT_SIGMA


def make_grid(width, height=None):
    """Create a heatmap grid covering either a square or a rectangular image."""
    if height is None:
        height = width
    x_coordinates = (np.arange(HEAT_GRID) + 0.5) * (width / HEAT_GRID)
    y_coordinates = (np.arange(HEAT_GRID) + 0.5) * (height / HEAT_GRID)
    grid_x, grid_y = np.meshgrid(x_coordinates, y_coordinates)
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)
    return grid_x, grid_y, points


def vote_heatmap(old_points, new_points, grid_points):
    """Vote for FOE positions with Gaussian bands around flow lines."""
    displacement = new_points - old_points
    magnitude = np.linalg.norm(displacement, axis=1)
    usable = magnitude > FOE_MIN_FLOW
    if usable.sum() < FOE_MIN_PTS:
        return None

    origins = old_points[usable]
    displacement = displacement[usable]
    magnitude = magnitude[usable]
    direction = displacement / magnitude[:, None]
    normals = np.stack([direction[:, 1], -direction[:, 0]], axis=1).astype(np.float32)
    constants = (normals * origins).sum(axis=1).astype(np.float32)
    weights = np.minimum(magnitude, 5.0).astype(np.float32)

    distance = grid_points @ normals.T - constants
    heat = (np.exp(-distance**2 / (2 * HEAT_SIGMA**2)) * weights).sum(axis=1)
    return (heat / weights.sum()).reshape(HEAT_GRID, HEAT_GRID)


def update_heatmap(previous, current, smoothing):
    if current is None:
        return previous
    return smoothing * previous + (1 - smoothing) * current


def heat_centroid(heat, grid_x, grid_y, cutoff):
    """Return the centroid using only heat cells above the display threshold."""
    maximum = heat.max()
    if maximum <= 0:
        return None

    normalized = heat / maximum
    selected = normalized > cutoff
    if not selected.any():
        return None

    # Keep only visible cells. This avoids weighting the full grid and
    # excludes the same low heat that is omitted from the visualization.
    values = normalized[selected] - cutoff
    x = float((values * grid_x[selected]).sum() / values.sum())
    y = float((values * grid_y[selected]).sum() / values.sum())
    return x, y
