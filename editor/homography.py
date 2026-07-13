"""Planar homography estimation (normalized DLT) and application.

Maps image pixel coordinates to pitch coordinates in meters, given >= 4
point correspondences on the ground plane. The projective model absorbs
the broadcast camera's perspective.
"""

import numpy as np


class CalibrationError(ValueError):
    pass


def _normalize(pts):
    """Translate/scale points so centroid is origin and mean distance is sqrt(2)."""
    pts = np.asarray(pts, dtype=float)
    centroid = pts.mean(axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    mean_dist = dists.mean()
    if mean_dist < 1e-12:
        raise CalibrationError("calibration points are coincident")
    s = np.sqrt(2) / mean_dist
    T = np.array([
        [s, 0, -s * centroid[0]],
        [0, s, -s * centroid[1]],
        [0, 0, 1],
    ])
    homog = np.column_stack([pts, np.ones(len(pts))])
    return (T @ homog.T).T[:, :2], T


def compute_homography(src_pts, dst_pts):
    """Solve H such that [x_dst, y_dst, 1] ~ H @ [x_src, y_src, 1].

    src_pts: Nx2 pixel coordinates, dst_pts: Nx2 pitch meters, N >= 4.
    """
    src_pts = np.asarray(src_pts, dtype=float)
    dst_pts = np.asarray(dst_pts, dtype=float)
    if len(src_pts) < 4 or len(src_pts) != len(dst_pts):
        raise CalibrationError("need at least 4 point correspondences")

    src_n, T_src = _normalize(src_pts)
    dst_n, T_dst = _normalize(dst_pts)

    rows = []
    for (u, v), (x, y) in zip(src_n, dst_n):
        rows.append([-u, -v, -1, 0, 0, 0, u * x, v * x, x])
        rows.append([0, 0, 0, -u, -v, -1, u * y, v * y, y])
    A = np.array(rows)

    _, sv, Vt = np.linalg.svd(A)
    # a (near-)zero second-smallest singular value means the solution is not
    # unique - the points are degenerate (e.g. collinear)
    if sv[-2] < 1e-9 * sv[0]:
        raise CalibrationError("calibration points are degenerate (collinear?)")
    H = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T_dst) @ H @ T_src
    return H / H[2, 2]


def apply_homography(H, pts):
    """Map Nx2 pixel points through H; returns Nx2 pitch coordinates."""
    pts = np.asarray(pts, dtype=float)
    homog = np.column_stack([pts, np.ones(len(pts))])
    mapped = (H @ homog.T).T
    w = mapped[:, 2]
    if np.any(np.abs(w) < 1e-12):
        raise CalibrationError("point maps to infinity (on the horizon line)")
    return mapped[:, :2] / w[:, None]
