import numpy as np
import cv2

# Defaults for the sample video (``input_videos/arg-egy-14_57-goal-low-fps.mp4``).
# The calibration trapezoid covers only ~23 m of the pitch length -- it was the
# original hardcoded per-camera calibration, kept here so ``pipeline.py`` still
# runs identically for the sample when no editor calibration is supplied. For a
# different camera (or to cover more of the pitch) build a ``ViewTransformer``
# from the editor's calibration via :meth:`from_calibration`.
DEFAULT_PIXEL_VERTICES = [[110, 1035], [265, 275], [910, 260], [1640, 915]]
DEFAULT_TARGET_VERTICES = [[0, 68], [0, 0], [23.32, 0], [23.32, 68]]
DEFAULT_PITCH_LENGTH = 105.0
DEFAULT_PITCH_WIDTH = 68.0

# Standard pitch dims in meters; only carry-through metadata for callers that
# want to know the calibrated plane extent (e.g. to clip / center coords).
# They do NOT affect the homography itself -- only the supplied (pixel, target)
# correspondences do.


class ViewTransformer():
    def __init__(self, pixel_vertices=None, target_vertices=None,
                 pitch_length=None, pitch_width=None):
        if pixel_vertices is not None and target_vertices is not None:
            self.pixel_vertices = np.array(pixel_vertices, dtype=np.float32)
            self.target_vertices = np.array(target_vertices, dtype=np.float32)
        else:
            self.pixel_vertices = np.array(DEFAULT_PIXEL_VERTICES, dtype=np.float32)
            self.target_vertices = np.array(DEFAULT_TARGET_VERTICES, dtype=np.float32)
        self.pitch_length = float(pitch_length) if pitch_length is not None else DEFAULT_PITCH_LENGTH
        self.pitch_width = float(pitch_width) if pitch_width is not None else DEFAULT_PITCH_WIDTH

        # 4 points -> exact; >4 -> least-squares fit. Both yield a 3x3 worked on
        # by cv2.perspectiveTransform, so the rest of the API is uniform.
        if len(self.pixel_vertices) == 4:
            self.persepctive_trasnformer = cv2.getPerspectiveTransform(
                self.pixel_vertices, self.target_vertices)
        else:
            mat, _ = cv2.findHomography(self.pixel_vertices, self.target_vertices)
            if mat is None:
                raise ValueError("calibration points are collinear or otherwise degenerate")
            self.persepctive_trasnformer = mat

    @classmethod
    def from_calibration(cls, points, pitch_length, pitch_width):
        """Build a transformer from editor calibration correspondences.

        ``points`` is a list of ``[px, py, mx, my]`` 4-tuples mapping a pixel
        location on frame 0 to corner-origin metric coordinates on the pitch.
        At least 4 non-collinear points are required.
        """
        if len(points) < 4:
            raise ValueError(f"need >=4 calibration points, got {len(points)}")
        pixel = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
        target = np.array([[p[2], p[3]] for p in points], dtype=np.float32)
        return cls(pixel_vertices=pixel, target_vertices=target,
                   pitch_length=pitch_length, pitch_width=pitch_width)

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        tranform_point = cv2.perspectiveTransform(reshaped_point, self.persepctive_trasnformer)
        return tranform_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_trasnformed = self.transform_point(position)
                    if position_trasnformed is not None:
                        position_trasnformed = position_trasnformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_trasnformed