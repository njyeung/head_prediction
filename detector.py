import cv2
import numpy as np


class HeadDetector:
    """
    Head detector for overhead depth frames.

    All tunable parameters can be passed to the constructor.

    Usage:
        detector = HeadDetector()
        points = detector.predict(frame)  # [(x, y), ...]
    """

    def __init__(
        self,
        max_depth=4500,
        min_depth=200,
        floor_percentile=85,
        floor_margin=300,
        min_head_height=500,
        min_blob_area=200,
        max_blob_area=50000,
        head_depth_window=150,
        min_head_pixels=15,
        erode_iters=2,
        dilate_iters=3,
    ):
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.floor_percentile = floor_percentile
        self.floor_margin = floor_margin
        self.min_head_height = min_head_height
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.head_depth_window = head_depth_window
        self.min_head_pixels = min_head_pixels
        self.erode_iters = erode_iters
        self.dilate_iters = dilate_iters
        self._kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    def _estimate_floor(self, frame):
        valid = frame[(frame > self.min_depth) & (frame < self.max_depth)]
        if len(valid) < 1000:
            return 2500
        return float(np.percentile(valid, self.floor_percentile))

    def predict(self, frame, floor_depth=None):
        """
        Detect head positions in an overhead depth frame.

        Args:
            frame: 2D numpy array of depth values in mm (0 = invalid).
            floor_depth: optional known floor depth in mm.

        Returns:
            List of (x, y) tuples in pixel coordinates.
        """
        if floor_depth is None:
            floor_depth = self._estimate_floor(frame)

        valid = (frame > self.min_depth) & (frame < self.max_depth)
        fg = (valid & (frame < floor_depth - self.floor_margin)).astype(np.uint8)

        fg = cv2.erode(fg, self._kernel, iterations=self.erode_iters)
        fg = cv2.dilate(fg, self._kernel, iterations=self.dilate_iters)
        fg &= valid.astype(np.uint8)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)

        heads = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_blob_area or area > self.max_blob_area:
                continue

            x0 = stats[i, cv2.CC_STAT_LEFT]
            y0 = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            roi = frame[y0:y0+h, x0:x0+w]
            roi_fg = (labels[y0:y0+h, x0:x0+w] == i)

            roi_valid = roi_fg & (roi > self.min_depth)
            if not roi_valid.any():
                continue
            blob_min_depth = roi[roi_valid].min()
            if blob_min_depth > floor_depth - self.min_head_height:
                continue

            head_mask = roi_fg & (roi <= blob_min_depth + self.head_depth_window)
            if head_mask.sum() < self.min_head_pixels:
                continue

            hys, hxs = np.where(head_mask)
            cx, cy = float(hxs.mean()), float(hys.mean())
            heads.append((cx + x0, cy + y0))

        return heads
