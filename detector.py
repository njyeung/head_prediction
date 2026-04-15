import cv2
import numpy as np

# --- Tunable parameters ---
MAX_DEPTH = 4500
MIN_DEPTH = 200            # mm, ignore closer (sensor noise)
FLOOR_PERCENTILE = 85      # percentile for floor depth estimate
FLOOR_MARGIN = 300         # mm, foreground = closer than floor by this much
MIN_HEAD_HEIGHT = 500      # mm, blob's peak must be this much above floor
MIN_BLOB_AREA = 200        # min foreground pixels per person
MAX_BLOB_AREA = 50000      # max pixels (reject whole-scene blobs)
HEAD_DEPTH_WINDOW = 150    # mm, centroid region around depth peak
MIN_HEAD_PIXELS = 15       # min pixels in a head region to count
ERODE_ITERS = 2
DILATE_ITERS = 3

MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def _estimate_floor(frame):
    """Estimate floor depth from high percentile of valid depths."""
    valid = frame[(frame > MIN_DEPTH) & (frame < MAX_DEPTH)]
    if len(valid) < 1000:
        return 2500
    return float(np.percentile(valid, FLOOR_PERCENTILE))


def _detect_heads(frame, floor_depth):
    """
    Raw detection: returns list of (x, y) pixel coordinates.
    """
    valid = (frame > MIN_DEPTH) & (frame < MAX_DEPTH)
    fg = (valid & (frame < floor_depth - FLOOR_MARGIN)).astype(np.uint8)

    fg = cv2.erode(fg, MORPH_KERNEL, iterations=ERODE_ITERS)
    fg = cv2.dilate(fg, MORPH_KERNEL, iterations=DILATE_ITERS)
    fg &= valid.astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)

    heads = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_BLOB_AREA or area > MAX_BLOB_AREA:
            continue

        x0 = stats[i, cv2.CC_STAT_LEFT]
        y0 = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        roi = frame[y0:y0+h, x0:x0+w]
        roi_fg = (labels[y0:y0+h, x0:x0+w] == i)

        roi_valid = roi_fg & (roi > MIN_DEPTH)
        if not roi_valid.any():
            continue
        blob_min_depth = roi[roi_valid].min()
        if blob_min_depth > floor_depth - MIN_HEAD_HEIGHT:
            continue

        head_mask = roi_fg & (roi <= blob_min_depth + HEAD_DEPTH_WINDOW)
        if head_mask.sum() < MIN_HEAD_PIXELS:
            continue

        hys, hxs = np.where(head_mask)
        cx, cy = float(hxs.mean()), float(hys.mean())
        heads.append((cx + x0, cy + y0))

    return heads


class HeadDetector:
    """
    Stateless head detector for overhead depth frames.

    Usage:
        detector = HeadDetector()
        points = detector.predict(frame)  # [(x, y), ...]
    """

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
            floor_depth = _estimate_floor(frame)
        return _detect_heads(frame, floor_depth)
