"""
SORT: Simple Online and Realtime Tracking
A simple implementation of SORT tracker using Kalman Filter + Hungarian Algorithm
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(bb_test, bb_gt):
    """Compute IoU between two bounding boxes [x1,y1,x2,y2]"""
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    intersection = w * h
    area_test = (bb_test[2]-bb_test[0]) * (bb_test[3]-bb_test[1])
    area_gt   = (bb_gt[2]-bb_gt[0])   * (bb_gt[3]-bb_gt[1])
    union = area_test + area_gt - intersection
    return intersection / union if union > 0 else 0.


def convert_bbox_to_z(bbox):
    """Convert [x1,y1,x2,y2] to [cx,cy,s,r] for Kalman state"""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h          # scale (area)
    r = w / float(h)   # aspect ratio
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """Convert Kalman state [cx,cy,s,r] back to [x1,y1,x2,y2]"""
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


class KalmanBoxTracker:
    """Tracks a single object using a Kalman Filter"""
    count = 0

    def __init__(self, bbox):
        # State: [cx, cy, s, r, vx, vy, vs]
        # Only cx, cy, s have velocity components (vx, vy, vs at indices 4,5,6).
        # r (aspect ratio, index 3) has NO velocity term and stays constant.
        self.kf_F = np.eye(7, 7)
        self.kf_F[0, 4] = 1  # cx += vx
        self.kf_F[1, 5] = 1  # cy += vy
        self.kf_F[2, 6] = 1  # s  += vs

        self.kf_H = np.eye(4, 7)

        self.kf_R = np.eye(4) * 1.
        self.kf_R[2:, 2:] *= 10.

        self.kf_P = np.eye(7) * 10.
        self.kf_P[4:, 4:] *= 1000.

        self.kf_Q = np.eye(7)
        self.kf_Q[4:, 4:] *= 0.01

        self.x = np.zeros((7, 1))
        self.x[:4] = convert_bbox_to_z(bbox)
        self.P = self.kf_P.copy()

        KalmanBoxTracker.count += 1
        self.id           = KalmanBoxTracker.count
        self.history      = []
        self.hits         = 0
        self.hit_streak   = 0
        self.age          = 0
        self.time_since_update = 0

    def _predict(self):
        # Safety: prevent scale (area, index 2) from going non-positive
        if (self.x[2] + self.x[6]) <= 0:
            self.x[6] *= 0.0
        self.x = self.kf_F @ self.x
        self.x[2] = max(self.x[2], 1.0)  # area must stay positive
        self.P = self.kf_F @ self.P @ self.kf_F.T + self.kf_Q
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.age += 1
        self.history.append(convert_x_to_bbox(self.x))
        return self.history[-1]

    def _update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        z = convert_bbox_to_z(bbox)
        y = z - self.kf_H @ self.x
        S = self.kf_H @ self.P @ self.kf_H.T + self.kf_R
        K = self.P @ self.kf_H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.kf_H) @ self.P

    def get_state(self):
        return convert_x_to_bbox(self.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)))
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices = np.stack([row_ind, col_ind], axis=1)

    unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
    unmatched_trackers   = [t for t in range(len(trackers))   if t not in matched_indices[:, 1]]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    matches = np.empty((0, 2), dtype=int) if len(matches) == 0 else np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort:
    """SORT Multi-Object Tracker"""

    def __init__(self, max_age=5, min_hits=2, iou_threshold=0.3):
        self.max_age      = max_age
        self.min_hits     = min_hits
        self.iou_threshold = iou_threshold
        self.trackers     = []
        self.frame_count  = 0

    def update(self, dets=np.empty((0, 5))):
        """
        dets: [[x1,y1,x2,y2,score], ...]
        Returns: [[x1,y1,x2,y2,track_id], ...]
        """
        self.frame_count += 1

        # Predict new locations
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t]._predict()[0]
            trks[t] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets[:, :4], trks, self.iou_threshold)

        for m in matched:
            self.trackers[m[1]]._update(dets[m[0], :4])

        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :4]))

        ret = []
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < self.max_age) and \
               (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def reset(self):
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0
