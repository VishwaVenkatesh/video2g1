"""Stitch several retargeted clips into one continuous motion.

mjlab's tracking task trains on a single motion file. Stitching clips with smooth transitions lets
one policy learn several skills: training samples random moments of the stitched clip.

Each clip is resampled to a common frame rate, then moved so it starts where the previous one ended
(same position on the floor, same heading), and joined to it by a short blend.

Clips made with `--upper-body` have no real leg motion. They take over the previous clip's final
stance (root and legs), so the feet stay planted through the join instead of sliding.
"""
from __future__ import annotations

import numpy as np


def _slerp(q0, q1, t):
    """Quaternion slerp, wxyz. q0, q1: (4,), t: (N,) in [0, 1]."""
    q0, q1 = q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        out = q0[None] + t[:, None] * (q1 - q0)[None]
    else:
        th = np.arccos(d)
        out = (np.sin((1 - t) * th)[:, None] * q0 + np.sin(t * th)[:, None] * q1) / np.sin(th)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def resample(qpos: np.ndarray, fps_in: float, fps_out: float) -> np.ndarray:
    """Resample a (T, 36) qpos trajectory: linear for positions and joints, slerp for the root quaternion."""
    T = len(qpos)
    t_in = np.arange(T) / fps_in
    t_out = np.arange(0.0, t_in[-1] + 1e-9, 1.0 / fps_out)
    out = np.empty((len(t_out), qpos.shape[1]))
    for j in [*range(3), *range(7, qpos.shape[1])]:
        out[:, j] = np.interp(t_out, t_in, qpos[:, j])
    idx = np.clip(np.searchsorted(t_in, t_out, side="right") - 1, 0, T - 2)
    frac = (t_out - t_in[idx]) / (t_in[idx + 1] - t_in[idx])
    for k, (i, f) in enumerate(zip(idx, frac)):
        out[k, 3:7] = _slerp(qpos[i, 3:7], qpos[i + 1, 3:7], np.array([f]))[0]
    return out


def _yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _place(qpos: np.ndarray, xy: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate and shift a clip so its first frame sits at floor position xy with heading yaw."""
    q = qpos.copy()
    dyaw = yaw - _yaw(q[0, 3:7])
    c, s = np.cos(dyaw), np.sin(dyaw)
    R = np.array([[c, -s], [s, c]])
    q[:, :2] = (q[:, :2] - q[0, :2]) @ R.T + xy
    rz = np.array([np.cos(dyaw / 2), 0.0, 0.0, np.sin(dyaw / 2)])
    q[:, 3:7] = np.array([_quat_mul(rz, r) for r in q[:, 3:7]])
    return q


def concat(clips: list[tuple[np.ndarray, float]], fps: float = 30.0, blend_s: float = 1.0) -> np.ndarray:
    """clips: list of (qpos (T,36), fps). Returns one stitched (T,36) qpos at `fps`."""
    nb = max(2, int(round(blend_s * fps)))
    out = resample(clips[0][0], clips[0][1], fps)
    for q, f in clips[1:]:
        nxt = _place(resample(q, f, fps), out[-1, :2], _yaw(out[-1, 3:7]))
        if np.ptp(nxt[:, 7:19], axis=0).max() < 1e-6:      # upper-body clip: keep the previous stance
            nxt[:, :7] = out[-1, :7]
            nxt[:, 7:19] = out[-1, 7:19]
        a, b = out[-1], nxt[0]
        t = np.linspace(0, 1, nb + 2)[1:-1]
        s = 0.5 - 0.5 * np.cos(np.pi * t)                  # ease in and out
        blend = np.empty((nb, out.shape[1]))
        blend[:, :3] = a[:3] + s[:, None] * (b[:3] - a[:3])
        blend[:, 7:] = a[7:] + s[:, None] * (b[7:] - a[7:])
        blend[:, 3:7] = _slerp(a[3:7], b[3:7], s)
        out = np.concatenate([out, blend, nxt])
    return out
