"""A synthetic "human" (stick figure) in MediaPipe format: wave -> squat -> walk.

Lets you test the whole retargeting pipeline with no video and no pose model.
"""
from __future__ import annotations

import numpy as np

from .retarget import MP, MP_TO_WORLD


def make_motion(fps: float = 30.0, seconds: float = 9.0) -> np.ndarray:
    T = int(fps * seconds)
    kp = np.zeros((T, 33, 3))
    thigh, shank, upper, fore = 0.42, 0.42, 0.28, 0.25
    for i in range(T):
        t = i / fps
        K = {}
        lean = 0.0
        arm_abd = {"l": 0.15, "r": 0.15}     # arm raise sideways (rad from hanging)
        arm_fwd = {"l": 0.0, "r": 0.0}       # arm swing forwards
        elbow = {"l": 0.2, "r": 0.2}
        hip_fl = {"l": 0.0, "r": 0.0}        # hip flexion
        knee_fl = {"l": 0.0, "r": 0.0}
        if t < 3.0:                          # 1) wave both arms
            a = 0.5 * (1 - np.cos(2 * np.pi * t / 3.0))
            arm_abd = {"l": 0.15 + 2.3 * a, "r": 0.15 + 1.4 * a}
            elbow = {"l": 0.2 + 0.6 * a * np.sin(8 * t), "r": 0.2}
        elif t < 6.0:                        # 2) squat with arms forward
            a = 0.5 * (1 - np.cos(2 * np.pi * (t - 3.0) / 3.0))
            hip_fl = {"l": 1.0 * a, "r": 1.0 * a}
            knee_fl = {"l": 1.9 * a, "r": 1.9 * a}
            arm_fwd = {"l": 1.3 * a, "r": 1.3 * a}
            lean = 0.35 * a
        else:                                # 3) walk forward
            w = 2 * np.pi * 0.9 * (t - 6.0)
            hip_fl = {"l": 0.4 * np.sin(w), "r": -0.4 * np.sin(w)}
            knee_fl = {"l": 0.8 * max(0.0, np.cos(w)), "r": 0.8 * max(0.0, -np.cos(w))}
            arm_fwd = {"l": -0.4 * np.sin(w), "r": 0.4 * np.sin(w)}
            elbow = {"l": 0.5, "r": 0.5}
        torso = np.array([np.sin(lean), 0.0, np.cos(lean)])
        for s, sign in (("l", 1.0), ("r", -1.0)):
            hip = np.array([0.0, sign * 0.10, 0.0])
            sh = 0.5 * torso + np.array([0.0, sign * 0.18, 0.0])
            h, k = hip_fl[s], knee_fl[s]
            knee = hip + thigh * np.array([np.sin(h), 0.0, -np.cos(h)])
            ankle = knee + shank * np.array([np.sin(h - k), 0.0, -np.cos(h - k)])
            ab, fw, eb = arm_abd[s], arm_fwd[s], elbow[s]
            d_up = np.array([np.sin(fw) * np.cos(ab), sign * np.sin(ab), -np.cos(fw) * np.cos(ab)])
            d_fo = np.array([np.sin(fw + eb) * np.cos(ab), sign * np.sin(ab), -np.cos(fw + eb) * np.cos(ab)])
            el = sh + upper * d_up
            wr = el + fore * d_fo / np.linalg.norm(d_fo)
            K.update({s + "_hip": hip, s + "_sh": sh, s + "_kn": knee, s + "_an": ankle,
                      s + "_el": el, s + "_wr": wr})
        for name, idx in MP.items():
            kp[i, idx] = MP_TO_WORLD.T @ K[name]          # world axes -> MediaPipe axes
    return kp
