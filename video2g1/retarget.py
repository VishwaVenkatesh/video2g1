"""Retarget 3D human keypoints onto the Unitree G1 (29 DoF) with damped least-squares IK.

Input : (T, 33, 3) MediaPipe "world" landmarks (hip-centred, metres).
Output: (T, 36) MuJoCo qpos  = root xyz, root quat (wxyz), 29 joint angles.

Method (per frame)
  1. Root orientation comes straight from the human hips + torso.
  2. Limb *directions* are copied from the human, limb *lengths* from the robot, which
     gives reachable targets for shoulders, elbows, wrists, knees and ankles.
  3. Damped least-squares IK over the 29 joints, warm-started from the previous frame,
     with joint limits, a smoothness term and a "keep the feet flat" term.
  4. Root height and horizontal travel are recovered by anchoring the stance foot to
     the ground (MediaPipe gives no global translation).
"""
from __future__ import annotations

import mujoco
import numpy as np

# MediaPipe Pose landmark indices
MP = dict(l_sh=11, r_sh=12, l_el=13, r_el=14, l_wr=15, r_wr=16,
          l_hip=23, r_hip=24, l_kn=25, r_kn=26, l_an=27, r_an=28)

# MediaPipe world axes (x: image right, y: down, z: away from camera)
#   -> robot/world axes (x: forward, y: left, z: up), for a person facing the camera.
MP_TO_WORLD = np.array([[0.0, 0.0, -1.0],
                        [1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0]])
UP = np.array([0.0, 0.0, 1.0])


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along time (axis 0)."""
    if sigma <= 0 or len(x) < 3:
        return x
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)])
    flat = pad.reshape(len(pad), -1)
    out = np.stack([np.convolve(flat[:, i], k, mode="valid") for i in range(flat.shape[1])], 1)
    return out.reshape(x.shape)


class G1Retargeter:
    def __init__(self, model_path: str, lam=1e-3, smooth_w=0.02, posture_w=0.002,
                 foot_flat_w=0.3, iters=15):
        self.m = mujoco.MjModel.from_xml_path(model_path)
        self.d = mujoco.MjData(self.m)
        m, d = self.m, self.d
        if m.nq != 36:
            raise ValueError(f"expected the 29-DoF G1 (nq=36), got nq={m.nq}")
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(m, d, kid)
        else:
            mujoco.mj_resetData(m, d)
        self.q0 = d.qpos.copy()
        self.lam, self.mu, self.nu, self.wf, self.iters = lam, smooth_w, posture_w, foot_flat_w, iters
        self.lo, self.hi = m.jnt_range[1:, 0].copy(), m.jnt_range[1:, 1].copy()

        b = lambda n: m.body(n).id
        self.pelvis = b("pelvis")
        self.B = {}
        for s, side in (("l", "left"), ("r", "right")):
            self.B[s + "_sh"] = b(f"{side}_shoulder_pitch_link")
            self.B[s + "_el"] = b(f"{side}_elbow_link")
            self.B[s + "_wr"] = b(f"{side}_wrist_yaw_link")
            self.B[s + "_hip"] = b(f"{side}_hip_pitch_link")
            self.B[s + "_kn"] = b(f"{side}_knee_link")
            self.B[s + "_an"] = b(f"{side}_ankle_roll_link")

        # Robot dimensions, measured once in the zero pose (all joints at 0, upright root).
        d.qpos[:] = 0
        d.qpos[3] = 1
        mujoco.mj_kinematics(m, d)
        P = lambda k: d.xpos[self.B[k]].copy()
        dist = lambda a, c: float(np.linalg.norm(P(a) - P(c)))
        self.torso_len = float(np.linalg.norm(0.5 * (P("l_sh") + P("r_sh")) - d.xpos[self.pelvis]))
        self.half_sh = 0.5 * dist("l_sh", "r_sh")
        self.upper_arm, self.forearm = dist("l_sh", "l_el"), dist("l_el", "l_wr")
        self.thigh, self.shank = dist("l_hip", "l_kn"), dist("l_kn", "l_an")

        # Ankle height above the floor when standing with flat feet.
        d.qpos[:] = self.q0
        mujoco.mj_kinematics(m, d)
        foot_bodies = {self.B["l_an"], self.B["r_an"]}
        lowest = min((d.geom_xpos[g][2] - m.geom_rbound[g]) for g in range(m.ngeom)
                     if m.geom_bodyid[g] in foot_bodies and m.geom_contype[g] + m.geom_conaffinity[g] > 0)
        self.ankle_h = float(d.xpos[self.B["l_an"]][2] - lowest)
        self.ankle_h = float(np.clip(self.ankle_h, 0.02, 0.08))

    # ------------------------------------------------------------------ per frame
    def _root_rotation(self, K):
        y = _unit(K["hip_axis"]) if "hip_axis" in K else _unit(K["l_hip"] - K["r_hip"])
        torso = _unit(0.5 * (K["l_sh"] + K["r_sh"]) - 0.5 * (K["l_hip"] + K["r_hip"]))
        up = _unit(UP + torso)            # pelvis takes half of the torso lean, waist the rest
        x = _unit(np.cross(y, up))
        z = np.cross(x, y)
        return np.stack([x, y, z], 1), torso

    def solve_frame(self, K: dict, q_prev: np.ndarray, iters: int | None = None, pins: dict | None = None):
        """pins: optional {"l_an"/"r_an": world xyz}. Those ankles are held at fixed world points
        and the root is free to translate, so a planted foot cannot slide."""
        m, d = self.m, self.d
        R, torso = self._root_rotation(K)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R.flatten())
        if np.dot(quat, q_prev[3:7]) < 0:
            quat = -quat
        d.qpos[:] = q_prev
        d.qpos[:3] = q_prev[:3] if pins else (0, 0, 1.0)
        d.qpos[3:7] = quat
        mujoco.mj_kinematics(m, d)

        root = d.xpos[self.pelvis].copy()
        sh_mid = root + self.torso_len * torso
        sh_axis = _unit(K["l_sh"] - K["r_sh"])
        tgt = {}
        for s, sign in (("l", 1.0), ("r", -1.0)):
            tgt[s + "_sh"] = sh_mid + sign * self.half_sh * sh_axis
            tgt[s + "_el"] = tgt[s + "_sh"] + self.upper_arm * _unit(K[s + "_el"] - K[s + "_sh"])
            tgt[s + "_wr"] = tgt[s + "_el"] + self.forearm * _unit(K[s + "_wr"] - K[s + "_el"])
            hip = d.xpos[self.B[s + "_hip"]].copy()
            tgt[s + "_kn"] = hip + self.thigh * _unit(K[s + "_kn"] - K[s + "_hip"])
            tgt[s + "_an"] = tgt[s + "_kn"] + self.shank * _unit(K[s + "_an"] - K[s + "_kn"])
        weights = {"sh": 0.5, "el": 1.0, "wr": 1.0, "kn": 1.0, "an": 1.0}
        pins = pins or {}
        offs = {k: p - root for k, p in tgt.items()}         # targets move with the root
        cols = np.r_[0:3, 6:m.nv] if pins else np.r_[6:m.nv]   # free root translation only if pinned
        n = len(cols)
        reg = np.full(n, self.lam + self.mu + self.nu)
        if pins:
            reg[:3] = self.lam + self.mu

        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters or self.iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            root = d.xpos[self.pelvis].copy()
            Js, es = [], []
            for k, o in offs.items():
                w = weights[k[2:]]
                mujoco.mj_jacBody(m, d, jacp, None, self.B[k])
                J = jacp[:, cols].copy()
                if pins:
                    J[:, :3] -= np.eye(3)                     # the target moves with the root too
                Js.append(w * J)
                es.append(w * (root + o - d.xpos[self.B[k]]))
            for k, p in pins.items():                         # planted feet stay where they are
                mujoco.mj_jacBody(m, d, jacp, None, self.B[k])
                Js.append(10.0 * jacp[:, cols])
                es.append(10.0 * (p - d.xpos[self.B[k]]))
            for k in ("l_an", "r_an"):                       # keep the soles parallel to the floor
                mujoco.mj_jacBody(m, d, None, jacr, self.B[k])
                z = d.xmat[self.B[k]].reshape(3, 3)[:, 2]
                Js.append(self.wf * jacr[:, cols])
                es.append(self.wf * np.cross(z, UP))
            J, e = np.vstack(Js), np.concatenate(es)
            x = np.r_[d.qpos[:3], d.qpos[7:]] if pins else d.qpos[7:].copy()
            xp = np.r_[q_prev[:3], q_prev[7:]] if pins else q_prev[7:]
            x0 = np.r_[q_prev[:3], self.q0[7:]] if pins else self.q0[7:]
            H = J.T @ J + np.diag(reg)
            g = J.T @ e + self.mu * (xp - x) + self.nu * (x0 - x) * (reg > self.lam + self.mu)
            x = x + np.linalg.solve(H, g)
            if pins:
                d.qpos[:3] = x[:3]
            d.qpos[7:] = np.clip(x[-29:], self.lo, self.hi)

        mujoco.mj_kinematics(m, d)
        root = d.xpos[self.pelvis].copy()
        err = np.mean([np.linalg.norm(root + o - d.xpos[self.B[k]]) for k, o in offs.items()])
        feet = {s: d.xpos[self.B[s + "_an"]] - root for s in ("l", "r")}
        return d.qpos.copy(), feet, float(err)

    # ------------------------------------------------------------------ whole clip
    def retarget(self, landmarks: np.ndarray, fps: float, smooth_sigma_s: float = 0.05,
                 heading_sigma_s: float = 0.25, anchor_feet: bool = True, in_place: bool = False,
                 verbose: bool = True):
        """landmarks: (T, 33, 3) MediaPipe world landmarks. Returns (qpos (T,36), info dict).

        smooth_sigma_s : Gaussian smoothing of all keypoints (seconds).
        heading_sigma_s: extra smoothing of the hip direction, which sets the robot's heading.
                         Single-camera depth makes it the noisiest signal, so it gets more.
        anchor_feet    : recover forward travel from the planted foot (walking clips).
        in_place       : lock the heading and keep the feet centred (squat, wave, arm motions).
        """
        kp = np.asarray(landmarks, dtype=float) @ MP_TO_WORLD.T
        kp = smooth(kp, smooth_sigma_s * fps)
        T = len(kp)
        hip_axis = smooth(kp[:, MP["l_hip"]] - kp[:, MP["r_hip"]], heading_sigma_s * fps)
        if in_place:                                    # same heading in every frame, keep the tilt
            h = hip_axis[:, :2]
            u = _unit(np.median(h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-9), axis=0))
            hip_axis = np.concatenate([np.linalg.norm(h, axis=1, keepdims=True) * u, hip_axis[:, 2:]], 1)
        qpos, errs = np.zeros((T, 36)), np.zeros(T)
        q_prev = self.q0.copy()
        stance, anchor, root_xy = None, np.zeros(2), np.zeros(2)
        pins = None
        for t in range(T):
            K = {name: kp[t, i] for name, i in MP.items()}
            K["hip_axis"] = hip_axis[t]
            if in_place and t > 0:
                q, feet, errs[t] = self.solve_frame(K, q_prev, pins=pins)
                qpos[t] = q_prev = q
                continue
            q, feet, errs[t] = self.solve_frame(K, q_prev, iters=60 if t == 0 else None)
            low = min(feet["l"][2], feet["r"][2])
            if in_place:                                   # first frame: put both feet on the floor
                q[:3] = (*(-0.5 * (feet["l"][:2] + feet["r"][:2])), self.ankle_h - low)
                self.d.qpos[:] = q
                mujoco.mj_kinematics(self.m, self.d)
                pins = {k: self.d.xpos[self.B[k]].copy() * (1, 1, 0) + (0, 0, self.ankle_h)
                        for k in ("l_an", "r_an")}
                qpos[t] = q_prev = q
                continue
            if stance is None:                              # walking: the lowest foot is planted
                stance = min(feet, key=lambda s: feet[s][2])
                anchor = root_xy + feet[stance][:2]
            other = "r" if stance == "l" else "l"
            root_xy = anchor - feet[stance][:2] if anchor_feet else np.zeros(2)
            if feet[other][2] < feet[stance][2] - 0.02:      # stance switch, with 2 cm hysteresis
                stance = other
                anchor = root_xy + feet[stance][:2]
            q[:2] = root_xy
            q[2] = self.ankle_h - low
            qpos[t] = q
            q_prev = q
        if not in_place:
            qpos[:, :3] = smooth(qpos[:, :3], 0.08 * fps)   # remove jitter from contact switches
        info = dict(mean_err_cm=100 * errs.mean(), max_err_cm=100 * errs.max(),
                    travel_m=float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2])))
        if verbose:
            print(f"[retarget] {T} frames @ {fps:.1f} fps | IK error mean {info['mean_err_cm']:.2f} cm, "
                  f"max {info['max_err_cm']:.2f} cm | root travel {info['travel_m']:.2f} m")
        return qpos, info
