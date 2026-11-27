"""Save retargeted motion + render it.

CSV layout (no header, one row per frame, 36 columns) is Unitree's generalized-coordinate
convention used by the LAFAN1 retargeted dataset, BeyondMimic and mjlab:
    root x y z | root quaternion qx qy qz qw | 29 joint angles (rad)
"""
from __future__ import annotations

import os

import mujoco
import numpy as np


def save_csv(qpos: np.ndarray, path: str):
    out = np.concatenate([qpos[:, :3], qpos[:, [4, 5, 6, 3]], qpos[:, 7:]], axis=1)   # wxyz -> xyzw
    np.savetxt(path, out, delimiter=",", fmt="%.6f")
    return path


def save_npz(qpos: np.ndarray, fps: float, model: mujoco.MjModel, path: str):
    names = [model.joint(i).name for i in range(1, model.njnt)]
    np.savez(path, qpos=qpos, fps=fps, joint_names=np.array(names))
    return path


def render(model_path: str, qpos: np.ndarray, fps: float, out_path: str,
           width: int = 640, height: int = 480, gif: bool = True):
    """Offscreen render to mp4 (and a small gif). On a headless machine set MUJOCO_GL=egl."""
    import imageio.v2 as imageio

    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height, width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 2.6, 180.0, -8.0     # front view, like the source video
    frames = []
    for q in qpos:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        cam.lookat[:] = (q[0], q[1], 0.75)
        renderer.update_scene(d, cam)
        frames.append(renderer.render().copy())
    renderer.close()
    imageio.mimsave(out_path, frames, fps=fps, macro_block_size=1)
    if gif:
        step = max(1, int(round(fps / 15)))
        imageio.mimsave(os.path.splitext(out_path)[0] + ".gif", [f[::2, ::2] for f in frames[::step]],
                        duration=step / fps, loop=0)
    return out_path


def view(model_path: str, qpos: np.ndarray, fps: float):
    """Interactive looped playback in the MuJoCo viewer."""
    import time

    import mujoco.viewer

    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            for q in qpos:
                d.qpos[:] = q
                mujoco.mj_forward(m, d)
                v.sync()
                time.sleep(1.0 / fps)
                if not v.is_running():
                    break


def side_by_side(video_path: str, robot_mp4: str, out_path: str, start: float = 0.0):
    """Source video (left) next to the robot render (right), same height, robot's frame rate."""
    import cv2
    import imageio

    robot = imageio.mimread(robot_mp4, memtest=False)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * start)
    h = robot[0].shape[0]
    frames = []
    for r in robot:
        ok, bgr = cap.read()
        if not ok:
            break
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        w = int(round(img.shape[1] * h / img.shape[0])) // 2 * 2
        frames.append(np.hstack([cv2.resize(img, (w, h)), r]))
    cap.release()
    fps = imageio.v3.immeta(robot_mp4).get("fps", 30)
    imageio.mimsave(out_path, frames, fps=fps, macro_block_size=1)
    return out_path
