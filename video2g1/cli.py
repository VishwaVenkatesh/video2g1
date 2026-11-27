"""video2g1 command line.

  python -m video2g1.cli demo  --model path/to/unitree_g1/scene.xml
  python -m video2g1.cli run my_video.mp4 --model path/to/unitree_g1/scene.xml
  python -m video2g1.cli concat outputs/squat.npz outputs/wave.npz --model ... --out outputs/squat_wave
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .export import render, save_csv, save_npz, side_by_side, view
from .quality import report
from .retarget import G1Retargeter


def main():
    ap = argparse.ArgumentParser(prog="video2g1")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "demo"):
        p = sub.add_parser(name)
        if name == "run":
            p.add_argument("video", help="video file, or a .npy/.npz of (T,33,3) MediaPipe world landmarks")
            p.add_argument("--fps", type=float, default=None, help="override fps (needed for .npy input)")
            p.add_argument("--start", type=float, default=0.0, help="start time in seconds (video input)")
            p.add_argument("--end", type=float, default=None, help="end time in seconds (video input)")
        p.add_argument("--model", required=True, help="path to unitree_g1/scene.xml (MuJoCo Menagerie)")
        p.add_argument("--out", default="outputs/clip", help="output path prefix")
        p.add_argument("--no-anchor", action="store_true", help="keep the robot in place (no root travel)")
        p.add_argument("--in-place", action="store_true",
                       help="feet stay planted, heading locked (squats, waves, arm motions)")
        p.add_argument("--upper-body", action="store_true",
                       help="waist-up video: legs stay standing, only arms and torso are copied")
        p.add_argument("--smooth", type=float, default=0.05,
                       help="keypoint smoothing in seconds (try 0.1 for shaky clips)")
        p.add_argument("--no-render", action="store_true")
        p.add_argument("--view", action="store_true", help="open the interactive MuJoCo viewer")
    c = sub.add_parser("concat", help="stitch retargeted clips (.npz) into one motion for training")
    c.add_argument("clips", nargs="+", help="robot motion files written by `run` (outputs/*.npz)")
    c.add_argument("--model", required=True, help="path to unitree_g1/scene.xml (MuJoCo Menagerie)")
    c.add_argument("--out", default="outputs/combined", help="output path prefix")
    c.add_argument("--fps", type=float, default=30.0, help="frame rate of the stitched clip")
    c.add_argument("--blend", type=float, default=1.0, help="transition length between clips (s)")
    c.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    if a.cmd == "concat":
        return _concat(a)

    if a.cmd == "demo":
        from .synthetic import make_motion
        fps = 30.0
        landmarks = make_motion(fps)
    elif a.video.endswith((".npy", ".npz")):
        data = np.load(a.video)
        landmarks = data if isinstance(data, np.ndarray) else data["landmarks"]
        fps = a.fps or (float(data["fps"]) if not isinstance(data, np.ndarray) and "fps" in data else 30.0)
    else:
        from .pose import extract_keypoints
        landmarks, fps = extract_keypoints(a.video, start=a.start, end=a.end)
        fps = a.fps or fps

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out + "_landmarks.npz", landmarks=landmarks, fps=fps)
    rt = G1Retargeter(a.model)
    qpos, _ = rt.retarget(landmarks, fps, smooth_sigma_s=a.smooth, anchor_feet=not a.no_anchor,
                          in_place=a.in_place, upper_body=a.upper_body)
    report(a.model, qpos, fps)
    print("[save]", save_csv(qpos, a.out + ".csv"), f"(Unitree CSV, {fps:.0f} fps)")
    print("[save]", save_npz(qpos, fps, rt.m, a.out + ".npz"))
    if not a.no_render:
        print("[save]", render(a.model, qpos, fps, a.out + ".mp4"))
        if a.cmd == "run" and not a.video.endswith((".npy", ".npz")):
            print("[save]", side_by_side(a.video, a.out + ".mp4", a.out + "_side_by_side.mp4", a.start))
    if a.view:
        view(a.model, qpos, fps)


def _concat(a):
    import mujoco

    from .concat import concat
    clips = []
    for f in a.clips:
        d = np.load(f)
        if "qpos" not in d.files:
            raise SystemExit(f"{f}: not a robot motion file (use the outputs/*.npz from `run`, "
                             f"not *_landmarks.npz)")
        clips.append((d["qpos"], float(d["fps"])))
        print(f"[concat] {f}: {len(d['qpos'])} frames @ {float(d['fps']):.2f} fps")
    qpos = concat(clips, fps=a.fps, blend_s=a.blend)
    print(f"[concat] {len(qpos)} frames @ {a.fps:.1f} fps ({len(qpos) / a.fps:.1f} s)")
    report(a.model, qpos, a.fps)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    print("[save]", save_csv(qpos, a.out + ".csv"), f"(Unitree CSV, {a.fps:g} fps)")
    print("[save]", save_npz(qpos, a.fps, mujoco.MjModel.from_xml_path(a.model), a.out + ".npz"))
    if not a.no_render:
        print("[save]", render(a.model, qpos, a.fps, a.out + ".mp4"))


if __name__ == "__main__":
    main()
