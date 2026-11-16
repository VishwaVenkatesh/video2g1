"""Human video -> 3D keypoints, using MediaPipe Pose Landmarker (runs on CPU, no GPU needed).

Returns (T, 33, 3) hip-centred "world" landmarks in metres plus the video fps.
Frames where no person is detected are filled by linear interpolation.
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")


def _model_path() -> str:
    path = os.path.join(os.path.expanduser("~"), ".cache", "video2g1", "pose_landmarker_heavy.task")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"[pose] downloading pose model -> {path}")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def extract_keypoints(video_path: str, model_path: str | None = None, max_frames: int | None = None):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path or _model_path()),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, i = [], 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, bgr = cap.read()
            if not ok or (max_frames and i >= max_frames):
                break
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            res = landmarker.detect_for_video(image, int(1000 * i / fps))
            if res.pose_world_landmarks:
                frames.append([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]])
            else:
                frames.append(np.full((33, 3), np.nan))
            i += 1
    cap.release()
    kp = np.asarray(frames, dtype=float)
    if len(kp) == 0 or np.isnan(kp).all():
        raise RuntimeError("no person detected in the video")
    missing = np.isnan(kp[:, 0, 0])
    if missing.any():                                   # fill gaps by interpolation
        t, good = np.arange(len(kp)), ~missing
        flat = kp.reshape(len(kp), -1)
        for c in range(flat.shape[1]):
            flat[missing, c] = np.interp(t[missing], t[good], flat[good, c])
        print(f"[pose] interpolated {missing.sum()} frames with no detection")
    print(f"[pose] {len(kp)} frames @ {fps:.1f} fps")
    return kp, float(fps)
