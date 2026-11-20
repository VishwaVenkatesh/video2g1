# video2g1

**Human video → Unitree G1 motion → tracking policy → language-commanded skills.**

There will never be enough robot data, but there are billions of hours of video of humans moving.
This project is an open, lightweight pipeline that turns any human video into a skill a humanoid can
perform, and then lets you ask for that skill in plain language.

![demo](media/demo.gif)

*Phase 1 output: a human motion (wave → squat → walk) retargeted to the 29-DoF Unitree G1 in MuJoCo.*

## The three phases

| Phase | What it does | Status |
|---|---|---|
| **1. Video → robot motion** | Pose estimation from a single video (MediaPipe, CPU only), retargeting to the G1 with joint limits, flat feet and stance-foot anchoring, export to Unitree CSV | **Working** (retargeting tested on synthetic motion; video front end written but not yet run end to end) |
| **2. Robot motion → policy** | Train an RL motion-tracking policy on the retargeted clips (mjlab / BeyondMimic take the CSV directly), then a single multi-clip tracker | Not started (see [docs/ROADMAP.md](docs/ROADMAP.md)) |
| **3. Language → skill** | A language model picks and sequences skills; the tracker executes them | Stub (`video2g1/commander.py`, keyword matching only) |

## Quick start

```bash
pip install -e .            # core: mujoco + numpy
./scripts/get_g1.sh         # fetches the G1 model from MuJoCo Menagerie
G1=third_party/mujoco_menagerie/unitree_g1/scene.xml

# 1) No video needed: synthetic human -> G1
python -m video2g1.cli demo --model $G1 --out outputs/demo

# 2) Your own video (person fully visible, one person, static camera works best)
pip install -e ".[video]"
python -m video2g1.cli run my_video.mp4 --model $G1 --out outputs/my_clip --view
```

Useful flags for `run`:

| Flag | Use it for |
|---|---|
| `--in-place` | Motions where the feet stay put (squat, wave, arm moves): feet are pinned to the floor, heading is locked |
| `--smooth 0.1` | Shaky clips (default 0.05 s of keypoint smoothing) |
| `--start 5.2 --end 10.1` | Keep one part of a clip; the tool warns if it finds camera cuts |
| `--view` | Open the interactive MuJoCo viewer |

On a headless machine add `MUJOCO_GL=egl`. Each run writes:

- `*.csv`: 36 columns, no header: root `x y z`, root quaternion `qx qy qz qw`, 29 joint angles.
  This is Unitree's convention, the format the LAFAN1 retargeted dataset, BeyondMimic and mjlab use.
- `*.npz`: MuJoCo `qpos` (quaternion `wxyz`), fps and joint names.
- `*.mp4` / `*.gif`: a render of the G1 performing the motion.
- `*_side_by_side.mp4`: the source video next to the G1 (video input only).

Each run also prints quality numbers: foot slide while planted, pelvis path, heading range,
joint jitter and peak joint speed (see `video2g1/quality.py`).

## How the retargeting works

1. **Root orientation** comes from the human hips and torso.
2. **Directions from the human, lengths from the robot.** Each limb segment keeps the human's direction
   but uses the G1's own segment length, so every target is reachable.
3. **Damped least-squares IK** over the 29 joints, warm-started from the previous frame, with joint
   limits, a smoothness term and a term that keeps the soles parallel to the floor.
4. **Stance-foot anchoring.** MediaPipe gives a hip-centred pose with no global position. The lower
   foot is assumed to be planted, which recovers root height and forward travel and prevents that
   foot from sliding.

## Known limitations (v0.1)

- Kinematic only: nothing checks balance or torque limits. That is the tracking policy's job (phase 2).
- No jumps or flight phases (one foot is always assumed to be on the ground).
- Single-camera depth from MediaPipe is noisy; wrists and feet have no orientation tracking yet.
- No self-collision avoidance.

## Related work

This builds on ideas from, and is meant to plug into, existing open tools:
[GMR](https://github.com/YanjieZe/GMR) (general motion retargeting, also supports video via GVHMR),
[GVHMR](https://github.com/zju3dv/GVHMR) (world-grounded human motion from video),
[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) and
[mjlab](https://github.com/mujocolab/mjlab) (motion-tracking RL).
GMR + GVHMR give higher-quality retargeting but need a GPU and the SMPL-X body model.
video2g1 aims to be the version anyone can run on a laptop, and to carry the pipeline through to
a trained policy and a language interface.

## License

MIT. The G1 model comes from MuJoCo Menagerie under its own license.
