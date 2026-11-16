# video2g1: full project playbook and handoff

Written 16 Nov 2026. This file is the complete brief for any AI assistant or collaborator who
continues the project. Read sections 1–4 before changing code.

---

## 1. Who and why

- **Owner:** Vishwa Venkatesh, MSc Advanced Robotics (École Centrale de Nantes, 2026). GitHub: `VishwaVenkatesh`.
- **Background:** master's thesis on learned whole-body control for the Unitree G1: PPO policy trained in
  MuJoCo/MJX, evaluated zero-shot in Isaac Lab. Simulation only, never run on a physical G1.
- **Goal of this project:** a flagship open-source project that (a) shows frontier humanoid skills on
  GitHub, (b) gets attention on Reddit/X/LinkedIn, (c) becomes an arXiv paper, (d) makes humanoid
  robotics companies interested in hiring him.
- **Working style:** short, direct answers; plain language with analogies; ask before making a change
  that could introduce inaccuracies; one question at a time.

## 2. The project in one line

**Human video → Unitree G1 motion → RL tracking policy → language-commanded skills.**

| Phase | Input → output | Status on 16 Nov 2026 |
|---|---|---|
| 1 | Video → 3D human keypoints → G1 joint trajectory (CSV) | Retargeting works and is tested on synthetic motion. Video front end written, NOT yet run on a real video. |
| 2 | G1 trajectory → RL policy that physically tracks it in simulation | Not started |
| 3 | Text command → skill selection/sequence → tracker executes | Stub only (keyword matching) |

## 3. What exists in the repo (v0.1, one commit on `main`)

```
video2g1/
  README.md              public description, quick start, limitations, related work
  LICENSE                MIT
  pyproject.toml         deps: mujoco, numpy, imageio, imageio-ffmpeg; extra [video]: mediapipe, opencv-python
  scripts/get_g1.sh      sparse-clones MuJoCo Menagerie, only unitree_g1/
  skills/skills.json     example skill library for phase 3 (motion paths are placeholders)
  docs/ROADMAP.md        checklist per phase
  docs/PLAYBOOK.md       this file
  media/demo.gif, demo.mp4   render of the synthetic test motion on the G1
  video2g1/
    pose.py        video -> (T,33,3) MediaPipe world landmarks + fps
    retarget.py    landmarks -> (T,36) MuJoCo qpos for the 29-DoF G1
    export.py      CSV / NPZ writers, offscreen render, interactive viewer
    synthetic.py   fake human motion (wave 0-3 s, squat 3-6 s, walk 6-9 s) in MediaPipe format
    commander.py   phase-3 stub: text -> skill name by word overlap
    cli.py         `demo` and `run` commands
```

### How the retargeting works (`retarget.py`)

1. Convert MediaPipe axes (x right, y down, z away from camera) to robot axes (x forward, y left, z up)
   with the fixed matrix `MP_TO_WORLD`. Smooth keypoints over time (Gaussian, 0.05 s).
2. Root orientation: y axis = left hip − right hip; up axis = halfway between world-up and the torso
   direction (so the pelvis takes half of a forward lean and the waist joints take the rest).
3. Targets: for shoulders, elbows, wrists, knees and ankles, keep the human limb *direction* and use the
   robot's own limb *length* (measured from the model in the zero pose).
4. Damped least-squares IK over the 29 joints (root fixed during the solve), 15 iterations per frame
   (60 on the first), warm-started from the previous frame. Extra terms: smoothness toward the previous
   frame, weak pull toward the `stand` keyframe, and a term that keeps both soles parallel to the floor.
   Joint limits are enforced by clipping.
5. Root position: MediaPipe is hip-centred, so the lower foot is treated as the stance foot. Root height
   is set so that foot is on the floor; root x/y moves so the stance foot does not slide (2 cm hysteresis
   on stance switches). Root position is smoothed at the end.

Body names used (MuJoCo Menagerie `unitree_g1/scene.xml`, 29 DoF, nq=36): `pelvis`,
`{left,right}_shoulder_pitch_link`, `_elbow_link`, `_wrist_yaw_link`, `_hip_pitch_link`, `_knee_link`,
`_ankle_roll_link`. Joint order in qpos: left leg 6, right leg 6, waist 3, left arm 7, right arm 7.

### Output formats

- `*.csv`: 36 columns, no header: root `x y z`, root quaternion `qx qy qz qw`, 29 joint angles (rad).
  This is Unitree's convention (LAFAN1 retargeted dataset, BeyondMimic, mjlab).
- `*.npz`: `qpos` (quaternion `wxyz`, MuJoCo order), `fps`, `joint_names`.
- `*_landmarks.npz`: the raw human keypoints, so retargeting can be re-run without the video.

### What was verified (in a Linux sandbox, MuJoCo 4.0.1)

- `python -m video2g1.cli demo` runs end to end on the synthetic motion (270 frames, 30 fps).
- IK error: mean 1.45 cm, max 2.78 cm. Joint-limit violations: 0. Max joint speed: 4.1 rad/s.
- Feet stay within about 1 cm of the floor; root travels 2.70 m during the 3 s walk.
- Rendered frames were inspected: wave, squat and walk look correct.
- `mediapipe 0.10.32` imports and exposes `PoseLandmarker`, `RunningMode.VIDEO`, `mp.Image`.

### What was NOT verified (do these first)

- `pose.py` has never processed a real video. The model download URL in `MODEL_URL` was not reachable
  from the sandbox and is unchecked.
- The axis mapping assumes the person roughly faces the camera at the start. Real clips may show a
  mirrored or rotated robot; check left/right first.
- MediaPipe depth is noisy. Expect jitter and occasional left/right leg swaps on real clips.
- The CSV has not been passed through mjlab's `csv_to_npz` yet.
- No unit tests exist.

## 4. Run it

```bash
unzip video2g1.zip && cd video2g1
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[video]"
./scripts/get_g1.sh
G1=third_party/mujoco_menagerie/unitree_g1/scene.xml

# A) synthetic test, no video needed
python -m video2g1.cli demo --model $G1 --out outputs/demo --view

# B) your own video
python -m video2g1.cli run my_video.mp4 --model $G1 --out outputs/me --view

# C) re-run retargeting from saved keypoints
python -m video2g1.cli run outputs/me_landmarks.npz --model $G1 --out outputs/me2
```

Flags: `--no-anchor` keeps the robot in place, `--no-render` skips the mp4, `--view` opens the MuJoCo
viewer. Headless machine: prefix with `MUJOCO_GL=egl`.

Filming tips: full body in frame, static camera, one person, plain background, 5–15 seconds,
start facing the camera.

## 5. Publish on GitHub

1. github.com → New repository → name `video2g1`, Public, do NOT add a README/licence (repo has them).
2. In the folder:
   ```bash
   git remote add origin https://github.com/VishwaVenkatesh/video2g1.git
   git push -u origin main
   ```
   If it asks for a password, use `gh auth login` (GitHub CLI) or a personal access token.
   The first commit uses the author email `VishwaVenkatesh@users.noreply.github.com`; change with
   `git commit --amend --reset-author` before pushing if another address is preferred.
3. Repo page → About: description + topics `humanoid`, `unitree-g1`, `mujoco`, `motion-retargeting`,
   `reinforcement-learning`, `imitation-learning`, `robotics`.
4. Releases → "v0.1: retargeting". Pin the repo on the profile.
5. After real videos work: replace `media/demo.gif` with a side-by-side (human left, G1 right).

## 6. Build plan (in order)

### Milestone 1: real videos (week 1)
- [ ] Run section 4B on 3 own clips: wave, squat, walk. Fix `pose.py` and axis issues.
- [ ] Side-by-side video writer (human frame + G1 render in one mp4/gif).
- [ ] Use heel/toe landmarks (MediaPipe 29–32) for foot yaw; reject frames with low landmark visibility.
- [ ] Quality metrics per clip: foot slide, floor penetration, joint-speed peaks.
- [ ] Unit tests on the synthetic motion (error < 3 cm, no limit violations).
- [ ] Live webcam mode (optional, very good for the first post).
- **Post #1** (section 9).

### Milestone 2: tracking policy (weeks 2–4)
- [ ] Convert one CSV with mjlab's `csv_to_npz` (set `--input-fps` to the clip fps) and train task
      `Mjlab-Tracking-Flat-Unitree-G1`. Docs: https://mujocolab.github.io/mjlab/main/source/training/motion_imitation.html
- [ ] Needs an NVIDIA GPU. No GPU at home: rent one by the hour (a cloud GPU provider or Modal).
- [ ] Train trackers for 5–10 clips; record success rate and tracking error.
- [ ] Sim-to-sim: evaluate the MuJoCo-trained policy in Isaac Lab (thesis tooling).
- [ ] One multi-clip policy conditioned on the reference motion.
- **Post #2.**

### Milestone 3: the experiment that makes it a paper (weeks 5–8)
Research question: **how cheap can the human data be?** Train trackers for the same motions from
three reference sources and compare:
1. laptop-CPU monocular pose (this repo, MediaPipe),
2. high-quality video pose (GVHMR + GMR retargeting, needs GPU and SMPL-X),
3. motion capture (public LAFAN1 retargeted to G1).

Report: policy success rate, joint/keypoint tracking error, foot slide, training time, and the
MuJoCo → Isaac Lab transfer gap. Add ablations of the retargeting terms (foot anchoring, flat-foot
term, smoothing).

### Milestone 4: language layer (weeks 6–9)
- [ ] Skill library from trained clips (`skills/skills.json` with real paths and descriptions).
- [ ] Replace keyword matching in `commander.py` with an LLM that returns a JSON skill sequence.
- [ ] Transitions between skills (blend references or rely on the multi-clip tracker).
- [ ] Vision in the loop: a VLM looks at the simulated camera image and picks the next skill.
- **Post #3.**

### Milestone 5: real robot (when access exists)
- [ ] Find a G1: university lab, Unitree reseller/partner, or a company demo day.
- [ ] Deploy one tracked clip with safety limits. One real clip matters more than ten sim clips.

## 7. Paper

- **Not ready yet.** A paper needs milestones 1–3 with real numbers. The current repo is a tool.
- **Honest positioning:** video → G1 retargeting already exists (GMR with GVHMR; `video2robot`).
  The contribution here is the cheap-data comparison and the end-to-end open pipeline.
- **Structure (6–8 pages):** abstract; introduction; related work (GMR, GVHMR, BeyondMimic, mjlab,
  TWIST, PHC, H2O/OmniH2O, ExBody, HumanPlus); method; experiments; limitations; conclusion.
- **arXiv:** category cs.RO (cross-list cs.LG). First-time submitters usually need an endorsement;
  ask the thesis supervisor and check arXiv's current policy. Use the IEEE conference LaTeX template.
- **Co-author:** invite the thesis supervisor. It adds credibility and review.
- **Peer-reviewed targets:** workshops at CoRL, ICRA, RSS (easiest entry); IEEE-RAS Humanoids
  conference; IEEE RA-L (journal). Deadlines are not checked here: look them up before planning.
- Release code, clips and trained policies with the paper.

## 8. Where to showcase

| Place | What to put there |
|---|---|
| GitHub repo + profile README | Code, GIF at the top, one-command demo |
| Project website (GitHub Pages) | Videos, 5-line explanation, links to code/paper |
| Hugging Face Space | Live demo: upload a video, get the G1 motion back |
| Hugging Face dataset | Retargeted clips (own videos only, or data whose licence allows it) |
| Reddit | r/robotics, r/reinforcementlearning, r/MachineLearning with the [P] tag |
| X/Twitter | 30 s video; this is where most humanoid researchers are |
| LinkedIn | Same video plus what you learned; recruiters see this |
| YouTube | 60–90 s explainer |
| Hacker News | "Show HN" once the live demo exists |
| Communities | LeRobot, MuJoCo and Unitree developer Discords / GitHub discussions |
| CV and applications | Link at the top; two-line message to engineers at target companies |
| Meetups | Paris robotics / AI meetups and hackathons: demo it live |

## 9. Post templates

**Reddit title:** "I built an open-source tool that turns a phone video of a human into Unitree G1
motion, on a laptop CPU"

**Body:** GIF first. Then: what it does (2 lines), how it works (3 bullets), what does not work yet
(2 bullets), repo link, "feedback welcome". Always say it is simulation only.

Do not post until a real video works. Do not claim results that have not been run.

## 10. What humanoid companies look for

- Real hardware results, or a clear "simulation only" label.
- Numbers with fair baselines, not only videos.
- Code that runs with one command; tests; clean commit history.
- Steady visible progress: one release and one post per milestone.
- Understanding of sim-to-real: domain randomisation, actuator limits, latency.

## 11. With no hardware or software limits

Kept separate on purpose: none of this is needed for the paper.

- **Real G1 fleet deployment** with onboard cameras, safety monitor and fall recovery.
- **Universal motion tracker:** one policy trained on all of AMASS plus thousands of hours of web video.
- **Text-to-skill:** a video-generation model creates the human video, so the robot learns motions
  nobody has filmed.
- **Foundation-model brain:** a vision-language-action model that selects, blends and corrects skills
  from camera input (slow brain above the fast tracker).
- **Objects and hands:** carrying, doors, tools, with dexterous hands and tactile sensing; retarget
  hand and object motion from video, not only the body.
- **VR teleoperation** (headset + controllers) to collect correction data on the real robot.
- **Real-to-sim:** scan real rooms with a phone (Gaussian splatting) and train inside them.
- **Cross-embodiment:** the same pipeline for Unitree H1, Booster T1, Fourier, Figure-class robots.
- **World model:** predict whether a skill will succeed before running it; use it to evaluate policies.
- **Diffusion motion prior:** generate and steer whole-body motion at test time.
- **Rough terrain and stairs** with perception.
- **Public benchmark and leaderboard** for video-to-humanoid quality.

## 12. Rules for whoever continues this

1. Every claim in the README must be backed by a command someone can run.
2. Mark clearly what is tested and what is not. Nothing has run on a real robot.
3. Credit prior work (GMR, GVHMR, BeyondMimic, mjlab, MuJoCo Menagerie).
4. Check dataset and video licences before publishing clips. LAFAN1 is non-commercial.
5. Verify current library versions and docs before relying on an API; this field changes monthly.
