# Roadmap

Each milestone is a release and a post (r/robotics, r/reinforcementlearning, LinkedIn).

## Phase 1: video → G1 motion

- [x] Retargeting IK for the 29-DoF G1 (joint limits, flat feet, stance-foot anchoring)
- [x] Unitree CSV export, MuJoCo render, synthetic test motion
- [x] Video front end runs on a real clip (squat). Found: camera cuts, side views, heading wobble, foot slide
- [x] `--in-place` (feet pinned in the IK), `--smooth`, `--start/--end`, camera-cut warning
- [x] Side-by-side video output and per-clip quality numbers
- [x] Wave clip: `--upper-body` mode for waist-up videos (legs held standing; joint-speed spike 12 → 2 rad/s)
- [x] Walk clip: turning is captured, but steps toward the camera are weak (single-camera depth).
      Fix = high-quality backend below, not more MediaPipe patches
- [x] Three-panel showcase (human | retargeted | trained policy) in the README
- [x] Published on GitHub
- [ ] Post the GIF (LinkedIn, Reddit)  ← next
- [ ] Use MediaPipe heel/toe landmarks for foot yaw; hand landmarks for wrist orientation
- [ ] Live webcam mode (mirror yourself on the G1 in real time). This is the demo GIF for post #1
- [ ] Optional high-quality backend: GVHMR for pose + GMR for retargeting, same CSV out
- [ ] Quality metrics: add ground penetration and self-collision

**Post #1:** "I built a tool that turns any video of a human into Unitree G1 motion, on a laptop CPU."

## Phase 2: G1 motion → tracking policy

Goal: the robot physically performs the clip in simulation, with balance, not just kinematic playback.

- [x] Single-clip tracker trained with mjlab on the squat clip (task `Mjlab-Tracking-Flat-Unitree-G1`,
      RTX 4050 laptop, 1024 envs, 3000 iterations). Commands in the README.
- [x] `concat` command: stitch clips into one motion (resample, align, 1 s blend). Squat + wave = 21.2 s
- [x] First two-motion run (40 min, 4000 iterations): robot survives ~98% of episodes, but steps
      0.5 m sideways at the squat→wave join because the reference slid the feet to a narrower stance
- [x] Fix: waist-up clips inherit the previous stance (feet move 0.000 m through the join)
- [x] Retrained squat + wave with the fixed clip (4000 iterations, 43 min): the robot stays on the
      reference through the whole squat → wave, no step at the join
- [ ] Wrist rotation from hand landmarks, so the hand actually waves  ← next
- [ ] Reproduce on an MJX backend (reuse the thesis PPO setup) and compare
- [ ] Sim-to-sim check: evaluate the MuJoCo-trained tracker in Isaac Lab (thesis tooling)
- [ ] Multi-clip tracker: one policy conditioned on the reference motion, trained on 20+ clips
      (own videos + the public LAFAN1 retargeted set)
- [ ] Report: success rate and tracking error per clip, video-sourced vs mocap-sourced references
- [ ] Real G1 deployment (needs access to a robot)

**Post #2:** "From a phone video to a humanoid doing the motion with a learned controller."

Compute: a single clip trains in hours on one modern GPU. No GPU at home: rent one by the hour.

## Phase 3: language → skills

Goal: type "wave, then walk forward and squat" and watch the G1 do it.

- [ ] Skill library: each tracked clip gets a name + description (`skills/skills.json`)
- [ ] Replace keyword matching in `commander.py` with an LLM that outputs a skill sequence (JSON)
- [ ] Transitions between skills (blend reference motions, or let the multi-clip tracker handle it)
- [ ] Vision in the loop: a VLM looks at the scene camera and chooses the next skill
- [ ] Stretch: text-to-video model generates the human video → pipeline learns a brand-new skill

**Post #3:** "Language-commanded humanoid skills, learned from video, fully open source."

## Rules for the project

- Every claim in the README is backed by a command someone can run.
- State clearly what is simulation only. Nothing here has run on a real G1 yet.
