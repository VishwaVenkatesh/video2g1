#!/usr/bin/env bash
# Fetch only the Unitree G1 model from MuJoCo Menagerie into third_party/.
set -e
mkdir -p third_party && cd third_party
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
cd mujoco_menagerie && git sparse-checkout set unitree_g1
echo "G1 model: third_party/mujoco_menagerie/unitree_g1/scene.xml"
