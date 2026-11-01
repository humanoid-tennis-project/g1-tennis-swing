#!/bin/bash
# Two from-scratch arms, identical except the hit-time source. Run sequentially on the local GPU.
set -e
cd "$(dirname "$0")"
PY="../../Prototypes/Humanoid_Tennis_OSS/upstream/AdaPT/.venv/Scripts/python.exe"
COMMON="--num-envs 2048 --updates 3000 --seconds 18000 --randomized-goals --reward-variant joint_pose_ee_hitbalanced_nonracket_rootvel_adaptive_v4 --min-action-std 0.1"
$PY train_launcher.py --impact-manifest ../impact_improvement_20261030_v1/impact_manifest_v12_footlock20_local.json --output arm_annotated $COMMON > arm_annotated.log 2>&1
$PY train_launcher.py --impact-manifest impact_manifest_v12_footlock20_autohit.json --output arm_auto $COMMON > arm_auto.log 2>&1
echo done
