#!/bin/bash
# After both arms finish: each arm given annotated hit times (the main comparison), and the auto arm given its own auto times.
set -e
cd "$(dirname "$0")"
PY="../../Prototypes/Humanoid_Tennis_OSS/upstream/AdaPT/.venv/Scripts/python.exe"; V=joint_pose_ee_hitbalanced_nonracket_rootvel_adaptive_v4
ANN=../impact_improvement_20261030_v1/impact_manifest_v12_footlock20_local.json; AUTO=impact_manifest_v12_footlock20_autohit.json
$PY eval_launcher.py --manifest $ANN --checkpoint arm_annotated/final.pt --output eval_annotatedarm_annotatedtime_948 --reward-variant $V --batch-size 32 > eval_annotatedarm_annotatedtime.log 2>&1
$PY eval_launcher.py --manifest $ANN --checkpoint arm_auto/final.pt --output eval_autoarm_annotatedtime_948 --reward-variant $V --batch-size 32 > eval_autoarm_annotatedtime.log 2>&1
$PY eval_launcher.py --manifest $AUTO --checkpoint arm_auto/final.pt --output eval_autoarm_autotime_948 --reward-variant $V --batch-size 32 > eval_autoarm_autotime.log 2>&1
$PY score_arms.py
