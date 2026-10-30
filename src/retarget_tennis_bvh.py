"""Use the unchanged GMR solver to retarget audited Tennis-MoCap to AdaPT G1.

Input BVH and generated motion arrays stay local. Code: project's MIT license.
Derived motion data: source CC BY-SA 3.0, not the code license.
"""
import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
import types
from importlib.metadata import version
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
from project_paths import GMR_ROOT, ADAPT_ROOT as ADA_ROOT
TRACKED_BODIES = (
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)


def core_modules():
    # The public package __init__ also imports optional Torch/viewer/VR modules.
    # Load the original CPU solver and its original BVH parser without those extras.
    package = types.ModuleType("general_motion_retargeting")
    package.__path__ = [str(GMR_ROOT / "general_motion_retargeting")]
    sys.modules[package.__name__] = package
    params = importlib.import_module("general_motion_retargeting.params")
    solver = importlib.import_module("general_motion_retargeting.motion_retarget")
    # GMR passes limits as positional argument 6. Mink 1.3 inserted safety_break
    # there, silently treating that list as True. Preserve GMR's intended limits
    # with an in-memory API adapter; do not modify the upstream checkout.
    import inspect
    import mink
    if list(inspect.signature(mink.solve_ik).parameters)[5] == "safety_break":
        original_solve_ik = mink.solve_ik
        def legacy_solve_ik(configuration, tasks, dt, solver, damping, limits):
            return original_solve_ik(configuration, tasks, dt, solver,
                                     damping=damping, limits=limits)
        mink.solve_ik = legacy_solve_ik
    parser = importlib.import_module("general_motion_retargeting.utils.lafan_vendor.extract")
    math = importlib.import_module("general_motion_retargeting.utils.lafan_vendor.utils")
    return params, solver.GeneralMotionRetargeting, parser.read_bvh, math


def make_config(model, data, human_rest, world_rotation):
    names = {
        "pelvis": "Hips", "torso_link": "Chest",
        "left_hip_yaw_link": "LeftHip", "right_hip_yaw_link": "RightHip",
        "left_knee_link": "LeftKnee", "right_knee_link": "RightKnee",
        "left_ankle_roll_link": "LeftAnkle", "right_ankle_roll_link": "RightAnkle",
        "left_shoulder_yaw_link": "LeftShoulder", "right_shoulder_yaw_link": "RightShoulder",
        "left_elbow_link": "LeftElbow", "right_elbow_link": "RightElbow",
        "left_wrist_yaw_link": "LeftWrist", "right_wrist_yaw_link": "RightWrist",
    }
    # Calibrate rotation/position offsets from BVH rest geometry to G1's T pose.
    # This replaces the LAFAN1-specific offsets, not the GMR optimization algorithm.
    data.qpos[:] = model.qpos0
    data.qpos[:3] = 0
    data.qpos[3:7] = [1, 0, 0, 0]
    for name, angle in [("left_shoulder_roll_joint", np.pi / 2), ("right_shoulder_roll_joint", -np.pi / 2)]:
        j = model.joint(name).id
        data.qpos[model.jnt_qposadr[j]] = angle
    mujoco.mj_forward(model, data)
    robot_leg = np.linalg.norm(data.xpos[model.body("left_ankle_roll_link").id] - data.xpos[model.body("left_hip_yaw_link").id])
    human_leg = np.linalg.norm(human_rest["LeftAnkle"] - human_rest["LeftHip"])
    scale = float(robot_leg / human_leg)
    root_human = human_rest["Hips"]
    root_robot = data.xpos[model.body("pelvis").id].copy()
    tables = [{}, {}]
    for robot_name, human_name in names.items():
        bid = model.body(robot_name).id
        target_rot = Rotation.from_quat(data.xquat[bid], scalar_first=True)
        offset_rot = world_rotation.inv() * target_rot
        calibrated_delta = data.xpos[bid] - root_robot - scale * (human_rest[human_name] - root_human)
        offset_pos = target_rot.inv().apply(calibrated_delta).tolist()
        if robot_name == "pelvis":
            costs = [(0, 10), (100, 5)]
        elif "ankle" in robot_name:
            costs = [(50, 10), (50, 10)]
        elif robot_name == "torso_link":
            costs = [(0, 100), (0, 10)]
        elif "shoulder" in robot_name:
            costs = [(0, 100), (5, 100)]
        else:
            costs = [(0, 10), (10, 5)]
        for table, (position, orientation) in zip(tables, costs):
            table[robot_name] = [human_name, position, orientation, offset_pos, offset_rot.as_quat(scalar_first=True).tolist()]
    return {"robot_root_name": "pelvis", "human_root_name": "Hips", "ground_height": 0,
            "human_height_assumption": 1.0, "use_ik_match_table1": True,
            "use_ik_match_table2": True, "human_scale_table": {n: scale for n in names.values()},
            "ik_match_table1": tables[0], "ik_match_table2": tables[1]}


def reference_arrays(model, poses, fps):
    data = mujoco.MjData(model)
    velocity = np.zeros((len(poses), model.nv))
    for i in range(len(poses)):
        lo, hi = max(0, i - 1), min(len(poses) - 1, i + 1)
        mujoco.mj_differentiatePos(model, velocity[i], (hi - lo) / fps, poses[lo], poses[hi])
    positions, quaternions, linvel, angvel, racket = [], [], [], [], []
    site = model.site("racket_point").id
    for q, v in zip(poses, velocity):
        data.qpos[:], data.qvel[:] = q, v
        mujoco.mj_forward(model, data)
        positions.append(data.xpos.copy())
        quaternions.append(data.xquat.copy())
        linear, angular = [], []
        for bid in range(model.nbody):
            output = np.zeros(6)
            # XBODY is the link origin; BODY would report center-of-mass velocity.
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, bid, output, 0)
            angular.append(output[:3].copy())
            linear.append(output[3:].copy())
        linvel.append(linear)
        angvel.append(angular)
        racket.append(data.site_xpos[site].copy())
    joints = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    qids, vids = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    bids = [model.body(name).id for name in TRACKED_BODIES]
    # Compact tracked layout avoids the world-body offset between a standalone
    # MuJoCo model and mjlab's Entity-relative body indices. Loader supports it.
    arrays = {"fps": np.array([fps], dtype=np.int64), "joint_pos": poses[:, qids].astype(np.float32),
              "joint_vel": velocity[:, vids].astype(np.float32),
              "joint_names": np.array([model.joint(i).name for i in joints]),
              "body_names": np.array(TRACKED_BODIES),
              "body_pos_w": np.asarray(positions, dtype=np.float32)[:, bids],
              "body_quat_w": np.asarray(quaternions, dtype=np.float32)[:, bids],
              "body_lin_vel_w": np.asarray(linvel, dtype=np.float32)[:, bids],
              "body_ang_vel_w": np.asarray(angvel, dtype=np.float32)[:, bids]}
    return arrays, np.asarray(racket), velocity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bvh", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--racket-hand", choices=["right", "left"], default="right")
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--avoid-hand-collisions", action="store_true",
                   help="Explicit preprocessing extension: Mink hand/racket-to-torso collision constraints")
    p.add_argument("--posture-cost", type=float, default=0,
                   help="Optional temporal IK regularizer toward the preceding solution; explicit extension")
    p.add_argument("--calibration-profile", choices=["legacy", "anatomical-chain"], default="legacy")
    p.add_argument("--full-racket-hull", action="store_true", help="Conservative full visual hull IK experiment")
    p.add_argument("--anatomical-position-scale", type=float, default=1)
    p.add_argument("--anatomical-arm-objective", choices=["hybrid", "landmarks"], default="hybrid")
    p.add_argument("--settle-first-frame", action="store_true",
                   help="Converge first IK target without advancing human motion time")
    p.add_argument("--collision-bound-profile",choices=["legacy","upstream-displacement"],default="legacy",
                   help="Opt-in narrow backport of upstream Mink collision displacement-bound fix")
    a = p.parse_args()
    if (a.calibration_profile == "anatomical-chain" or a.full_racket_hull) and a.racket_hand != "right":
        raise ValueError("Experimental anatomical/full-hull recipe is validated for right racket only")
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    params, GMR, read_bvh, vendor = core_modules()
    collision_bound={"profile":"legacy"}
    if a.collision_bound_profile=="upstream-displacement":
        if not a.full_racket_hull or not a.avoid_hand_collisions or not version("mujoco").startswith("3.12."):
            raise ValueError("Displacement experiment requires full-hull/hand QA and the fixed 3.12 runtime")
        from mink_collision_displacement import install
        collision_bound=install()
    raw = a.bvh.read_text(encoding="utf-8")
    input_dt = float(re.search(r"Frame Time:\s*([0-9.]+)", raw).group(1))
    if a.calibration_profile == "anatomical-chain" and (a.fps != 50 or not np.isclose(input_dt,.01)):
        raise ValueError("Experimental anatomical recipe requires audited 100Hz source and 50Hz output")
    declared_frames = int(re.search(r"Frames:\s*(\d+)", raw).group(1))
    motion_rows = [line.split() for line in re.split(r"Frame Time:[^\n]*\n", raw)[1].splitlines() if line.strip()]
    actual_frames = len(motion_rows)
    if len(set(map(len, motion_rows))) != 1:
        raise ValueError("Inconsistent BVH numeric row lengths")
    # Upstream LAFAN parser splits numeric rows on a literal single space.
    # Normalize whitespace in a separate local copy; preserve the original BVH.
    normalized = a.output / "normalized_input_local.bvh"
    normalized_text = "\n".join(" ".join(line.split()) for line in raw.splitlines() if line.strip()) + "\n"
    normalized_text = re.sub(r"Frames:\s*\d+", f"Frames: {actual_frames}", normalized_text, count=1)
    normalized.write_text(normalized_text, encoding="utf-8")
    bvh = read_bvh(str(normalized))
    gquat, gpos = vendor.quat_fk(bvh.quats, bvh.pos, bvh.parents)
    # Arena: Y up, +X anatomical left, +Z forward. Target: Z up, +Y left, +X forward.
    conversion = Rotation.from_matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    cm_to_m = 0.01
    gpos = conversion.apply(gpos.reshape(-1, 3)).reshape(gpos.shape) * cm_to_m
    gquat = (conversion * Rotation.from_quat(gquat.reshape(-1, 4), scalar_first=True)).as_quat(scalar_first=True).reshape(gquat.shape)
    rest_rot = np.zeros_like(bvh.quats[:1]); rest_rot[..., 0] = 1
    rest_local = np.asarray(bvh.pos[:1]).copy()
    rest_local[:, 0] = 0
    _, rest_pos = vendor.quat_fk(rest_rot, rest_local, bvh.parents)
    human_rest = {n: conversion.apply(rest_pos[0, i]) * cm_to_m for i, n in enumerate(bvh.bones)}
    xml = ADA_ROOT / f"src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1_{a.racket_hand}racket.xml"
    # Relative XML path avoids the Windows Unicode filename issue.
    model = mujoco.MjModel.from_xml_path(os.path.relpath(xml))
    hinge_count = int(np.sum(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE))
    # Standalone asset XML has no actuators; mjlab adds them when building an environment.
    if hinge_count != 27 or model.nq != 34 or model.nv != 33:
        raise ValueError("Expected the exact AdaPT 27-joint model")
    calibration = {"profile": a.calibration_profile}
    if a.calibration_profile == "anatomical-chain":
        from anatomical_reference import fit_rest_pose, remap_chain, make_anatomical_config
        rest_pose, rest_positions, fit = fit_rest_pose(model, human_rest)
        if any(not r["optimizer_success"] or r["direction_residual_norm"]>.005 for r in fit.values()):
            raise ValueError("Anatomical rest-direction fit failed; keep source for review")
        source_positions = gpos.copy()
        gpos, mapping = remap_chain(gpos, gquat, bvh.bones, human_rest, rest_positions, conversion)
        if not 0 < a.anatomical_position_scale <= 20:
            raise ValueError("Anatomical position-cost scale must be in (0,20]")
        cfg = make_anatomical_config(model, rest_pose, conversion, a.anatomical_position_scale, a.anatomical_arm_objective)
        calibration.update(fit=fit, mapping=mapping, position_cost_scale=a.anatomical_position_scale,
                           arm_objective=a.anatomical_arm_objective)
        np.savez_compressed(a.output / "human_targets_local.npz", source_pos=source_positions,
                            target_pos=gpos, source_quat=gquat, bone_names=np.array(bvh.bones),
                            source_fps=np.array([1/input_dt]), rest_qpos=rest_pose)
    else:
        cfg = make_config(model, mujoco.MjData(model), human_rest, conversion)
    cfg_path = a.output / "tennis_bvh_to_adapt_g1.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    params.ROBOT_XML_DICT["adapt_g1"] = Path(os.path.relpath(xml))
    params.IK_CONFIG_DICT["tennis_bvh"] = {"adapt_g1": cfg_path}
    retargeter = GMR("tennis_bvh", "adapt_g1", solver="daqp", verbose=False, use_velocity_limit=False)
    if a.calibration_profile == "anatomical-chain":
        from anatomical_reference import add_shoulder_position_tasks
        add_shoulder_position_tasks(retargeter, a.anatomical_position_scale)
    if a.full_racket_hull:
        from anatomical_reference import add_racket_hull_constraints
        calibration["hull_constraints"] = add_racket_hull_constraints(retargeter)
    if a.avoid_hand_collisions:
        import mink
        # The standalone AdaPT asset has no motors, so GMR's motor-name-based
        # velocity-limit option is empty. Use its intended 3*pi limit on hinge
        # joint names directly to keep each linearized IK step small.
        retargeter.ik_limits.append(mink.VelocityLimit(
            retargeter.model, {retargeter.model.joint(i).name: 3*np.pi
                               for i in range(retargeter.model.njnt)
                               if retargeter.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE}))
        retargeter.max_iter = 50
        retargeter.ik_limits.append(mink.CollisionAvoidanceLimit(
            retargeter.model,
            geom_pairs=[(["left_rubber_hand_collision", "left_wrist_collision", "right_wrist_collision", "tennis_racket_collision"],
                         ["torso_collision", "head_collision", "pelvis_collision",
                          "left_hip_collision", "right_hip_collision",
                          "left_thigh_collision", "right_thigh_collision",
                          "left_shin_collision", "right_shin_collision"])],
            gain=0.5, minimum_distance_from_collisions=0.03, collision_detection_distance=0.15,
        ))
        retargeter.ik_limits.append(mink.CollisionAvoidanceLimit(
            retargeter.model,
            geom_pairs=[(["left_rubber_hand_collision", "left_wrist_collision"],
                         ["right_wrist_collision", "tennis_racket_collision"])],
            gain=0.5, minimum_distance_from_collisions=0.005, collision_detection_distance=0.1,
        ))
    timestamps = np.arange(int(np.floor((len(gpos) - 1) * input_dt * a.fps)) + 1) / a.fps
    indices = np.rint(timestamps / input_dt).astype(int)
    if not np.allclose(indices * input_dt, timestamps):
        raise ValueError("This initial converter only supports exact input-frame resampling")
    poses, errors = [], []
    first_frame_extra_passes = 0
    posture_task = None
    if a.posture_cost > 0:
        import mink
        posture_task = mink.PostureTask(retargeter.model, cost=a.posture_cost)
        posture_task.set_target_from_configuration(retargeter.configuration)
        retargeter.tasks1.append(posture_task)
        retargeter.tasks2.append(posture_task)
    for ix in indices:
        if posture_task is not None:
            posture_task.set_target_from_configuration(retargeter.configuration)
        human = {n: [gpos[ix, i], gquat[ix, i]] for i, n in enumerate(bvh.bones)}
        pose = retargeter.retarget(human)
        if a.settle_first_frame and ix == indices[0]:
            for _ in range(20):
                previous = pose.copy()
                pose = retargeter.retarget(human)
                first_frame_extra_passes += 1
                if np.max(np.abs(pose-previous)) < 1e-6:
                    break
        if a.avoid_hand_collisions:
            # Linearized collision constraints can still penetrate on a large
            # nonlinear IK step. Continue the same solver at the same target,
            # bounded to 20 passes; QA still reports any remaining penetration.
            for _ in range(20):
                mujoco.mj_forward(retargeter.model, retargeter.configuration.data)
                if all(c.dist >= -0.005 for c in retargeter.configuration.data.contact):
                    break
                pose = retargeter.retarget(human)
        poses.append(pose)
        errors.append([retargeter.error1(), retargeter.error2()])
    poses = np.asarray(poses)
    # One constant floor calibration; preserve vertical motion instead of grounding each frame.
    data = mujoco.MjData(model)
    foot_geoms = [i for i in range(model.ngeom) if model.geom(i).name and "foot" in model.geom(i).name and "collision" in model.geom(i).name]
    foot_minima = []
    for q in poses:
        data.qpos[:] = q; mujoco.mj_forward(model, data)
        values = []
        for gid in foot_geoms:
            if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_CAPSULE:
                raise ValueError("Foot calibration currently requires capsule geoms")
            extent = model.geom_size[gid, 0] + model.geom_size[gid, 1] * abs(data.geom_xmat[gid].reshape(3, 3)[2, 2])
            values.append(float(data.geom_xpos[gid, 2] - extent))
        foot_minima.append(min(values))
    # Exact capsule vertical extent. Leave 2 mm clearance at the lowest frame.
    shift = 0.002 - float(min(foot_minima))
    poses[:, 2] += shift
    arrays, racket, velocity = reference_arrays(model, poses, a.fps)
    np.savez_compressed(a.output / "motion.npz", **arrays)
    np.savez_compressed(a.output / "qpos_reference_local.npz", qpos=poses, qvel=velocity)
    joint_ids = [model.joint(n).id for n in arrays["joint_names"]]
    limits = model.jnt_range[joint_ids]
    q = arrays["joint_pos"]
    limited = model.jnt_limited[joint_ids].astype(bool)
    violation = (np.maximum(limits[:, 0] - q, 0) + np.maximum(q - limits[:, 1], 0))[:, limited]
    self_contacts = []
    for pose in poses:
        data.qpos[:] = pose; mujoco.mj_forward(model, data)
        for contact in data.contact:
            if contact.dist < -0.005:
                self_contacts.append({"geom1": model.geom(int(contact.geom1)).name,
                                      "geom2": model.geom(int(contact.geom2)).name,
                                      "penetration_m": float(-contact.dist)})
    racket_speed = np.linalg.norm(np.gradient(racket, 1 / a.fps, axis=0), axis=1)
    peak = int(np.argmax(racket_speed))
    metadata = {"type": "GMR_kinematic_reference_not_physical_correction", "input_file": a.bvh.name,
                "input_sha256": hashlib.sha256(a.bvh.read_bytes()).hexdigest(), "input_fps": 1/input_dt,
                "declared_bvh_frames": declared_frames, "actual_bvh_frames": actual_frames,
                "normalization": "Whitespace only; Frames header set to observed numeric row count; original preserved",
                "frames": len(poses), "fps": a.fps, "racket_hand": a.racket_hand,
                "target_model_sha256": hashlib.sha256(xml.read_bytes()).hexdigest(),
                "gmr_commit": subprocess.check_output(["git", "-C", str(GMR_ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "mujoco_version": mujoco.__version__, "solver": "daqp", "use_velocity_limit": False,
                "collision_bound":collision_bound,
                "dependency_versions":{name:version(name) for name in ["mujoco","mink","daqp","numpy","scipy"]},
                "adapt_commit":subprocess.check_output(["git","-C",str(ADA_ROOT),"rev-parse","HEAD"],text=True).strip(),
                "conversion_code_sha256":{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in
                    [Path(__file__),ROOT/"anatomical_reference.py",ROOT/"mink_collision_displacement.py"]},
                "hand_collision_avoidance_extension": a.avoid_hand_collisions,
                "collision_continuation_max_passes": 20 if a.avoid_hand_collisions else 0,
                "ik_hinge_step_velocity_limit_rad_s": float(3*np.pi) if a.avoid_hand_collisions else None,
                "ik_velocity_note": "Limit is per optimization step, NOT a bound on the final reference joint velocities",
                "ik_max_iter_per_stage": retargeter.max_iter,
                "ik_dt_s":float(retargeter.model.opt.timestep),
                "temporal_posture_cost": a.posture_cost,
                "mink_compatibility": "Pass GMR's legacy sixth positional limits argument by keyword",
                "calibration": calibration,
                "first_frame_extra_ik_passes": first_frame_extra_passes,
                "first_frame_note": "Same source timestamp; IK convergence only, not time stretching",
                "unit_assumption": "Arena BVH centimeters; checked against skeleton dimensions",
                "world_axes": "BVH XYZ -> G1 YZX; Y-up to Z-up, anatomical left to +Y",
                "ground_shift_m": shift, "all_finite": bool(all(np.isfinite(v).all() for v in arrays.values() if np.issubdtype(v.dtype,np.number))),
                "support_foot_clearance_quantiles_m": np.quantile(np.asarray(foot_minima) + shift, [0, 0.5, 1]).tolist(),
                "body_layout": "Compact 14 tracked bodies in exact AdaPT MotionCommandCfg order; XBODY link velocities",
                "self_contact_samples_over_5mm": len(self_contacts),
                "largest_self_penetration": max(self_contacts, key=lambda c: c["penetration_m"], default=None),
                "max_joint_limit_violation_rad": float(violation.max()),
                "declared_limited_joints":[model.joint(i).name for i in joint_ids if model.jnt_limited[i]],
                "unlimited_asset_joints":[model.joint(i).name for i in joint_ids if not model.jnt_limited[i]],
                "limit_note":"Only asset-declared limits checked; not a certification of hardware limits or reset soft-limit equality",
                "max_joint_speed_rad_s": float(np.max(np.abs(arrays["joint_vel"]))),
                "mean_final_ik_error": np.mean(errors, axis=0).tolist(),
                "swing_peak_time_s": peak / a.fps, "swing_peak_speed_m_s": float(racket_speed[peak]),
                "event_note": "Maximum reference racket speed, NOT a measured ball-contact annotation",
                "data_license": "CC BY-SA 3.0 (Tennis-MoCap); independent code license MIT",
                "policy_loaded": False, "physics_stepped": False, "wall_seconds": time.perf_counter()-start}
    (a.output / "reference_quality.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if a.preview:
        renderer = mujoco.Renderer(model, height=360, width=480)
        camera = mujoco.MjvCamera(); camera.distance=3; camera.azimuth=125; camera.elevation=-15
        sheets = []
        with imageio.get_writer(a.output / "reference_only.mp4", fps=25) as writer:
            for i in range(0, len(poses), 2):
                data.qpos[:], data.qvel[:] = poses[i], velocity[i]; mujoco.mj_forward(model,data)
                camera.lookat[:] = poses[i, :3] + [0,0,0.15]
                renderer.update_scene(data,camera=camera)
                picture=Image.fromarray(renderer.render()); draw=ImageDraw.Draw(picture)
                draw.rectangle((0,0,480,28), fill="black")
                draw.text((8,7),f"GMR reference ONLY | t={i/a.fps:.2f}s | no policy/physics", fill="white")
                writer.append_data(np.array(picture))
                if i % 20 == 0: sheets.append(picture)
        renderer.close()
        chosen=[sheets[i] for i in np.linspace(0,len(sheets)-1,8,dtype=int)]
        sheet=Image.new("RGB",(1920,720))
        for i,picture in enumerate(chosen): sheet.paste(picture,((i%4)*480,(i//4)*360))
        sheet.save(a.output / "reference_contact_sheet.png")
    print(json.dumps(metadata,indent=2))


if __name__ == "__main__":
    main()
