"""Local GVHMR -> unchanged GMR SMPL-X solver -> AdaPT 27-DoF reference (MIT).

Derived data retain CalTennis/SMPL-X restrictions; never imply dynamic feasibility.
The first 22 joints suffice for every target in GMR's smplx_to_g1 configuration.
"""
import argparse
import hashlib
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mink
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp

from retarget_tennis_bvh import ADA_ROOT, GMR_ROOT, core_modules, reference_arrays
from sole_grounding import capsule_bottom, foot_geometry


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def human_frames(joints_path, parameters_path, body_model_path, fps):
    joints = np.load(joints_path, allow_pickle=False)
    params = np.load(parameters_path, allow_pickle=False)
    positions = joints["joints_global"].astype(float)
    count, links, _ = positions.shape
    assert links == 22 and params["global__body_pose"].shape == (count, 63)
    parents = np.load(body_model_path, allow_pickle=False)["kintree_table"][0, :22].astype(int)
    parents[0] = -1
    assert np.all(parents[1:] < np.arange(1, 22))
    source_time = joints["time_seconds"]
    target_time = np.arange(int(np.floor(source_time[-1] * fps)) + 1) / fps
    conversion = Rotation.from_matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotations = []
    for i in range(22):
        local = Rotation.from_rotvec(params["global__global_orient"] if i == 0 else
                                    params["global__body_pose"][:, 3*(i-1):3*i])
        rotations.append(local if i == 0 else rotations[parents[i]] * local)
    pos = np.empty((len(target_time), 22, 3))
    quat = np.empty((len(target_time), 22, 4))
    for j in range(22):
        transformed = conversion.apply(positions[:, j])
        pos[:, j] = np.stack([np.interp(target_time, source_time, transformed[:, k])
                              for k in range(3)], axis=-1)
        quat[:, j] = Slerp(source_time, conversion * rotations[j])(target_time).as_quat(scalar_first=True)
    # GMR's own GVHMR loader uses this heuristic, not a measured subject height.
    height = float(1.66 + .1 * params["global__betas"][0, 0])
    assert np.isfinite(pos).all() and np.isfinite(quat).all()
    return pos, quat, joints["joint_names"].tolist(), height, target_time


def make_ik_model(source_xml, output):
    tree = ET.parse(source_xml)
    root = tree.getroot()
    # MuJoCo's Windows mesh loader cannot open absolute Unicode paths here.
    root.find("compiler").set("meshdir", os.path.relpath(source_xml.parent / "assets", output).replace("\\", "/"))
    # Original GMR targets fixed toe frames; no additional joints or mass.
    for side in ("left", "right"):
        ankle = root.find(f".//body[@name='{side}_ankle_roll_link']")
        ET.SubElement(ankle, "body", name=f"{side}_toe_link", pos="0.1 0 -0.02")
    path = output / "ik_model_local.xml"
    tree.write(path, encoding="unicode")
    return path


def preview(model, poses, velocities, output, fps):
    renderer = mujoco.Renderer(model, height=360, width=480)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3, 125, -15
    chosen = set(np.linspace(0, len(poses)-1, 8, dtype=int).tolist())
    pictures = []
    option = mujoco.MjvOption()
    option.geomgroup[3] = False
    try:
        with imageio.get_writer(output / "reference_only.mp4", fps=fps) as writer:
            for i, pose in enumerate(poses):
                data.qpos[:], data.qvel[:] = pose, velocities[i]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = pose[:3] + [0, 0, .15]
                renderer.update_scene(data, camera=camera, scene_option=option)
                picture = Image.fromarray(renderer.render())
                ImageDraw.Draw(picture).text((8, 8), f"GMR reference ONLY | {i/fps:.2f}s | no physics", fill="white")
                writer.append_data(np.asarray(picture))
                if i in chosen:
                    pictures.append(picture)
        sheet = Image.new("RGB", (1920, 720))
        for i, picture in enumerate(pictures):
            sheet.paste(picture, ((i % 4)*480, (i//4)*360))
        sheet.save(output / "reference_contact_sheet.png")
    finally:
        renderer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joints", type=Path, required=True)
    p.add_argument("--parameters", type=Path, required=True)
    p.add_argument("--body-model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--frame-rate-limit-rad-s",type=float,default=None,
                   help="Experimental cumulative per-frame hinge bound; not motor feasibility")
    p.add_argument("--racket-body-avoidance",action="store_true",
                   help="Experimental Mink full-racket hull versus torso/legs limit; arms/grip excluded")
    a = p.parse_args()
    if a.frame_rate_limit_rad_s is not None and a.frame_rate_limit_rad_s<=0:
        raise ValueError("Frame rate limit must be positive")
    if a.fps != 50:
        raise ValueError("This bounded experiment uses AdaPT's 50 Hz control")
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    pos, quat, names, height, timestamps = human_frames(a.joints, a.parameters, a.body_model, a.fps)
    params, GMR, _, _ = core_modules()
    source_xml = ADA_ROOT / "src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1_rightracket.xml"
    xml = make_ik_model(source_xml, a.output)
    config = GMR_ROOT / "general_motion_retargeting/ik_configs/smplx_to_g1.json"
    params.ROBOT_XML_DICT["adapt_smplx"] = Path(os.path.relpath(xml))
    params.IK_CONFIG_DICT["smplx"]["adapt_smplx"] = config
    retarget = GMR("smplx", "adapt_smplx", actual_human_height=height, verbose=False)
    model = retarget.model
    assert (model.nq, model.nv) == (34, 33)
    hinges = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    limited = hinges[model.jnt_limited[hinges].astype(bool)]
    hard_limits = model.jnt_range.copy()
    center = hard_limits[limited].mean(axis=1)
    half = .45 * np.diff(hard_limits[limited], axis=1)[:, 0]
    model.jnt_range[limited] = np.stack([center-half, center+half], axis=1)
    initial = retarget.configuration.data.qpos.copy()
    initial[model.jnt_qposadr[limited]] = np.clip(initial[model.jnt_qposadr[limited]], center-half+1e-6, center+half-1e-6)
    retarget.configuration.update(initial)
    # Recreate configuration limit after shrinking ranges to runtime soft bounds.
    retarget.ik_limits = [mink.ConfigurationLimit(model)]
    retarget.ik_limits.append(mink.VelocityLimit(model, {model.joint(i).name: 3*np.pi for i in hinges}))
    retarget.max_iter = 50
    original_masks=(model.geom_contype.copy(),model.geom_conaffinity.copy())
    avoided_bodies=[]
    if a.racket_body_avoidance:
        racket_id=model.geom('racket_geom').id
        others=[]
        for gid in range(model.ngeom):
            name=model.body(int(model.geom_bodyid[gid])).name
            if (model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_MESH
                and gid!=racket_id and int(model.geom_bodyid[gid])!=int(model.geom_bodyid[racket_id])
                and not any(part in name for part in ('shoulder','elbow','wrist','rubber_hand'))):
                others.append(gid);avoided_bodies.append(name)
        for gid in [racket_id,*others]:
            model.geom_contype[gid]|=1;model.geom_conaffinity[gid]|=1
        collision_limit=mink.CollisionAvoidanceLimit(model,[([racket_id],others)],
            gain=.5,minimum_distance_from_collisions=.01,collision_detection_distance=.08)
        if not collision_limit.geom_id_pairs:raise ValueError('No active racket/body collision pairs')
        retarget.ik_limits.append(collision_limit)
    poses, errors = [], []
    soft_ranges=model.jnt_range.copy()
    for i in range(len(timestamps)):
        if i>0 and a.frame_rate_limit_rad_s is not None:
            # Mink's per-iteration velocity bound does not bound the cumulative
            # displacement over GMR's repeated IK iterations. Anchor joint boxes
            # to the previous OUTPUT frame, fixed across all current iterations.
            bound=a.frame_rate_limit_rad_s/a.fps
            previous_hinges=poses[-1][model.jnt_qposadr[limited]]
            model.jnt_range[limited,0]=np.maximum(soft_ranges[limited,0],previous_hinges-bound)
            model.jnt_range[limited,1]=np.minimum(soft_ranges[limited,1],previous_hinges+bound)
            retarget.ik_limits[0]=mink.ConfigurationLimit(model)
        human = {name: [pos[i, j], quat[i, j]] for j, name in enumerate(names)}
        q = retarget.retarget(human)
        if i == 0:
            for _ in range(20):
                previous = q.copy()
                q = retarget.retarget(human)
                if np.max(np.abs(q-previous)) < 1e-6:
                    break
        poses.append(q)
        errors.append([retarget.error1(), retarget.error2()])
    poses = np.asarray(poses)
    model.jnt_range[:]=soft_ranges
    model.geom_contype[:],model.geom_conaffinity[:]=original_masks
    feet = foot_geometry(model)
    data = mujoco.MjData(model)
    bottoms = []
    for q in poses:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        bottoms.append([capsule_bottom(model, data, f[2]) for f in feet])
    bottoms = np.asarray(bottoms)
    shift = .001 - bottoms.min()
    poses[:, 2] += shift
    arrays, racket, velocities = reference_arrays(model, poses, a.fps)
    np.savez_compressed(a.output / "motion.npz", **arrays)
    np.savez_compressed(a.output / "qpos_reference_local.npz", qpos=poses, qvel=velocities)
    qids = model.jnt_qposadr[limited]
    violation = np.maximum(model.jnt_range[limited, 0]-poses[:, qids], 0) + np.maximum(poses[:, qids]-model.jnt_range[limited, 1], 0)
    contacts = []
    for i, q in enumerate(poses):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for c in data.contact:
            if c.dist < 0:
                contacts.append(dict(frame=i, geom1=model.geom(c.geom1).name,
                                     geom2=model.geom(c.geom2).name, penetration_m=float(-c.dist)))
    speeds = np.linalg.norm(np.gradient(racket, 1/a.fps, axis=0), axis=1)
    result = dict(type="GMR_kinematic_reference_not_physical_correction", frames=len(poses), fps=a.fps,
                  calibration={"profile": "upstream-smplx-to-g1"},
                  all_finite=bool(all(np.isfinite(v).all() for v in arrays.values() if v.dtype.kind in "fiu")),
                  max_joint_limit_violation_rad=float(violation.max()),
                  largest_self_penetration=max(contacts, key=lambda c:c["penetration_m"], default=None),
                  contact_samples=len(contacts), ground_shift_m=float(shift),
                  initial_foot_clearances_m=(bottoms[0]+shift).tolist(),
                  support_foot_clearance_min_m=float(np.min(bottoms+shift)),
                  support_foot_clearance_max_m=float(np.max(np.min(bottoms+shift, axis=1))),
                  max_joint_speed_rad_s=float(np.abs(arrays["joint_vel"]).max()),
                  heuristic_human_height_m=height, height_note="Original GMR beta0 heuristic; not measured height",
                  swing_peak_time_s=float(np.argmax(speeds)/a.fps),
                  event_note="Speed peak only, NOT ball-contact annotation; motion phase requires source-video review",
                  target_model_sha256=sha(source_xml), gmr_config_sha256=sha(config),
                  source_sha256={"joints":sha(a.joints), "parameters":sha(a.parameters)},
                  code_sha256=sha(__file__), mean_ik_error=np.mean(errors, axis=0).tolist(),
                  joint_limit_profile="0.9 soft range on declared limited joints only; unlimited joints remain unlimited",
                  ik_velocity_note="3pi per solver step, NOT final trajectory speed limit",
                  experimental_frame_rate_limit_rad_s=a.frame_rate_limit_rad_s,
                  experimental_racket_body_avoidance=dict(enabled=a.racket_body_avoidance,
                      bodies=sorted(set(avoided_bodies)),min_distance_m=.01,detection_distance_m=.08,
                      note="Mink convex-hull velocity constraint; arms/grip excluded, no physics guarantee"),
                  virtual_toes="GMR fixed toe frames, 0.1 0 -0.02; no added DoF or inertia",
                  resampling="30 -> 50Hz linear XYZ and SLERP global rotations; no extrapolation; all source frame0 included",
                  last_timestamp_s=float(timestamps[-1]), source_last_timestamp_s=float((len(np.load(a.joints)["time_seconds"])-1)/30),
                  racket_hand="right", racket_orientation_estimated=False,
                  physics_stepped=False, policy_loaded=False, data_license="CalTennis CC BY-NC 4.0 + SMPL-X terms; local only",
                  wall_seconds=time.perf_counter()-start)
    (a.output / "reference_quality.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if a.preview:
        preview(model, poses, velocities, a.output, a.fps)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
