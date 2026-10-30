"""Experimental anatomical rest calibration and parent-driven bone remapping.

The original GMR solver is retained. This is a NEW conversion recipe to test,
not a claim that legacy Tennis-MoCap mapping was official GMR/AdaPT behavior.
"""
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ANCHORS = {
    "Hips": "pelvis", "Chest": "torso_link",
    "LeftHip": "left_hip_yaw_link", "RightHip": "right_hip_yaw_link",
    "LeftKnee": "left_knee_link", "RightKnee": "right_knee_link",
    "LeftAnkle": "left_ankle_roll_link", "RightAnkle": "right_ankle_roll_link",
    "LeftShoulder": "left_shoulder_roll_link", "RightShoulder": "right_shoulder_roll_link",
    "LeftElbow": "left_elbow_link", "RightElbow": "right_elbow_link",
    "LeftWrist": "left_wrist_yaw_link", "RightWrist": "right_wrist_yaw_link",
}
PARENTS = {
    "Chest": "Hips", "LeftHip": "Hips", "RightHip": "Hips",
    "LeftKnee": "LeftHip", "RightKnee": "RightHip",
    "LeftAnkle": "LeftKnee", "RightAnkle": "RightKnee",
    "LeftShoulder": "Chest", "RightShoulder": "Chest",
    "LeftElbow": "LeftShoulder", "RightElbow": "RightShoulder",
    "LeftWrist": "LeftElbow", "RightWrist": "RightElbow",
}


def fit_rest_pose(model, human_rest):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[:3] = 0
    data.qpos[3:7] = [1, 0, 0, 0]
    results = {}
    for side, prefix in [("left", "Left"), ("right", "Right")]:
        names = [f"{side}_shoulder_pitch_joint", f"{side}_shoulder_roll_joint",
                 f"{side}_shoulder_yaw_joint", f"{side}_elbow_joint"]
        ids = [model.joint(name).id for name in names]
        addresses = model.jnt_qposadr[ids]
        initial = np.array([0, np.pi/2 if side == "left" else -np.pi/2, 0, np.pi/2])
        desired = []
        for parent, child in [(prefix+"Shoulder", prefix+"Elbow"), (prefix+"Elbow", prefix+"Wrist")]:
            delta = human_rest[child] - human_rest[parent]
            desired.append(delta / np.linalg.norm(delta))
        def residual(angles):
            data.qpos[addresses] = angles
            mujoco.mj_forward(model, data)
            directions = []
            for parent, child in [(prefix+"Shoulder", prefix+"Elbow"), (prefix+"Elbow", prefix+"Wrist")]:
                delta = data.xpos[model.body(ANCHORS[child]).id] - data.xpos[model.body(ANCHORS[parent]).id]
                directions.append(delta / np.linalg.norm(delta))
            return np.r_[np.asarray(directions).ravel() - np.asarray(desired).ravel(), .002*(angles-initial)]
        limited = model.jnt_limited[ids].astype(bool)
        lower = np.where(limited, model.jnt_range[ids, 0]+1e-5, -np.pi)
        upper = np.where(limited, model.jnt_range[ids, 1]-1e-5, np.pi)
        solved = least_squares(residual, np.clip(initial, lower+1e-5, upper-1e-5), bounds=(lower, upper),
                               max_nfev=200, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        data.qpos[addresses] = solved.x
        error = residual(solved.x)[:6].reshape(2, 3)
        results[side] = {"joint_angles_rad": dict(zip(names, map(float, solved.x))),
                         "direction_residual_norm": float(np.linalg.norm(error)),
                         "optimizer_success": bool(solved.success)}
    mujoco.mj_forward(model, data)
    positions = {name: data.xpos[model.body(body).id].copy() for name, body in ANCHORS.items()}
    return data.qpos.copy(), positions, results


def remap_chain(gpos, gquat, bone_names, human_rest, rest_positions, world_rotation):
    """Use parent rotation deltas to move fixed robot-calibrated bone vectors.

    Wrist position is driven by the elbow parent, never by wrist rotation.
    Root translation alone uses the leg-length ratio. No temporal filtering.
    """
    indexes = {str(name): i for i, name in enumerate(bone_names)}
    root = indexes["Hips"]
    scale = np.linalg.norm(rest_positions["LeftAnkle"]-rest_positions["LeftHip"]) / np.linalg.norm(human_rest["LeftAnkle"]-human_rest["LeftHip"])
    target = gpos.copy()
    target[:, root] = gpos[:, root] * scale
    for child, parent in PARENTS.items():
        rotation_delta = Rotation.from_quat(gquat[:, indexes[parent]], scalar_first=True) * world_rotation.inv()
        bone = rest_positions[child] - rest_positions[parent]
        target[:, indexes[child]] = target[:, indexes[parent]] + rotation_delta.apply(bone)
    # Fixture: changing only wrist orientation cannot move its target position.
    altered = gquat.copy()
    for side in ["Left", "Right"]:
        wrist = indexes[side+"Wrist"]
        altered[:, wrist] = (Rotation.from_quat(altered[:, wrist], scalar_first=True) * Rotation.from_euler("x", np.pi/2)).as_quat(scalar_first=True)
    invariant = True
    for side in ["Left", "Right"]:
        parent = indexes[side+"Elbow"]
        delta = Rotation.from_quat(altered[:, parent], scalar_first=True) * world_rotation.inv()
        check = target[:, parent] + delta.apply(rest_positions[side+"Wrist"]-rest_positions[side+"Elbow"])
        invariant &= bool(np.allclose(check, target[:, indexes[side+"Wrist"]], atol=1e-12))
    return target, {"root_leg_scale": float(scale), "wrist_rotation_position_invariance_fixture": invariant,
                    "position_offsets": "zero; bone lengths and anchor semantics remapped before GMR",
                    "temporal_filtering": False}


def make_anatomical_config(model, rest_pose, world_rotation, position_scale=1, arm_objective="hybrid"):
    data = mujoco.MjData(model)
    data.qpos[:] = rest_pose
    mujoco.mj_forward(model, data)
    tables = [{}, {}]
    for human, position_anchor in ANCHORS.items():
        robot = position_anchor.replace("shoulder_roll", "shoulder_yaw") if "Shoulder" in human else position_anchor
        orientation = Rotation.from_quat(data.xquat[model.body(robot).id], scalar_first=True)
        offset = (world_rotation.inv()*orientation).as_quat(scalar_first=True).tolist()
        if human == "Hips":
            costs = [(0, 10), (100, 5)]
        elif "Ankle" in human:
            costs = [(50, 10), (50, 10)]
        elif human == "Chest":
            costs = [(0, 100), (0, 10)]
        elif "Shoulder" in human:
            costs = [(0, 100), (0, 100)]
        else:
            costs = [(0, 10), (10*position_scale, 5)]
        if arm_objective == "landmarks":
            if "Shoulder" in human:
                # Position is matched at the anatomical roll-origin task below.
                # Tiny orientation keeps the original GMR offset entry present.
                costs = [(0, 1e-5), (0, 1e-5)]
            elif "Elbow" in human:
                costs = [(10*position_scale, 0), (10*position_scale, 0)]
            elif "Wrist" in human:
                costs = [(10*position_scale, 1), (10*position_scale, 5)]
        for table, (pos, rot) in zip(tables, costs):
            table[robot] = [human, pos, rot, [0, 0, 0], offset]
    return {"robot_root_name": "pelvis", "human_root_name": "Hips", "ground_height": 0,
            "human_height_assumption": 1, "use_ik_match_table1": True, "use_ik_match_table2": True,
            "human_scale_table": {name: 1 for name in ANCHORS},
            "ik_match_table1": tables[0], "ik_match_table2": tables[1]}


def add_shoulder_position_tasks(retargeter, position_scale=1):
    import mink
    tasks = []
    for side in ["Left", "Right"]:
        task = mink.FrameTask(frame_name=ANCHORS[side+"Shoulder"], frame_type="body",
                              position_cost=10*position_scale, orientation_cost=0, lm_damping=1)
        retargeter.tasks2.append(task)
        tasks.append((side+"Shoulder", task))
    update = retargeter.update_targets
    def updated(human_data, offset_to_ground=False):
        update(human_data, offset_to_ground)
        for name, task in tasks:
            pos, quat = retargeter.scaled_human_data[name]
            task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(quat), pos))
    retargeter.update_targets = updated


def add_racket_hull_constraints(retargeter):
    """Conservative experiment: full visual racket hull vs visual body hulls.

    Proxy changes are memory-only and only used in IK. Negative hull distances
    are not exact rendered triangle intersection; no source is rejected here.
    """
    import mink
    model = retargeter.model
    racket = model.geom("racket_geom").id
    excluded = {"right_wrist_yaw_link", "right_wrist_pitch_link", "right_wrist_roll_link"}
    geometries = [i for i in range(model.ngeom)
                  if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                  and model.body(int(model.geom_bodyid[i])).name not in excluded
                  and i != racket]
    for i in [racket, *geometries]:
        model.geom_contype[i] = 1
        model.geom_conaffinity[i] = 1
    limit = mink.CollisionAvoidanceLimit(model, geom_pairs=[([racket], geometries)], gain=.5,
                minimum_distance_from_collisions=.005, collision_detection_distance=.1)
    retargeter.ik_limits.append(limit)
    return {"full_racket_hull_constraint": True, "requested_body_visual_geoms": len(geometries),
            "racket_hull_pairs_after_mink_filtering": len(limit.geom_id_pairs),
            "held_hand_wrist_mesh_excluded": sorted(excluded),
            "nativeccd": not bool(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD)}
