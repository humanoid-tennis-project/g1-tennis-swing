"""Small bounded double-support IK for a fixed initial pose (MIT).

Keeps upper-body angles and root orientation fixed; adjusts root height and
12 leg angles. This is initialization, not a whole-trajectory physics filter.
"""
import mujoco
import numpy as np
from scipy.optimize import least_squares


def foot_geometry(model):
    feet = []
    for side in ("left", "right"):
        body = next(i for i in range(model.nbody)
                    if model.body(i).name.split("/")[-1] == side + "_ankle_roll_link")
        geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body]
        capsules = [i for i in geoms if "foot" in model.geom(i).name
                    and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CAPSULE]
        mesh = next(i for i in geoms if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH)
        mesh_id = model.geom_dataid[mesh]
        vertices = model.mesh_vert[model.mesh_vertadr[mesh_id]:
                                   model.mesh_vertadr[mesh_id] + model.mesh_vertnum[mesh_id]].copy()
        feet.append((side, body, capsules, mesh, vertices))
    return feet


def capsule_bottom(model, data, ids):
    return min(float(data.geom_xpos[i, 2] - model.geom_size[i, 0]
                     - model.geom_size[i, 1] * abs(data.geom_xmat[i].reshape(3, 3)[2, 2])) for i in ids)


def report(model, data, feet):
    result = {}
    for side, body, capsules, mesh, vertices in feet:
        world = vertices @ data.geom_xmat[mesh].reshape(3, 3).T + data.geom_xpos[mesh]
        result[side] = {
            "collision_bottom_m": capsule_bottom(model, data, capsules),
            "visual_mesh_bottom_m": float(world[:, 2].min()),
            "foot_normal_world": data.xmat[body].reshape(3, 3)[:, 2].tolist(),
            "ankle_world_position_m": data.xpos[body].tolist(),
        }
    return result


def solve_initial_support(model, initial_qpos, leg_limits=None):
    feet = foot_geometry(model)
    data = mujoco.MjData(model)
    original = np.asarray(initial_qpos).copy()
    leg_joints = [i for i in range(model.njnt)
                  if any(part in model.joint(i).name for part in ("hip_", "knee_", "ankle_"))]
    assert len(leg_joints) == 12
    indices = model.jnt_qposadr[leg_joints]
    data.qpos[:] = original
    mujoco.mj_forward(model, data)
    before = report(model, data, feet)
    target_xy = [data.xpos[f[1], :2].copy() for f in feet]
    # A smooth ankle-height target avoids the nonsmooth minimum switching
    # between capsule endpoints while the foot is being flattened.
    target_z = []
    for _, _, capsules, _, _ in feet:
        bottoms = []
        for gid in capsules:
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[gid])
            bottoms.append(model.geom_pos[gid, 2] - model.geom_size[gid, 0]
                           - model.geom_size[gid, 1] * abs(rotation.reshape(3, 3)[2, 2]))
        target_z.append(-min(bottoms) - .0001)
    x0 = np.r_[original[2], original[indices]]
    limits = model.jnt_range[leg_joints] if leg_limits is None else np.asarray(leg_limits)
    lower = np.r_[original[2] - .10, limits[:, 0]]
    upper = np.r_[original[2] + .03, limits[:, 1]]
    def apply(x):
        data.qpos[:] = original
        data.qpos[2] = x[0]
        data.qpos[indices] = x[1:]
        mujoco.mj_forward(model, data)
    def residual(x):
        apply(x)
        values = []
        for f, xy, z in zip(feet, target_xy, target_z):
            _, body, capsules, _, _ = f
            values.append(1000 * (data.xpos[body, 2] - z))
            values.extend(300 * data.xmat[body].reshape(3, 3)[:2, 2])
            values.extend(100 * (data.xpos[body, :2] - xy))
        values.extend(.15 * (x[1:] - x0[1:]))
        values.append(2 * (x[0] - x0[0]))
        return np.asarray(values)
    solved = least_squares(residual, np.clip(x0, lower + 1e-8, upper - 1e-8),
                           bounds=(lower, upper), max_nfev=200, diff_step=1e-4,
                           xtol=1e-10, ftol=1e-10, gtol=1e-10)
    apply(solved.x)
    after = report(model, data, feet)
    if (not solved.success
            or any(abs(f["collision_bottom_m"] + .0001) > .0005 for f in after.values())
            or any(np.linalg.norm(f["foot_normal_world"][:2]) > .005 for f in after.values())):
        raise ValueError(f"Grounding IK did not meet 0.5mm tolerance: {solved.message}; {after}")
    return data.qpos.copy(), {"before": before, "after": after,
        "optimizer_success": bool(solved.success), "optimizer_message": str(solved.message),
        "root_height_delta_m": float(data.qpos[2] - original[2]),
        "joint_deltas_rad": {model.joint(j).name: float(data.qpos[k] - original[k]) for j, k in zip(leg_joints, indices)},
        "maximum_leg_joint_delta_rad": float(np.max(np.abs(data.qpos[indices] - original[indices]))),
        "scope": "Fixed initial double support; original reference and velocities unchanged; not dynamic feasibility"}
