"""Shared configuration and state measurement for local AdaPT experiments (MIT)."""
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

TASK = "Mjlab-ServeTracking-Flat-Unitree-G1-Stage1-RandomDt"
ROOT = Path(__file__).resolve().parent


def apply_tracking_profile(cfg, profile):
    """Explicit local curriculum for an unannotated video clip; upstream intact."""
    if profile == "original":
        return
    if profile != "video-nominal":
        raise ValueError(profile)
    command = cfg.commands["motion"]
    cfg.events = {}
    command.pose_range = {}
    command.velocity_range = {}
    command.joint_position_range = (0, 0)
    command.motion_warmup_steps = 0
    command.dt_delta_range = (0, 0)
    # No measured contact frame or racket face exists in this clip. Keep
    # ordinary body/velocity/wrist tracking and all original failure thresholds.
    for name in ("motion_keyframe_hit_arm_joint_pos", "motion_keyframe_racket_site_pos"):
        cfg.rewards[name].weight = 0


def json_default(value):
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    return str(value)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, default=json_default, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_config(motion_file, keyframe_time, num_envs, seed, play=False, condition="nominal"):
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.tasks.adapt_tennis.config.g1.env_cfgs import apply_racket_hand
    cfg = load_env_cfg(TASK, play=play)
    cfg.seed = seed
    cfg.scene.num_envs = num_envs
    command = cfg.commands["motion"]
    command.motion_file = str(Path(motion_file).resolve())
    command.debug_vis = False
    apply_racket_hand(cfg, "right")
    for name in ("motion_keyframe_hit_arm_joint_pos", "motion_keyframe_racket_site_pos"):
        cfg.rewards[name].params["keyframe_times"] = (keyframe_time,)
    if play:
        # Every evaluation starts at frame 0. Training keeps original adaptive RSI.
        command.sampling_mode = "start"
        command.motion_warmup_steps = 0
        if condition == "nominal":
            cfg.events = {}
            command.pose_range = {}
            command.velocity_range = {}
            command.joint_position_range = (0, 0)
        elif condition != "reset-perturbed":
            raise ValueError(condition)
    agent = load_rl_cfg(TASK)
    agent.logger = "tensorboard"
    agent.upload_model = False
    agent.save_interval = 100
    return cfg, agent


def check_motion_layout(env, motion_file, motion_index=0):
    """Check names and FK against the actual actuator-equipped runtime model."""
    arrays = np.load(motion_file, allow_pickle=False)
    robot = env.scene["robot"]
    # Newer mujoco-warp stores constant model arrays with a singleton world
    # dimension. The pinned mjlab copies these into per-env joint limit tensors
    # without expanding them. Preserve every value while fixing that shape.
    limit_expansions = []
    for field in ("default_joint_pos_limits", "joint_pos_limits", "soft_joint_pos_limits"):
        limits = getattr(robot.data, field)
        if limits.shape[0] == 1 and env.num_envs > 1:
            setattr(robot.data, field, limits.expand(env.num_envs, -1, -1).clone())
            limit_expansions.append(field)
    if list(arrays["joint_names"]) != list(robot.joint_names):
        raise ValueError("Reference joint order does not match runtime robot")
    motion = env.command_manager.get_term("motion")
    if list(arrays["body_names"]) != list(motion.cfg.body_names):
        raise ValueError("Reference body order does not match tracked body order")
    if env.sim.mj_model.nu != 27:
        raise ValueError("Runtime environment must have 27 actuators")
    # World heading alignment in MotionLoader is intentional. Verify internal
    # reference is self-consistent using its transformed root pose.
    model, data = env.sim.mj_model, mujoco.MjData(env.sim.mj_model)
    names = [model.joint(i).name.split("/")[-1] for i in range(model.njnt)]
    bodies = [model.body(i).name.split("/")[-1] for i in range(model.nbody)]
    loader = motion.motion.motions[motion_index]
    errors = []
    for frame in [0, len(arrays["joint_pos"]) // 2, len(arrays["joint_pos"]) - 1]:
        data.qpos[:] = model.qpos0
        data.qpos[:3] = loader.body_pos_w[frame, 0].cpu().numpy()
        data.qpos[3:7] = loader.body_quat_w[frame, 0].cpu().numpy()
        for col, name in enumerate(arrays["joint_names"]):
            data.qpos[model.jnt_qposadr[names.index(name)]] = loader.joint_pos[frame, col].item()
        mujoco.mj_forward(model, data)
        expected = loader.body_pos_w[frame].cpu().numpy()
        errors.append(float(np.max(np.linalg.norm(data.xpos[[bodies.index(n) for n in motion.cfg.body_names]] - expected, axis=1))))
    if max(errors) > 1e-4:
        raise ValueError(f"Runtime FK/reference disagreement: {errors}")
    return {"nu": model.nu, "nq": model.nq, "tracked_bodies": len(motion.cfg.body_names),
            "reference_fk_max_error_m": max(errors), "constant_joint_limit_expansions": limit_expansions}


def make_terminal_recorder():
    from mjlab.managers.recorder_manager import RecorderTerm
    class TerminalCapture(RecorderTerm):
        def record_pre_reset(self, env_ids):
            env = self._env
            if 0 in env_ids.tolist():
                env._evaluation_terminal = snapshot(env)
    return TerminalCapture


def snapshot(env):
    """Capture qpos/qvel and reference BEFORE automatic terminal reset.

    Recompute FK on a separate CPU MjData: GPU derived transforms at pre-reset
    are one physics substep old. Never write CPU state back to the simulator.
    """
    motion = env.command_manager.get_term("motion")
    reasons = [name for name in env.termination_manager.active_terms
               if env.termination_manager.get_term(name)[0].item()]
    state = {
        "qpos": env.sim.data.qpos[0].detach().cpu().numpy().copy(),
        "qvel": env.sim.data.qvel[0].detach().cpu().numpy().copy(),
        "joint_pos": motion.robot_joint_pos[0].detach().cpu().numpy().copy(),
        "ref_joint_pos": motion.joint_pos[0].detach().cpu().numpy().copy(),
        "ref_body_pos": motion.body_pos_w[0].detach().cpu().numpy().copy(),
        "ref_body_quat": motion.body_quat_w[0].detach().cpu().numpy().copy(),
        "motion_time_s": float(motion.current_time_seconds()[0].item()),
        "termination_reason": ",".join(reasons),
        "failure": bool(env.termination_manager.terminated[0].item()),
    }
    # Keep actual actuator signals before an automatic terminal reset. These
    # cannot in general be reconstructed from a reference or from qpos alone.
    for name in ('ctrl','actuator_force','qfrc_actuator'):
        value=getattr(env.sim.data,name,None)
        if value is not None:
            sample=value[0]  # mjlab uses a TorchArray proxy; indexing yields Tensor.
            if isinstance(sample,torch.Tensor):state[name]=sample.detach().cpu().numpy().copy()
    return state


def measurements(model, data, state, body_names):
    data.qpos[:], data.qvel[:] = state["qpos"], state["qvel"]
    mujoco.mj_forward(model, data)
    body_ids = {model.body(i).name.split("/")[-1]: i for i in range(model.nbody)}
    site_id = next(i for i in range(model.nsite) if model.site(i).name.split("/")[-1] == "racket_point")
    geom_id = next(i for i in range(model.ngeom) if model.geom(i).name.split("/")[-1] == "tennis_racket_collision")
    root_id, wrist_id = body_ids["pelvis"], body_names.index("right_wrist_yaw_link")
    ref_root = state["ref_body_pos"][body_names.index("pelvis")]
    ref_root_rotation = Rotation.from_quat(state["ref_body_quat"][body_names.index("pelvis")], scalar_first=True)
    root_rotation = Rotation.from_quat(data.xquat[root_id], scalar_first=True)
    ref_wrist_rotation = Rotation.from_quat(state["ref_body_quat"][wrist_id], scalar_first=True)
    ref_racket = state["ref_body_pos"][wrist_id] + ref_wrist_rotation.apply(model.site_pos[site_id])
    delta_rotation = root_rotation * ref_root_rotation.inv()
    dm = delta_rotation.as_matrix()
    yaw = np.arctan2(dm[1, 0], dm[0, 0])
    yaw_rotation = Rotation.from_euler("z", yaw)
    aligned_root = np.array([*data.xpos[root_id, :2], ref_root[2]])
    aligned_racket = aligned_root + yaw_rotation.apply(ref_racket - ref_root)
    # The site's local orientation is identity. Racket face belongs to the
    # CYLINDER GEOM's local Z axis, which has a different XML rotation.
    ref_normal = (ref_wrist_rotation * Rotation.from_quat(model.geom_quat[geom_id], scalar_first=True)).apply([0, 0, 1])
    actual_normal = data.geom_xmat[geom_id].reshape(3, 3)[:, 2]
    aligned_bodies = aligned_root + yaw_rotation.apply(state["ref_body_pos"] - ref_root)
    body_errors = np.linalg.norm(data.xpos[[body_ids[n] for n in body_names]]-aligned_bodies, axis=1)
    joint_error = state["joint_pos"] - state["ref_joint_pos"]
    contacts = [(int(c.geom1), int(c.geom2), float(c.dist)) for c in data.contact]
    return {
        "joint_rmse_rad": float(np.sqrt(np.mean(joint_error**2))),
        "body_aligned_rmse_m": float(np.sqrt(np.mean(body_errors**2))),
        "base_height_error_m": float(abs(data.xpos[root_id, 2] - ref_root[2])),
        "base_tilt_rad": float(np.arccos(np.clip(root_rotation.as_matrix()[2, 2], -1, 1))),
        "racket_world_error_m": float(np.linalg.norm(data.site_xpos[site_id] - ref_racket)),
        "racket_aligned_error_m": float(np.linalg.norm(data.site_xpos[site_id] - aligned_racket)),
        "racket_normal_world_error_rad": float(np.arccos(np.clip(np.dot(actual_normal, ref_normal), -1, 1))),
        "racket_normal_aligned_error_rad": float(np.arccos(np.clip(np.dot(actual_normal, yaw_rotation.apply(ref_normal)), -1, 1))),
        "max_geometric_penetration_m": max([max(0, -d) for _, _, d in contacts], default=0),
        "finite": bool(np.isfinite(state["qpos"]).all() and np.isfinite(state["qvel"]).all()),
    }
