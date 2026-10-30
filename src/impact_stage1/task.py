"""Paper-inspired Stage 1 adapter. No ball simulation or upper policy here."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import types
import torch
from .data import sha
from .networks import target_rewards, unit_vector


@dataclass(frozen=True)
class GoalConfig:
    # Deliberately narrower pilot ranges, NOT the paper's badminton parameters.
    randomized: bool = False
    position_radius_m: tuple = (.05, .05, .025)
    velocity_scale: tuple = (.9, 1.1)
    normal_angle_deg: float = 10.
    position_std_m: float = .15
    velocity_std_m_s: float = 2.
    normal_std_rad: float = .35


def align_target_to_runtime(target, motion):
    """Apply exactly the rigid heading change used by the loaded reference.

    The manifest is in source-NPZ coordinates; MotionLoader rotates positions
    around its first root position, and rotates velocity/normal as vectors.
    Require the loader metadata rather than silently assuming no rotation.
    """
    yaw = motion.heading_align_yaw_rad
    pivot = motion.heading_align_pivot_w.to(device=target.device, dtype=target.dtype)
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = target.new_tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return torch.cat((rotation @ (target[:3]-pivot)+pivot,
                      rotation @ target[3:6], rotation @ target[6:9]))


class Targets:
    def __init__(self, env, manifest, settings):
        self.env, self.settings = env, settings
        data = json.loads(Path(manifest).read_text(encoding='utf-8'))
        self.rows = data['motions']
        if not self.rows:
            raise ValueError('No manually confirmed impact motions')
        command = env.command_manager.get_term('motion')
        if command.motion.num_motions != len(self.rows):
            raise ValueError('Impact/reference ordering mismatch')
        device = env.device
        m, k = len(self.rows), max(len(r['events']) for r in self.rows)
        self.frames = torch.full((m, k), 1_000_000, device=device, dtype=torch.long)
        self.base = torch.zeros((m, k, 9), device=device)
        self.counts = torch.tensor([len(r['events']) for r in self.rows], device=device)
        for i, row in enumerate(self.rows):
            for j, e in enumerate(row['events']):
                self.frames[i, j] = e['frame']
                raw = torch.tensor(e['position']+e['velocity']+e['normal'], device=device)
                self.base[i, j] = align_target_to_runtime(raw, command.motion.motions[i])
        self.offsets = torch.zeros((env.num_envs, k, 3), device=device)
        self.scales = torch.ones((env.num_envs, k, 1), device=device)
        self.axes = torch.zeros((env.num_envs, k, 3), device=device)
        self.angles = torch.zeros((env.num_envs, k, 1), device=device)
        self.sample(torch.arange(env.num_envs, device=device))

    def sample(self, ids):
        if not self.settings.randomized:
            self.offsets[ids] = 0
            self.scales[ids] = 1
            self.angles[ids] = 0
            return
        shape = self.offsets[ids].shape
        self.offsets[ids] = (2*torch.rand(shape, device=self.env.device)-1)*torch.tensor(
            self.settings.position_radius_m, device=self.env.device)
        lo, hi = self.settings.velocity_scale
        self.scales[ids] = lo+(hi-lo)*torch.rand((*shape[:-1], 1), device=self.env.device)
        self.axes[ids] = unit_vector(torch.randn(shape, device=self.env.device))
        self.angles[ids] = (2*torch.rand((*shape[:-1], 1), device=self.env.device)-1)*(
            self.settings.normal_angle_deg*torch.pi/180)

    def current(self):
        c = self.env.command_manager.get_term('motion')
        frames = self.frames[c.motion_ids]
        # Retain the current target through the whole hit window. Thereafter use the next.
        progress = c.time_steps_progress
        index = (frames < progress[:, None]-2).sum(-1)
        index = torch.minimum(index, self.counts[c.motion_ids]-1)
        ids = torch.arange(self.env.num_envs, device=self.env.device)
        target = self.base[c.motion_ids, index]
        p = target[:, :3]+self.offsets[ids, index]+self.env.scene.env_origins
        v = target[:, 3:6]*self.scales[ids, index]
        n, axis, angle = target[:, 6:9], self.axes[ids, index], self.angles[ids, index]
        n = n*torch.cos(angle)+torch.cross(axis, n, dim=-1)*torch.sin(angle)+axis*(axis*n).sum(-1, keepdim=True)*(1-torch.cos(angle))
        hit = (frames-progress[:, None]).abs().amin(-1) <= 2
        remaining = (frames[ids, index]-progress).float()/50
        return p, v, n, remaining, hit


def state(env, manifest, settings):
    if not hasattr(env, '_impact_targets'):
        env._impact_targets = Targets(env, manifest, settings)
    return env._impact_targets


def task_observation(env, manifest, settings):
    from mjlab.utils.lab_api.math import quat_apply_inverse
    p, v, n, t, _ = state(env, manifest, settings).current()
    c = env.command_manager.get_term('motion')
    # Relative to the current physical anchor, no future true ball state is used.
    q = c.robot_anchor_quat_w
    return torch.cat((quat_apply_inverse(q, p-c.robot_anchor_pos_w),
                      quat_apply_inverse(q, v), quat_apply_inverse(q, n), t[:, None]), dim=-1)


def outside_hit_reward(env, original_func, original_params, manifest, settings):
    hit = state(env, manifest, settings).current()[-1]
    return original_func(env, **original_params)*(~hit).float()


def hit_target_reward(env, component, manifest, settings):
    from racket_tracking_rewards import racket_quantities
    p, v, n, _, hit = state(env, manifest, settings).current()
    actual_p, _, actual_v, _, actual_n, _ = racket_quantities(env)
    r = target_rewards(actual_p, actual_v, actual_n, p, v, n,
        settings.position_std_m, settings.velocity_std_m_s, settings.normal_std_rad)
    return r[:, component]*hit.float()


def install_reset_hook(env):
    """Called once after environment construction; catches even identical RSI resets."""
    command = env.command_manager.get_term('motion')
    if hasattr(command, '_impact_original_resample'):
        raise ValueError('Impact hook already installed')
    original = command._resample_command
    command._impact_original_resample = original
    def resample(self, ids):
        original(ids)
        if hasattr(env, '_impact_targets'):
            env._impact_targets.sample(ids)
    command._resample_command = types.MethodType(resample, command)


def configure(manifest, num_envs, seed, settings=GoalConfig()):
    from standard_joint_ppo import configure as base_configure
    from mjlab.managers.observation_manager import ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    manifest = Path(manifest).resolve()
    data = json.loads(manifest.read_text(encoding='utf-8'))
    if data.get('format') != 'tennis_impact_stage1_v1' or not data['motions']:
        raise ValueError('Prepare at least one confirmed impact first')
    paths = []
    for row in data['motions']:
        path = Path(row['motion_file']).resolve()
        if sha(path) != row['motion_sha256'] or sha(row['annotation_file']) != row['annotation_sha256']:
            raise ValueError('Impact source hash mismatch')
        paths.append(path)
    cfg, agent = base_configure(paths, num_envs, seed)
    args = dict(manifest=str(manifest), settings=settings)
    for group in ('actor', 'critic'):
        cfg.observations[group].terms['impact_goal'] = ObservationTermCfg(func=task_observation, params=args)
    for name, term in list(cfg.rewards.items()):
        if name.startswith(('motion_', 'local_')):
            cfg.rewards[name] = RewardTermCfg(func=outside_hit_reward, weight=term.weight,
                params=dict(original_func=term.func, original_params=term.params, **args))
    for i, component in enumerate(('position', 'velocity', 'plane')):
        cfg.rewards['impact_'+component] = RewardTermCfg(func=hit_target_reward, weight=1.,
            params=dict(component=i, **args))
    agent.actor.class_name = 'impact_stage1.policy:ImpactActor'
    agent.actor.custom_cfg = dict(latent_dim=8, reference_dim=63, context_dim=87, task_dim=10)
    # Critic remains privileged; no latent bottleneck in the value estimator.
    return cfg, agent, paths
