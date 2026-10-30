"""Dense reference-racket rewards; no assumed ball or impact label (MIT)."""
import torch


def racket_quantities(env):
    from mjlab.utils.lab_api.math import quat_apply
    if not hasattr(env,'_racket_reward_geometry'):
        model=env.sim.mj_model;robot=env.scene['robot'];motion=env.command_manager.get_term('motion')
        site=next(i for i in range(model.nsite) if model.site(i).name.split('/')[-1]=='racket_point')
        geom=next(i for i in range(model.ngeom) if model.geom(i).name.split('/')[-1]=='tennis_racket_collision')
        assert model.site_bodyid[site]==model.geom_bodyid[geom]
        assert abs(model.site_pos[site]-model.geom_pos[geom]).max()<1e-8
        offset=torch.tensor(model.site_pos[site],device=env.device,dtype=torch.float32)
        # Derive cylinder local Z axis from the model, not from visual mesh.
        import numpy as np
        from scipy.spatial.transform import Rotation
        normal=Rotation.from_quat(model.geom_quat[geom],scalar_first=True).apply([0,0,1])
        normal=torch.tensor(np.asarray(normal),device=env.device,dtype=torch.float32)
        env._racket_reward_geometry=(robot.site_names.index('racket_point'),
            robot.body_names.index('right_wrist_yaw_link'),motion.cfg.body_names.index('right_wrist_yaw_link'),offset,normal)
    si,bi,mi,offset,normal=env._racket_reward_geometry
    motion=env.command_manager.get_term('motion');robot=env.scene['robot']
    qref=motion.body_quat_w[:,mi];offref=quat_apply(qref,offset.expand(env.num_envs,-1))
    ref_pos=motion.body_pos_w[:,mi]+offref
    ref_vel=motion.body_lin_vel_w[:,mi]+torch.cross(motion.body_ang_vel_w[:,mi],offref,dim=-1)
    ref_normal=quat_apply(qref,normal.expand(env.num_envs,-1))
    actual_normal=quat_apply(robot.data.body_link_quat_w[:,bi],normal.expand(env.num_envs,-1))
    return robot.data.site_pos_w[:,si],ref_pos,robot.data.site_lin_vel_w[:,si],ref_vel,actual_normal,ref_normal


def racket_error_reward(env,component,std):
    p,rp,v,rv,n,rn=racket_quantities(env)
    if component=='position':error=torch.sum((p-rp)**2,dim=-1)
    elif component=='velocity':error=torch.sum((v-rv)**2,dim=-1)
    elif component=='plane':
        cosine=torch.sum(n*rn,dim=-1).abs().clamp(0,1)
        error=torch.acos(cosine)**2
    else:raise ValueError(component)
    return torch.exp(-error/std**2)


def configure_racket_rewards(cfg, omit=None):
    if omit not in (None,'wrist','position','plane','velocity'):
        raise ValueError(omit)
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.tasks.adapt_tennis import mdp
    cfg.rewards['local_right_wrist']=RewardTermCfg(func=mdp.motion_joint_position_error_exp,weight=1.,
        params=dict(command_name='motion',std=.4,joint_names=('right_wrist_roll_joint','right_wrist_pitch_joint','right_wrist_yaw_joint')))
    for component,weight,std in [('position',1.,.3),('plane',.5,.4),('velocity',.5,3.)]:
        cfg.rewards['local_racket_'+component]=RewardTermCfg(func=racket_error_reward,weight=weight,
            params=dict(component=component,std=std))
    if omit is not None:
        key='local_right_wrist' if omit=='wrist' else 'local_racket_'+omit
        cfg.rewards[key].weight=0.


def capture_reference_velocity(env,state):
    motion=env.command_manager.get_term('motion')
    state['ref_body_lin_vel']=motion.body_lin_vel_w[0].detach().cpu().numpy().copy()
    state['ref_body_ang_vel']=motion.body_ang_vel_w[0].detach().cpu().numpy().copy()
    return state


def cpu_racket_metrics(model,data,state,body_names):
    """Fresh CPU state, cached applied reference; same metric as repeatability."""
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    site=next(i for i in range(model.nsite) if model.site(i).name.split('/')[-1]=='racket_point')
    geom=next(i for i in range(model.ngeom) if model.geom(i).name.split('/')[-1]=='tennis_racket_collision')
    wrist=body_names.index('right_wrist_yaw_link')
    rotation=Rotation.from_quat(state['ref_body_quat'][wrist],scalar_first=True)
    offset=rotation.apply(model.site_pos[site])
    ref_vel=state['ref_body_lin_vel'][wrist]+np.cross(state['ref_body_ang_vel'][wrist],offset)
    velocity=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_SITE,site,velocity,0)
    normal=(rotation*Rotation.from_quat(model.geom_quat[geom],scalar_first=True)).apply([0,0,1])
    actual_normal=data.geom_xmat[geom].reshape(3,3)[:,2]
    return dict(racket_velocity_error_m_s=float(np.linalg.norm(velocity[3:]-ref_vel)),
        racket_actual_speed_m_s=float(np.linalg.norm(velocity[3:])),racket_reference_speed_m_s=float(np.linalg.norm(ref_vel)),
        racket_plane_error_deg=float(np.rad2deg(np.arccos(np.clip(abs(np.dot(normal,actual_normal)),0,1)))))
