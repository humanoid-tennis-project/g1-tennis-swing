"""Meaningful CPU tests. Synthetic labels never enter the real annotation store."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
import numpy as np
import torch
from .data import rotate, racket_series, prepare, sha
from .networks import SkillBottleneck, target_rewards
from .task import Targets, GoalConfig, install_reset_hook, align_target_to_runtime

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from project_paths import CACHE
TEST_TMP = (CACHE/'impact_stage1_tests').resolve()
TEST_TMP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CODE/'recording'))
from impact_editor import map_event, ImpactStore


class Stage1Tests(unittest.TestCase):
    def test_source_to_g1_clock(self):
        row = dict(annotation=dict(start_frame=900, end_frame=990), review_local_seconds=[2., 5.])
        e = map_event(row, 911, 1)
        self.assertEqual(e['human_local_frame'], 11)
        self.assertEqual(e['g1_frame'], 18)
        self.assertAlmostEqual(e['review_time_s'], 2+11/30)
        self.assertLessEqual(abs(e['time_rounding_error_s']), .01)
        for f in (899, 990, True):
            with self.assertRaises(ValueError):
                map_event(row, f, 1)

    def test_rigid_offset_velocity(self):
        d = dict(body_names=['wrist'], body_pos_w=np.zeros((3,1,3)),
            body_quat_w=np.tile([1.,0,0,0], (3,1,1)), body_lin_vel_w=np.zeros((3,1,3)),
            body_ang_vel_w=np.tile([0.,0,2.], (3,1,1)))
        mount=dict(parent_body='wrist',center_in_wrist_frame_m=[1,0,0],
            collision_proxy=dict(quaternion_wxyz=[1,0,0,0]))
        r=racket_series(d,mount)
        np.testing.assert_allclose(r['velocity'], np.tile([0,2,0], (3,1)))
        np.testing.assert_allclose(rotate(np.array([np.sqrt(.5),0,0,np.sqrt(.5)]),[1,0,0]), [0,1,0], atol=1e-6)

    def test_encoder_decoder_gradients_and_shapes(self):
        torch.manual_seed(7)
        net=SkillBottleneck(hidden=(32,16))
        x=torch.randn(9,160)
        actions=net(x)
        self.assertEqual(tuple(actions.shape),(9,27))
        self.assertTrue((net.encode(x).abs() <= 1).all())
        actions.square().mean().backward()
        for module in (net.encoder,net.decoder):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()),0)
        z=net.encode(x).detach()
        a=net.lower(x[:,63:150],z)
        modified=x.clone();modified[:,:63]+=100;modified[:,150:]+=100
        torch.testing.assert_close(a,net.lower(modified[:,63:150],z))

    def test_kernel_and_axial_plane(self):
        p=torch.tensor([[0.,0,0]])
        n=torch.tensor([[0.,0,1]])
        torch.testing.assert_close(target_rewards(p,p,n,p,p,-n),torch.ones(1,3))
        self.assertLess(target_rewards(p+1,p,n,p,p,n)[0,0],1e-5)

    def test_multiple_events_and_environment_origins(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            path=Path(directory)/'manifest.json'
            path.write_text(json.dumps(dict(motions=[dict(events=[
                dict(frame=10,position=[1,0,0],velocity=[2,0,0],normal=[0,0,1]),
                dict(frame=30,position=[3,0,0],velocity=[4,0,0],normal=[0,0,1])])])),encoding='utf-8')
            loader=NS(heading_align_yaw_rad=0.,heading_align_pivot_w=torch.zeros(3))
            command=NS(motion=NS(num_motions=1,motions=[loader]),motion_ids=torch.tensor([0,0]),time_steps_progress=torch.tensor([8,13]))
            command._resample_command=lambda ids:None
            env=NS(num_envs=2,device='cpu',scene=NS(env_origins=torch.tensor([[0.,0,0],[10.,0,0]])),
                command_manager=NS(get_term=lambda _:command))
            targets=Targets(env,path,GoalConfig())
            p,v,n,t,hit=targets.current()
            torch.testing.assert_close(p[:,0],torch.tensor([1.,13.]))
            self.assertEqual(hit.tolist(),[True,False])
            self.assertAlmostEqual(t[0].item(),.04,places=6)
            # Exact same-frame resets must also resample, without touching another env.
            env._impact_targets=targets
            targets.settings=GoalConfig(randomized=True)
            before=targets.offsets.clone()
            install_reset_hook(env);command._resample_command(torch.tensor([0]))
            self.assertFalse(torch.equal(targets.offsets[0],before[0]))
            torch.testing.assert_close(targets.offsets[1],before[1])

    def test_target_uses_runtime_heading_and_pivot(self):
        loader=NS(heading_align_yaw_rad=np.pi/2,
                  heading_align_pivot_w=torch.tensor([10.,0.,0.]))
        raw=torch.tensor([11.,0.,1., 2.,0.,0., 1.,0.,0.])
        aligned=align_target_to_runtime(raw,loader)
        torch.testing.assert_close(aligned,torch.tensor([10.,1.,1., 0.,2.,0., 0.,1.,0.]),atol=1e-6,rtol=0)
        with self.assertRaises(AttributeError):
            align_target_to_runtime(raw,NS())

    def test_append_only_annotation_and_training_gate(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            root=Path(directory);data=root/'labels';media=root/'review.mp4';media.write_bytes(b'fixture')
            motion=root/'motion.npz'
            n=100
            np.savez(motion,fps=np.array([50]),joint_pos=np.zeros((n,27)),body_names=np.array(['wrist']),
                body_pos_w=np.zeros((n,1,3)),body_quat_w=np.tile([1.,0,0,0],(n,1,1)),
                body_lin_vel_w=np.zeros((n,1,3)),body_ang_vel_w=np.zeros((n,1,3)))
            (root/'reference_quality.json').write_text(json.dumps(dict(target_model_sha256='model')))
            mount=root/'mount.json';mount.write_text(json.dumps(dict(parent_body='wrist',center_in_wrist_frame_m=[1,0,0],
                target_model_sha256='model',collision_proxy=dict(quaternion_wxyz=[1,0,0,0]))))
            manifest=root/'references.json';manifest.write_text(json.dumps(dict(motions=[dict(id='clip',motion_file='motion.npz',identity_review='approved',pose_review='approved')])))
            catalog=[dict(id='clip',annotation=dict(start_frame=900,end_frame=960,stroke_type='forehand'),
                source=dict(recording_id='rec'),review_local_seconds=[2.,4.],local_review_video='review.mp4',annotation_time_basis='test')]
            (root/'catalog.json').write_text(json.dumps(catalog))
            store=ImpactStore(root,manifest,data)
            empty=prepare(manifest,data,root/'empty',mount)
            self.assertEqual(len(empty['motions']),0)
            payload=dict(id='clip',expected_revision=0,status='confirmed',events=[dict(source_frame=930,evidence='visible_contact',uncertainty_source_frames=1)])
            payload['reviewer']='codex'
            saved=store.save(payload)
            self.assertEqual(saved['reviewer'],'codex')
            with self.assertRaises(RuntimeError):store.save(payload)
            ready=prepare(manifest,data,root/'ready',mount)
            self.assertEqual(ready['motions'][0]['events'][0]['frame'],50)
            payload['expected_revision']=1;payload['events'][0]['evidence']='estimated_swing';store.save(payload)
            blocked=prepare(manifest,data,root/'blocked',mount)
            self.assertEqual(len(blocked['motions']),0)
            self.assertEqual(len(list((data/'clip').glob('revision_*.json'))),2)
            # Exercise the real HTTP route and Range streaming without saving real labels.
            from http.server import ThreadingHTTPServer
            from threading import Thread
            from urllib.request import Request, urlopen
            from urllib.error import HTTPError
            from impact_editor import make_handler
            server=ThreadingHTTPServer(('127.0.0.1',0),lambda *a,**kw:None)
            server.RequestHandlerClass=make_handler(store,server.server_port)
            thread=Thread(target=server.serve_forever,daemon=True);thread.start()
            base=f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(Request(base+'/video?id=clip',headers={'Range':'bytes=1-3'})) as response:
                    self.assertEqual(response.status,206)
                    self.assertEqual(response.read(),b'ixt')
                bad=Request(base+'/api/annotation',data=b'{}',method='POST')
                with self.assertRaises(HTTPError) as error:urlopen(bad)
                self.assertEqual(error.exception.code,403)
                payload['expected_revision']=2;payload['status']='uncertain'
                request=Request(base+'/api/annotation',data=json.dumps(payload).encode(),
                    headers={'Content-Type':'application/json','X-Impact-Editor':'local'},method='POST')
                with urlopen(request) as response:self.assertEqual(json.load(response)['revision'],3)
            finally:
                server.shutdown();server.server_close();thread.join()

    def test_rsl_actor_real_interface(self):
        from tensordict import TensorDict
        from .policy import ImpactActor
        obs=TensorDict(dict(actor=torch.randn(8,160)),batch_size=[8])
        net=ImpactActor(obs,dict(actor=['actor']),'actor',27,hidden_dims=(32,16),
            obs_normalization=True,distribution_cfg=dict(class_name='GaussianDistribution',init_std=.3,std_type='scalar'))
        actions=net(obs,stochastic_output=True)
        # PPO uses the likelihood, not differentiable simulator trajectories.
        loss=-net.get_output_log_prob(actions.detach()).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in net.mlp.encoder.parameters()),0)
        self.assertEqual(tuple(net.output_std.shape),(8,27))

    def test_real_upstream_ppo_updates_both_networks(self):
        from tensordict import TensorDict
        from rsl_rl.algorithms import PPO
        from rsl_rl.models import MLPModel
        from rsl_rl.storage import RolloutStorage
        from .policy import ImpactActor
        torch.manual_seed(29)
        obs=TensorDict(dict(actor=torch.randn(8,160),critic=torch.randn(8,286)),batch_size=[8])
        groups=dict(actor=['actor'],critic=['critic'])
        actor=ImpactActor(obs,groups,'actor',27,hidden_dims=(32,16),obs_normalization=True,
            distribution_cfg=dict(class_name='GaussianDistribution',init_std=.3,std_type='scalar'))
        critic=MLPModel(obs,groups,'critic',1,hidden_dims=(32,16),obs_normalization=True)
        storage=RolloutStorage('rl',8,4,obs,[27],device='cpu')
        alg=PPO(actor,critic,storage,num_learning_epochs=2,num_mini_batches=2,device='cpu')
        before=[actor.mlp.encoder[0].weight.detach().clone(),actor.mlp.decoder[0].weight.detach().clone()]
        for _ in range(4):
            with torch.no_grad():
                actions=alg.act(obs)
                rewards=torch.exp(-actions.square().mean(-1))
                obs=TensorDict(dict(actor=torch.randn(8,160),critic=torch.randn(8,286)),batch_size=[8])
                alg.process_env_step(obs,rewards,torch.zeros(8,dtype=torch.long),{})
        with torch.no_grad():alg.compute_returns(obs)
        losses=alg.update()
        self.assertTrue(all(np.isfinite(v) for v in losses.values()))
        for initial,current in zip(before,[actor.mlp.encoder[0].weight,actor.mlp.decoder[0].weight]):
            self.assertFalse(torch.equal(initial,current.detach()))


if __name__=='__main__':
    unittest.main()
