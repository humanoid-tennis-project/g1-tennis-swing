# Tennis video to humanoid swings (Unitree G1, simulation)

Code for a course project: a simulated Unitree G1 humanoid learns tennis swings
from human references built from public practice video (CalTennis, one camera per clip),
with a shared low-level policy that is asked to reach a requested racket state
(position, velocity, face normal) at a given hit time.

Video: (YouTube URL) / Report: see the course submission / Data: (Hugging Face URLs)

- **Simulator / RL**: MuJoCo (mujoco-warp) through AdaPT/mjlab, PPO from RSL-RL.
- **References**: CalTennis video -> GVHMR (human 3D motion) -> GMR retargeting to G1
  (27 joints, 50 Hz); 474 rally strokes are used for training.
- **Policy**: a Stage 1 "hit-conditioned" low-level policy inspired by
  *Humanoid Badminton* (arXiv 2609.31840, Sec. 3.1 / App. A). This is an
  independent re-implementation, **not** the authors' code.

No ball, no ball contact physics, no high-level policy and no rally are
implemented. Racket states are computed from a rigid racket fixed to the robot
wrist, not measured from the video.

## Pipeline

```
CalTennis video (30 Hz)
  -> src/recording/boundary_editor.py   manual stroke interval selection (local web UI)
  -> src/recording/runner.py            official GVHMR inference (run inside a GVHMR checkout / docker/gvhmr.Dockerfile)
  -> src/recording/selected_export.py   export SMPL joints + parameters per selected interval
  -> src/retarget_selected_corpus.py    batch driver (calls src/adapt_selected_human.py,
                                        src/retarget_gvhmr_local.py = GMR SMPL-X -> G1, src/audit_g1_reference_candidate.py)
  -> src/check_selected_g1_loader.py    verify references load in the mjlab MotionLoader
  -> src/recording/impact_editor.py     manual hit-time annotation per stroke (local web UI)
  -> python -m impact_stage1.data       impact manifest: racket goal (p, v, n) at each annotated hit
  -> python -m impact_stage1.train      Stage 1 PPO (Encoder -> z8 -> Decoder, RSL-RL PPO)
  -> src/evaluate_impact_parallel.py    fixed-20 / all-948 physical evaluation
  -> src/summarize_impact_center.py, src/compare_impact_*.py   paired comparisons
  -> src/audit_impact_root_drift.py, src/audit_impact_foot_slide.py   error attribution
  -> src/build_foot_locked_references.py   stance-foot-locked reference copies (H9)
  -> src/diagnose_impact_followup.py    side-by-side videos rendered from saved states
```

`scripts/run_followup_local.py` chains control evaluation, one bounded training
continuation, restore verification, and paired evaluations. The configuration
history of the improvement steps H1-H9 is in `configs/experiment_lineage.json`; the final
policy adds an action-std floor of 0.1 (`--min-action-std 0.1`, H10) and continues with the same
settings to 16,611 updates (H11). `analysis/` holds the scripts behind the reported numbers.

### Stage 1 model

```
reference 63 + G1 proprioception 87 + hit goal 10 = 160
      -> Encoder 512/256/128 -> tanh(z, 8 dims)
proprioception 87 + z 8 = 95
      -> Decoder 512/256/128 -> 27 joint position targets
Critic: 276 privileged + 10 goal = 286 inputs
```

Inside a 5-frame window around each annotated hit, rewards on racket position,
velocity and face normal replace the dense racket/wrist tracking terms; outside
it, whole-body tracking (plus the H1-H8 terms in `src/impact_stage1/reward_variants.py`)
applies. Reward variants are explicit and resuming across variants is guarded by
`src/impact_stage1/scope_guard.py` and `src/impact_stage1/approved_reward_forks.json`
(keyed by SHA-256 of the source, so keep those files byte-identical).

## Repository layout

```
src/                     Python modules (put on PYTHONPATH)
  impact_stage1/         Stage 1 task, networks, PPO actor, rewards, training, evaluation, tests
  recording/             GVHMR inference wrapper, export, interval and hit-time annotation UIs
  *.py                   retargeting, shared tracking utilities, evaluation and audits
scripts/                 run_followup_local.py (bounded local experiment driver)
configs/                 racket_mount.json, experiment_lineage.json
requirements/            observed environment locks
docker/                  GVHMR inference image (build context: a GVHMR checkout)
tests/                   CPU unit tests
third_party/licenses/    upstream license texts
```

## How to reproduce

Tested only on Windows 11 + one NVIDIA GPU (training) and CPU (retargeting).
Nothing from upstream projects, no data and no model weights are included.

1. Check out the upstream projects at the pinned revisions (see
   `THIRD_PARTY_NOTICES.md`) under `upstream/` (git-ignored) or anywhere, and set
   ```
   ADAPT_ROOT=<AdaPT checkout>   GMR_ROOT=<GMR checkout>   TENNIS_CACHE=<ASCII-only writable dir>
   PYTHONPATH=<this repo>/src
   ```
   On Windows, NVRTC needs ASCII paths; `src/windows_mujoco_paths.py` uses NTFS
   short names and `TENNIS_CACHE` for the Warp kernel cache.
2. Training environment: Python 3.12, `pip install -e $ADAPT_ROOT -e $ADAPT_ROOT/src/rsl_rl_lib`,
   then the pins in `requirements/adapt-train-observed.txt`.
   Retargeting environment: `requirements/retarget-cpu.lock.txt` plus GMR.
3. Obtain CalTennis videos and the SMPL/SMPL-X body models and GVHMR weights
   yourself under their own licenses (none are redistributed here).
4. Human motion: run `src/recording/runner.py --video ... --output ... --source ...`
   inside the GVHMR container, then `src/recording/selected_export.py`.
5. Retarget: `python src/retarget_selected_corpus.py --selection <selection.json> --body-model <SMPLX_NEUTRAL.npz> --output <refs>`.
6. Annotate hit frames with `python src/recording/impact_editor.py --root ... --manifest ... --data ...`
   (serves on 127.0.0.1 only), then
   `python -m impact_stage1.data --manifest <refs>/reference_manifest.json --annotations <labels> --output <prepared> --skill-group rally --allow-estimated --max-uncertainty-s 0.5 --allow-boundary-windows`.
7. Train (short smoke run first):
   `python -m impact_stage1.train --impact-manifest <prepared>/impact_manifest.json --output runs/smoke --num-envs 256 --updates 2 --seconds 120 --randomized-goals`.
8. Evaluate: `python src/evaluate_impact_parallel.py --manifest ... --checkpoint ... --output ... --indices all --reward-variant <variant>`.
9. Follow-up trials: `scripts/run_followup_local.py` (see its docstring for the H9 example).

CPU tests: `python -m unittest impact_stage1.test_stage1` (from `src/`; the last
test needs the AdaPT environment) and `python -m unittest discover tests` with
`src` on `PYTHONPATH`.

## Results

Final policy, 474 training references x 2 start times (0 s, 0.04 s) = 948 trials, nominal goals,
no observation or physics noise. The evaluation uses the training references (no held-out motions).

| Measure | Value |
|---|---:|
| Trials still running at the hit time | 735 / 948 (350 references from both starts) |
| Trials running to the end | 636 / 948 |
| Racket centre to target at the hit, median / mean (735 trials) | 32.0 cm / 35.6 cm |
| Same, with the root aligned to the reference | 25.6 cm (median) |
| Within 15 cm (racket face radius) | 76 / 948 |
| Racket speed at the hit, robot / reference (median) | 5.7 / 7.1 m/s |
| Joint-angle error at the hit (median) | 19.4 deg |

The policy does **not** follow the racket target yet. With the training-time goal randomisation at
evaluation, the slope of racket shift on goal shift is 0.12 (position), 0.01 (velocity) and 0.05
(face normal); assuming the racket ignores the target gives a median error of 32.6 cm vs 32.3 cm
observed. Tripling the randomisation range gives the same slopes. The policy replays the references.

Hit-time labels: two policies trained from scratch for 3,000 updates (one seed each) differ only in
annotated vs automatic (peak racket speed) hit times. On 458 paired trials the racket's closest approach
falls within +-0.04 s of the annotated contact in 73.4% vs 65.9% (+7.4 points, 95% bootstrap interval
over references +2.6 to +12.5); no difference in position error was detected. Seed-to-seed variation is
not measured, so this is preliminary.

## Limitations

- Simulation only; no sim-to-real transfer, no ball, no ball-contact physics, no rally.
- The policy does not follow the racket target (see Results); outside the hit window the reward pays for
  tracking the unshifted reference, which may make ignoring the target the better choice (not verified).
- Evaluated on the training references only; one training seed per policy.
- Racket pose is a rigid attachment to the wrist, not measured from video.
- Hit times are manual annotations (part visible contact, part estimated from the
  swing; uncertainty up to 0.5 s is admitted for training).
- Video-derived references contain foot sliding and non-physical airborne phases;
  `build_foot_locked_references.py` is a simple rigid correction, not a physics fix.
- Most of the development used a single Windows machine and one cloud GPU type;
  local (MuJoCo 3.12.0) and cloud (3.12.1) runs are treated as different runtimes.
- Serve strokes (45) are excluded from the rally policy; a serve policy is not trained.

## Licenses

Original code: MIT (`LICENSE`), except `src/mink_collision_displacement.py`
(Apache-2.0). Upstream projects, robot/body models, CalTennis data and anything
derived from them keep their own terms; see `THIRD_PARTY_NOTICES.md`.

---

## 日本語概要

公開データセット CalTennis のテニス練習の動画（1本ごとに1台のカメラ）から人体動作をGVHMRで推定し、GMRでUnitree G1（27関節・50Hz）へ変換した参照474本を使い、MuJoCo/mjlab上でRSL-RLのPPOにより人らしいスイングを学習するコードです。下位方策は Humanoid Badminton（arXiv 2609.31840）のStage 1を参考にした独自実装で、指定した時刻にラケットの位置・速度・面の向きを実現するよう、Encoder（z8）とDecoderを共同学習します。

- 再現手順は上の「How to reproduce」を参照。上流コード・データ・重みは同梱していません。
- 球・接触物理・上位方策・ラリーは未実装。ラケット姿勢は手首への剛体固定から計算した推定値です。
- 打撃時刻は手作業の注釈（一部はスイングからの推定）です。
- 改善の経緯（H1〜H9）は `configs/experiment_lineage.json`。最終の方策は行動の標準偏差に下限0.1を付け（H10）、設定を変えずに16,611更新まで学習した（H11）。
- 結果（948試行）：打撃時刻まで続いた 735、打撃時刻のラケット位置の誤差は中央値 32.0 cm、15 cm 以内は 76。ラケット目標をずらしてもほぼ追従しない（位置の傾き 0.12）。今の方策は手本をなぞっている。
