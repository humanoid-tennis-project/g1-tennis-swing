# Third-party notices

This repository contains only original glue/experiment code. Upstream projects
are used at runtime (imported or invoked) from separate checkouts and are **not**
redistributed here. Their license texts are copied to `third_party/licenses/`
for reference. Model assets, body models, weights and data have terms separate
from the code and are not included.

| Component | Source / pinned revision | License (checked from the checkout) | How it is used |
|---|---|---|---|
| AdaPT (includes mjlab 1.2.0 fork, G1 tennis task, robot XMLs) | https://github.com/noitom-robotics/AdaPT @ `615b42f8edf8d78b1c8a9601673c70ea67e45ee9` | Apache-2.0 (`third_party/licenses/AdaPT-mjlab-Apache-2.0.txt`) | Imported (`mjlab`, task registry, G1 right-racket XML). Not modified; runtime patches only re-point file paths (`src/windows_mujoco_paths.py`). AdaPT checkpoints/motions are not used or redistributed. |
| mjlab | https://github.com/mujocolab/mjlab (as vendored in AdaPT, version 1.2.0) | Apache-2.0 | As above. |
| RSL-RL (`rsl-rl-lib` 5.0.1, vendored in AdaPT `src/rsl_rl_lib`) | https://github.com/leggedrobotics/rsl_rl | BSD-3-Clause, Copyright (c) 2026 ETH Zurich, NVIDIA CORPORATION & AFFILIATES (`third_party/licenses/RSL-RL-BSD-3-Clause.txt`) | PPO, storage, runner imported unchanged; only the actor mean network is replaced (`src/impact_stage1/policy.py`). |
| GMR (General Motion Retargeting) | https://github.com/YanjieZe/GMR @ `bb1bbe40774794fceb2a7c579a3464a28e68c844` | MIT, Copyright 2025 Yanjieze (`third_party/licenses/GMR-MIT.txt`) | SMPL-X -> G1 IK solver and `smplx_to_g1` config imported from the checkout (`src/retarget_tennis_bvh.py`, `src/retarget_gvhmr_local.py`). |
| GVHMR | https://github.com/zju3dv/GVHMR @ `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1` | Custom: educational, research and non-profit use only; modifications must be open source and non-commercial; attribution notices must be retained (`third_party/licenses/GVHMR-LICENSE.txt`) | `src/recording/runner.py` loads the official `tools/demo/demo.py` and `hmr4d` at runtime inside a GVHMR checkout. Use of that stage is subject to the GVHMR license. GVHMR weights (and YOLO, ViTPose, HMR2 weights) are not included. |
| Mink | https://github.com/kevinzakka/mink (pip `mink==1.3.0`) | Apache-2.0 (`third_party/licenses/mink-Apache-2.0.txt`) | Imported. `src/mink_collision_displacement.py` adapts upstream fix `1933012a7a8756e6c7732a7150155b0c7b46ec7b` and is therefore Apache-2.0. |
| MuJoCo / mujoco-warp | https://github.com/google-deepmind/mujoco (3.12.0 dev build locally, 3.12.1 on cloud), mujoco-warp 3.11.0 | Apache-2.0 | Pip packages, not redistributed. |
| NVIDIA Warp | `warp-lang` 1.16.0 | Apache-2.0 | Pip package, not redistributed. |
| PyTorch, NumPy, SciPy, TensorDict, imageio, Pillow, OpenCV, PyAV, matplotlib | PyPI (versions in `requirements/`) | Their respective licenses | Pip packages, not redistributed. |

## Models and data (not included)

| Item | Terms | Note |
|---|---|---|
| Unitree G1 27-DoF model with racket (`g1_rightracket.xml` + meshes in AdaPT) | Distributed inside AdaPT; no separate asset license file was found next to the model, so its terms were not independently confirmed | Loaded from the AdaPT checkout, not redistributed. The right-hand racket mount is AdaPT's. |
| SMPL / SMPL-X body models | Max Planck Institute non-commercial licenses | Must be obtained by the user. Never redistributed. |
| GVHMR, YOLOv8, ViTPose, HMR2 weights | Their own licenses | Must be obtained by the user. |
| CalTennis videos | Hugging Face dataset `demalenk/caltennis` @ `4af17dd4d398aad4116a35b5a74de4388658d356`; recorded as CC BY-NC 4.0 in our pipeline metadata - check the dataset card | Videos and any per-frame derivative are not included. |
| Derived human motions / G1 references / trained weights | Inherit restrictions of CalTennis and SMPL-X | Not included in this repository; any separate release needs its own notice. |

## Papers whose descriptions were re-implemented (no code copied)

- Humanoid Badminton, arXiv 2609.31840 (Stage 1 hit-conditioned low-level policy, Sec. 3.1, App. A).
- BeyondMimic (observation/reward profile in `src/beyondmimic_profile.py`).
- DeepMimic (end-effector position term, H3), KungfuBot / PBHC (adaptive error kernels, H5; contact-aware reference correction idea, H9).
