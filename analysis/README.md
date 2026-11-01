# Analysis scripts behind the report

These scripts produced the numbers in the final report. They read the saved states of the
evaluations (`simulated_states.npz`, `episodes.jsonl`) and do not change the policy.
Paths inside the scripts follow the original experiment layout
(`<experiments>/impact_improvement_20261030_v1/...`, `<experiments>/randomized_goal_eval_20261101_v1/...`);
set them to your own run folders before use.

## randomized_goals/

| Script | What it computes |
|---|---|
| `evaluate_impact_randomized.py` | Copy of `src/evaluate_impact_parallel.py` that samples the training-time goal randomisation at evaluation (`--goal-range-scale k` multiplies the ranges) and records each trial's goal shift |
| `compare.py` | Nominal vs randomised evaluation summary |
| `adaptation.py` | Horizontal racket shift vs goal shift at the hit (forward kinematics on CPU) |
| `extended_eval.py` | Following slopes for position, velocity and face normal; a 15 cm contact criterion from the racket collision cylinder; the ignore-the-target baseline; failure breakdown |
| `tracking_summary.py` | Whole-body joint and body-position errors and per-reference counts of the final evaluation |
| `hit_timing.py` | Annotated hit times vs the time of peak racket speed |

## hit_time/

Two policies trained from scratch with identical settings (seed 7, 3,000 updates); they differ only in the hit-time labels
(annotated vs time of peak racket speed).

| Script | What it does |
|---|---|
| `build_auto_manifest.py` | Builds the automatic hit-time manifest (targets recomputed with `impact_stage1.data.racket_series`) |
| `run_arms.sh`, `eval_arms.sh` | Train and evaluate both arms |
| `eval_current.py` | Scores trials against the annotated contact on the 326 references with visible contact |
| `score_arms.py`, `paired_stats.py` | Per-arm scores and the paired comparison with a bootstrap over references |
