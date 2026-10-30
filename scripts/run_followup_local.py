"""One bounded Stage 1 follow-up trial on a local GPU, with paired evaluations.

Generalised from the local campaign drivers used for hypotheses H4-H9. The
original drivers hard-coded experiment folders and frozen code bundles; here
every input is an argument. The sequence is unchanged:

  1. evaluate the parent checkpoint (control) on the fixed-20 and all-948 sets
  2. continue PPO from the parent with the chosen reward variant / data fork
  3. verify that NN, optimizer, normalizer, sampler and RNG state were restored
  4. evaluate the intermediate and final checkpoints on the same conditions
  5. write paired comparisons (matched sim-time prefixes and hit centres)

Example (H9: stance-foot-locked references, continuing from the H8 final model):

  python scripts/run_followup_local.py \
      --manifest data/impact_manifest_footlock.json \
      --parent runs/h8/train/final.pt --parent-update 11951 \
      --updates 1200 --seconds 3600 --num-envs 2048 \
      --variant joint_pose_ee_hitbalanced_nonracket_rootvel_adaptive_v4 \
      --train-flag=--fork-parent-scope --train-flag=--foot-lock-data-fork \
      --output runs/h9
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))

FIXED20 = '3,262,4,234,0,184,50,361,309,305'  # 10 clips x 2 start times
STAGES = [('eval20', FIXED20, 20, 600), ('eval948', 'all', 948, 1800)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--manifest', type=Path, required=True, help='impact manifest (impact_stage1.data output)')
    p.add_argument('--parent', type=Path, required=True, help='parent checkpoint to continue from')
    p.add_argument('--parent-update', type=int, required=True, help='total PPO updates already in the parent')
    p.add_argument('--updates', type=int, default=1200)
    p.add_argument('--seconds', type=float, default=3600)
    p.add_argument('--num-envs', type=int, default=2048)
    p.add_argument('--variant', required=True, help='reward variant name, see src/impact_stage1/reward_variants.py')
    p.add_argument('--train-flag', action='append', default=[],
                   help='extra flag forwarded to impact_stage1.train (repeatable)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--python', default=sys.executable, help='interpreter of the AdaPT/mjlab environment')
    p.add_argument('--skip-control', action='store_true', help='reuse an existing control evaluation')
    p.add_argument('--control', type=Path, help='folder holding control_eval20/control_eval948 (default: output)')
    a = p.parse_args()

    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    control_root = (a.control or out).resolve()
    env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', WANDB_MODE='disabled',
               PYTHONPATH=os.pathsep.join([str(SRC), os.environ.get('PYTHONPATH', '')]))

    def write(name, value):
        path = out / name
        tmp = Path(str(path) + '.writing')
        tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
        os.replace(tmp, path)

    def execute(name, args, timeout):
        write('dashboard.json', dict(state='running', stage=name, variant=a.variant, updated=time.time()))
        with (out/(name+'.log')).open('x', encoding='utf-8') as log, \
             (out/(name+'_error.log')).open('x', encoding='utf-8') as err:
            done = subprocess.run([a.python, *map(str, args)], cwd=SRC, env=env, stdout=log, stderr=err,
                                  timeout=timeout)
        if done.returncode:
            raise RuntimeError(name+' failed; inspect preserved artifacts before resuming')

    def evaluate(checkpoint, stage, root):
        key = stage.rsplit('_', 1)[1]
        _, indices, count, seconds = next(s for s in STAGES if s[0] == key)
        execute(stage, [SRC/'evaluate_impact_parallel.py', '--manifest', a.manifest.resolve(),
                        '--checkpoint', Path(checkpoint).resolve(), '--output', root/stage,
                        '--indices', indices, '--batch-size', '32', '--max-wall-seconds', str(seconds),
                        '--reward-variant', a.variant], seconds+300)
        result = json.loads((root/stage/'summary.json').read_text(encoding='utf-8'))
        if result['evaluated'] != count or result['requested'] != count:
            raise RuntimeError('Partial evaluation '+stage)
        from summarize_impact_center import summarize
        center = summarize(root/stage)
        write(stage+'_statistics.json', {k: v for k, v in center.items() if k != 'conditions'})
        return count

    write('plan.json', dict(parent=str(a.parent), parent_update=a.parent_update, parent_sha256=sha(a.parent),
                            updates=a.updates, training_seconds=a.seconds, num_envs=a.num_envs,
                            variant=a.variant, train_flags=a.train_flag, manifest_sha256=sha(a.manifest)))
    if not a.skip_control:
        for name, *_ in STAGES:
            evaluate(a.parent, 'control_'+name, out)
    execute('train', ['-m', 'impact_stage1.train', '--impact-manifest', a.manifest.resolve(),
                      '--output', out/'train', '--num-envs', str(a.num_envs), '--updates', str(a.updates),
                      '--seconds', str(a.seconds), '--randomized-goals', '--reward-variant', a.variant,
                      '--resume', a.parent.resolve(), *a.train_flag], int(a.seconds)+600)
    summary = json.loads((out/'train/summary.json').read_text(encoding='utf-8'))
    final_update = a.parent_update + summary['updates']
    from verify_impact_parent import main as verify
    verify(a.parent, out/'train/initial.pt', out/'parent_restore_verified.json')

    import torch
    middle = a.parent_update + a.updates//2
    candidates = []
    for path in sorted((out/'train').glob('wall_*.pt')):
        state = torch.load(path, map_location='cpu', weights_only=False)
        update = int(state['iter'])+1
        del state
        if a.parent_update < update < final_update:
            candidates.append((abs(update-middle), update, path))
    checkpoints = [('final', out/'train/final.pt', final_update)]
    if candidates:
        _, mid_update, mid_path = min(candidates)
        checkpoints.append(('mid', mid_path, mid_update))
    write('evaluation_checkpoints.json', {k: dict(path=str(v), update=u) for k, v, u in checkpoints})

    from compare_impact_followup import main as compare
    from compare_impact_centers import main as centers
    for label, path, update in checkpoints:
        for name, *_ in STAGES:
            stage = label+'_'+name
            count = evaluate(path, stage, out)
            compare(out, control_root/('control_'+name), out/stage, a.parent_update, update,
                    stage+'_comparison.json', count)
            centers(control_root/('control_'+name), out/stage, out/(stage+'_paired_centers.json'))
    write('dashboard.json', dict(state='completed', stage='analysis_due', variant=a.variant,
                                 total_updates=final_update, updated=time.time()))


if __name__ == '__main__':
    main()
