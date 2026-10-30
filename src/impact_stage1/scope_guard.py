"""Explicit reward-only checkpoint forks; data/MDP/runtime never bypassed."""
import json


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True))


def check_reward_fork(source, base, destination, allow_fork=False, approved=None):
    """Return exact/fork, rejecting every non-reward scope difference.

    `approved` maps explicitly frozen reward implementation SHAs to allowed
    variant names. It must come from the reviewed experiment, not checkpoint
    supplied data. No scope in the parent checkpoint is rewritten.
    """
    source, base, destination = map(normalized, (source, base, destination))
    if source == destination:
        return 'exact'
    if not allow_fork:
        raise ValueError('Scope differs: explicit reward-only fork required')
    if source == base:
        return 'fork'
    if not isinstance(source, dict):
        raise ValueError('Missing source scope')
    reward_keys = {'reward_variant', 'reward_variant_sha256'}
    if set(source) != set(base) | reward_keys:
        raise ValueError('Unknown parent scope fields')
    if {k: v for k, v in source.items() if k not in reward_keys} != base:
        raise ValueError('Parent data/task/network/goal/runtime changed')
    allowed = (approved or {}).get(source['reward_variant_sha256'], ())
    if source['reward_variant'] not in allowed:
        raise ValueError('Parent reward implementation is not approved')
    return 'fork'


def check_local_runtime_bridge(source,base,destination,parent_sha256,proof,allow_fork=False,approved=None):
    """Admit only an explicitly tested cloud->local minor-runtime transition."""
    if not allow_fork or proof.get('format')!='impact_local_runtime_bridge_v1':
        raise ValueError('Explicit validated local-runtime bridge required')
    if proof.get('checkpoint_sha256')!=parent_sha256:
        raise ValueError('Runtime comparison did not use this checkpoint')
    pair=(proof.get('source_runtime'),proof.get('destination_runtime'))
    if pair!=('3.12.1','3.12.0') or (source.get('mujoco'),base.get('mujoco'))!=pair:
        raise ValueError('Unreviewed runtime transition')
    if not proof.get('all_20_conditions_passed'):
        raise ValueError('Physical runtime comparison failed')
    adjusted=dict(base,mujoco=source['mujoco'])
    expected=dict(destination,mujoco=source['mujoco'])
    check_reward_fork(source,adjusted,expected,True,approved)
    return 'runtime_fork'


def check_cross_runtime_continuation(source,base,destination,parent_sha256,proof,allow_fork=False,approved=None):
    """Separate transfer experiment; never label it strict runtime equivalence."""
    if (not allow_fork or proof.get('format')!='impact_cross_runtime_continuation_v2'
            or proof.get('strict_equivalence_claimed') is not False
            or proof.get('final_validation_on_source_runtime_required') is not True):
        raise ValueError('Explicit cross-runtime continuation evidence required')
    if proof.get('checkpoint_sha256')!=parent_sha256 or not proof.get('all_20_conditions_passed'):
        raise ValueError('Same-checkpoint continuation comparison failed')
    if (source.get('mujoco'),base.get('mujoco'))!=('3.12.1','3.12.0'):
        raise ValueError('Unreviewed runtime continuation')
    if (proof.get('source_runtime'),proof.get('destination_runtime'))!=('3.12.1','3.12.0'):
        raise ValueError('Evidence refers to a different runtime pair')
    adjusted=dict(base,mujoco=source['mujoco'])
    check_reward_fork(source,adjusted,dict(destination,mujoco=source['mujoco']),True,approved)
    return 'runtime_continuation'


def check_foot_lock_data_fork(source,base,destination,manifest,allow_fork=False,approved=None):
    """Admit only foot-locked copies of exactly the parent's references.

    Each row must name the original motion SHA it was derived from; with those
    restored, the parent must pass the ordinary reward-only fork check.
    """
    lock=manifest.get('foot_lock')
    if not allow_fork or not lock or lock.get('joint_pos_modified') is not False:
        raise ValueError('Explicit foot-locked manifest required')
    original=[(r['id'],r['original_motion_sha256'],r['annotation_sha256']) for r in manifest['motions']]
    if normalized(original)==normalized(base['motions']):
        raise ValueError('Manifest does not change the references')
    restore=lambda s:dict(s,motions=original)
    check_reward_fork(source,normalized(restore(base)),normalized(restore(destination)),True,approved)
    return 'data_fork'
