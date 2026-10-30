"""Read an explicit, reviewed G1 motion set without changing the PPO recipe."""
import json
from pathlib import Path


def load_motion_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('format') != 'reviewed_g1_references_v1' or not data.get('motions'):
        raise ValueError('Expected a nonempty reviewed_g1_references_v1 manifest')
    paths = []
    for row in data['motions']:
        if row.get('identity_review') != 'approved' or row.get('pose_review') != 'approved':
            raise ValueError(f"Unreviewed reference: {row.get('motion_file')}")
        motion = (path.parent / row['motion_file']).resolve()
        evidence = (path.parent / row['review_evidence']).resolve()
        if motion.suffix != '.npz' or not motion.is_file() or not evidence.is_file():
            raise ValueError(f'Missing G1 motion or review evidence: {motion}')
        if motion in paths:
            raise ValueError(f'Duplicate reference: {motion}')
        paths.append(motion)
    return paths
