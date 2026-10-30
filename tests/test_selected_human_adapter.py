import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from adapt_selected_human import adapt, BODY_NAMES


class AdapterTests(unittest.TestCase):
    def test_preserves_source_samples_and_clock_and_rejects_changed_source(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory);n=4;frames=np.arange(120,124)
            xyz=np.arange(n*22*3,dtype=np.float32).reshape(n,22,3)/100
            pose=np.zeros((n,63),np.float32)
            source=root/'input.npz'
            np.savez(source,joints_global=xyz,joints_incam=xyz+1,fps=np.array(30.),
                parent_30Hz_frames=frames,parent_elapsed_seconds=frames/30,K_fullimg=np.zeros((n,3,3)),
                global__body_pose=pose,global__betas=np.zeros((n,10)))
            evidence=root/'review.json';evidence.write_text(json.dumps(dict(status='ok')))
            row=dict(id='test',human_motion_file=str(source),human_motion_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                review_evidence=str(evidence),review_evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())
            record=adapt(row,root/'adapted')
            self.assertEqual(record['source_interval_frames'],[120,124])
            with np.load(root/'adapted/human_joints_local.npz') as d:
                np.testing.assert_array_equal(d['joints_global'],xyz)
                np.testing.assert_allclose(d['time_seconds'],[0,1/30,2/30,3/30],rtol=0,atol=1e-15)
                self.assertEqual(d['joint_names'].tolist(),list(BODY_NAMES))
            with np.load(root/'adapted/human_motion_parameters.npz') as d:
                np.testing.assert_array_equal(d['global__body_pose'],pose)
            evidence.write_text('changed decision')
            with self.assertRaises(ValueError):adapt(row,root/'bad')


if __name__=='__main__':unittest.main()
