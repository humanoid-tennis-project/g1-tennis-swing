import json
from pathlib import Path
import tempfile
import unittest
from reference_manifest import load_motion_manifest


class ManifestTests(unittest.TestCase):
    def test_review_and_identity_gate_and_duplicate(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            (root/'motion.npz').write_bytes(b'placeholder; numerical audit is a separate gate')
            (root/'review.json').write_text('{}')
            row = dict(motion_file='motion.npz', review_evidence='review.json',
                       identity_review='approved', pose_review='approved')
            manifest = root/'manifest.json'
            def write(rows):
                manifest.write_text(json.dumps(dict(format='reviewed_g1_references_v1', motions=rows)))
            write([row])
            self.assertEqual(load_motion_manifest(manifest), [(root/'motion.npz').resolve()])
            write([dict(row, pose_review='pending')])
            with self.assertRaises(ValueError): load_motion_manifest(manifest)
            write([row,row])
            with self.assertRaises(ValueError): load_motion_manifest(manifest)
            write([dict(row,review_evidence='missing.json')])
            with self.assertRaises(ValueError): load_motion_manifest(manifest)


if __name__ == '__main__':
    unittest.main()
