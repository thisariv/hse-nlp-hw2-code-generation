"""Independent tests of the B7 launch guards, with no network or GPU."""
import copy, json, tempfile, unittest
from pathlib import Path
from b7_review_api import launch_gate, require_single_push, reserve_push


class Tests(unittest.TestCase):
    def test_b7_single_push_both_targets_and_exclusive_intent(self):
        for smoke in (False, True):
            require_single_push(0, [], smoke)
            for version, known in ((1, []), (1, [1]), (2, [1]), (0, [1])):
                with self.assertRaises(AssertionError):
                    require_single_push(version, known, smoke)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'intent.json'; reserve_push(path, {'version': 0})
            with self.assertRaises(FileExistsError): reserve_push(path, {'version': 1})
            self.assertEqual(json.loads(path.read_text()), {'version': 0})

    def test_b7_production_requires_same_source_config_smoke(self):
        identity = dict(modules={'a': 'sha'}, C_overrides={'a': 'c-sha'}, config_sha256='prod')
        hand = dict(ready=True, CPU_ready=True, modules=identity['modules'], C_overrides=identity['C_overrides'],
            production_config_sha256='prod', smoke_config_sha256='smoke', smoke_passed=False, GPU_smoke_pending=True)
        receipt = dict(identity, config_sha256='smoke', passed=True, smoke=True)
        with self.assertRaises(AssertionError): launch_gate(hand, identity, False, receipt)
        hand.update(smoke_passed=True, GPU_smoke_pending=False)
        launch_gate(hand, identity, False, receipt)
        for key, value in (('config_sha256','old'), ('C_overrides',{}), ('modules',{}), ('passed',False), ('smoke',False)):
            stale = copy.deepcopy(receipt); stale[key] = value
            with self.assertRaises(AssertionError): launch_gate(hand, identity, False, stale)


if __name__ == '__main__': unittest.main(verbosity=2)
