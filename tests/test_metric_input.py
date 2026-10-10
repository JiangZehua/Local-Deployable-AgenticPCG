from __future__ import annotations

import json
from types import SimpleNamespace
import unittest

from toolusepcg.config import config_from_dict, config_to_dict, load_config
from toolusepcg.metric_input import (
    DOMAINS, MetricInputError, apply_metric_request, parse_metric_request, request_prompt,
)
from toolusepcg.evaluator import EvalResult
from toolusepcg.optimizer.scoring import Scorer


def payload(domain, *targets, issues=None):
    return {"domain": domain, "targets": [
        {"metric": name, "mode": mode, "value": value} for name, mode, value in targets
    ], "issues": issues or []}


class MetricInputTests(unittest.TestCase):
    def config(self, domain):
        return load_config('configs/' + ('default' if domain == 'binary' else domain) + '.yaml')

    def test_every_domain_roundtrips_real_config_and_scores_target(self):
        cases = {
            'binary': ('path', 20), 'binarydoor': ('door_path', 20),
            'zelda': ('solution_length', 30), 'sokoban': ('crates', 2),
            'loderunner': ('ladder', 10), 'smb': ('coins', 3),
        }
        for domain, (metric, value) in cases.items():
            with self.subTest(domain=domain):
                original = self.config(domain)
                before = config_to_dict(original)
                config = apply_metric_request(original, payload(domain, (metric, 'exact', value)))
                self.assertEqual(before, config_to_dict(original))
                restored = config_from_dict(config_to_dict(config))
                self.assertEqual(getattr(restored.targets, metric).value, value)
                scorer = Scorer(restored)
                base = {'players': 1, 'keys': 1, 'doors': 1, 'regions': 1,
                        'num_connected_regions': 1, 'player_key': 15, 'key_door': 15,
                        'solution_length': 30, 'crates': 2, 'targets': 2,
                        'complete': 1, 'player': 1, 'gold': 3, 'collected_gold': 3}
                good = EvalResult(metrics={**base, metric: value}, problem_type=domain)
                bad = EvalResult(metrics={**base, metric: value + 1}, problem_type=domain)
                self.assertGreater(scorer.score(good), scorer.score(bad))

    def test_zelda_replaces_path_family_and_stale_threshold(self):
        config = apply_metric_request(self.config('zelda'), payload('zelda', ('solution_length', 'exact', 30)))
        self.assertIsNone(config.targets.player_key)
        config = apply_metric_request(config, payload('zelda', ('player_key', 'exact', 20), ('key_door', 'exact', 30)))
        self.assertIsNone(config.targets.solution_length)
        self.assertIsNone(config.targets.player_key.min_threshold)

    def test_zelda_conflicting_or_partial_paths_rejected(self):
        for entries in [(('player_key', 'exact', 20),),
                        (('player_key', 'exact', 20), ('key_door', 'exact', 30), ('solution_length', 'exact', 50))]:
            with self.assertRaises(MetricInputError):
                apply_metric_request(self.config('zelda'), payload('zelda', *entries))

    def test_sokoban_couples_counts_and_rejects_conflicts(self):
        config = apply_metric_request(self.config('sokoban'), payload('sokoban', ('targets', 'exact', 2)))
        self.assertEqual(config.targets.crates.value, 2)
        self.assertEqual(config.targets.targets.value, 2)
        with self.assertRaises(MetricInputError):
            apply_metric_request(config, payload('sokoban', ('crates', 'exact', 2), ('targets', 'exact', 3)))

    def test_fail_closed(self):
        bad = [
            payload('zelda', ('coins', 'exact', 2)),
            payload('smb', ('coin_tile_count', 'exact', 2)),
            payload('smb', ('coins', 'exact', True)),
            payload('smb', ('coins', 'exact', 2.5)),
            payload('smb', ('coins', 'exact', -1)),
            payload('smb', ('coins', 'exact', 6)),
            payload('smb', ('coins', 'min', 2)),
            payload('smb', ('coins', 'minimize', None)),
            payload('smb', ('coins', 'maximize', 2)),
            payload('smb', ('coins', 'exact', 2), ('coins', 'exact', 3)),
            payload('smb', ('coins', 'exact', 2), issues=['tile/event ambiguity']),
            payload('smb'), {}, {'domain': 'smb', 'targets': [], 'issues': 'bad'},
        ]
        for item in bad:
            with self.subTest(item=item), self.assertRaises(MetricInputError):
                apply_metric_request(self.config('smb'), item)

    def test_modes_are_capabilities_not_just_schema(self):
        with self.assertRaises(MetricInputError):
            apply_metric_request(self.config('zelda'), payload('zelda', ('enemies', 'maximize', None)))
        config = apply_metric_request(self.config('binary'), payload('binary', ('path', 'maximize', None)))
        self.assertEqual(config.targets.path.mode, 'maximize')
        self.assertIsNone(config.targets.path.value)

    def test_language_client_schema_and_incomplete_answer(self):
        calls = []
        choice = SimpleNamespace(finish_reason='stop', message=SimpleNamespace(
            content=json.dumps(payload('smb', ('coins', 'exact', 3)))))
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[choice])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        config, parsed = parse_metric_request('实际收集三枚金币', self.config('smb'), client)
        self.assertEqual(config.targets.coins.value, 3)
        self.assertTrue(calls[0]['response_format']['json_schema']['strict'])
        self.assertIn('NOT coin tiles', calls[0]['messages'][0]['content'])
        choice.finish_reason = 'length'
        with self.assertRaises(MetricInputError):
            parse_metric_request('coins', config, client)
        choice.finish_reason = 'stop'
        choice.message.content = 'not json'
        with self.assertRaises(MetricInputError):
            parse_metric_request('coins', config, client)

    def test_domain_vocabulary_isolated(self):
        self.assertEqual(len(DOMAINS), 6)
        self.assertIn('NOT enemy tiles', request_prompt('smb'))
        self.assertIn('not kills', request_prompt('zelda'))
        self.assertNotIn('coin_tile_count', request_prompt('zelda'))


if __name__ == '__main__':
    unittest.main()
