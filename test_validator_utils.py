import sys
import types
import unittest
from unittest import mock


if "bittensor" not in sys.modules:
    sys.modules["bittensor"] = types.SimpleNamespace(Subtensor=object, Metagraph=object)

if "bittensor_wallet" not in sys.modules:
    sys.modules["bittensor_wallet"] = types.SimpleNamespace(Wallet=object)

if "click" not in sys.modules:
    def _identity_decorator(*_args, **_kwargs):
        def _wrapped(func):
            return func
        return _wrapped

    sys.modules["click"] = types.SimpleNamespace(
        command=_identity_decorator,
        option=_identity_decorator,
        Choice=lambda *args, **kwargs: object(),
    )

import validator


class ValidatorUtilsTests(unittest.TestCase):
    class _FakeHotkey:
        ss58_address = "5FakeHotkeyAddress"

        @staticmethod
        def sign(message: bytes) -> bytes:
            return b"signed:" + message

    def test_extract_endpoint_from_string(self):
        value = validator._extract_endpoint_from_commitment("https://miner.example.com/")
        self.assertEqual(value, "https://miner.example.com")

    def test_extract_endpoint_from_json_string(self):
        payload = '{"endpoint":"http://127.0.0.1:8080"}'
        value = validator._extract_endpoint_from_commitment(payload)
        self.assertEqual(value, "http://127.0.0.1:8080")

    def test_extract_endpoint_from_dict(self):
        value = validator._extract_endpoint_from_commitment(
            {"url": "https://a.example.org/"}
        )
        self.assertEqual(value, "https://a.example.org")

    def test_merge_endpoint_sources_prefers_overrides(self):
        merged = validator.merge_endpoint_sources(
            chain_endpoints={2: "https://chain-a.example", 3: "https://chain-b.example"},
            override_endpoints={3: "https://override-b.example", 4: "https://override-c.example"},
        )
        self.assertEqual(
            merged,
            {
                2: "https://chain-a.example",
                3: "https://override-b.example",
                4: "https://override-c.example",
            },
        )

    def test_normalize_weights(self):
        uids, weights = validator.normalize_weights({7: 1.0, 2: 3.0})
        self.assertEqual(uids, [2, 7])
        self.assertAlmostEqual(sum(weights), 1.0, places=8)
        self.assertGreater(weights[0], weights[1])

    def test_choose_challenge_is_deterministic(self):
        c1 = validator.choose_challenge(block_number=12345, netuid=99)
        c2 = validator.choose_challenge(block_number=12345, netuid=99)
        self.assertEqual(c1.challenge_id, c2.challenge_id)

    def test_choose_challenges_batch_is_deterministic(self):
        b1 = validator.choose_challenges(block_number=3333, netuid=7, batch_size=4)
        b2 = validator.choose_challenges(block_number=3333, netuid=7, batch_size=4)
        self.assertEqual([c.challenge_id for c in b1], [c.challenge_id for c in b2])

    def test_submission_constraints(self):
        ok, _ = validator.validate_submission_constraints("by simp")
        self.assertTrue(ok)

        too_many_imports = "\n".join(["import Mathlib"] * (validator.MAX_IMPORT_LINES + 1))
        ok, reason = validator.validate_submission_constraints(too_many_imports)
        self.assertFalse(ok)
        self.assertIn("too many imports", reason)

        bad_token = "by\n  exact (by\n    unsafe"
        ok, reason = validator.validate_submission_constraints(bad_token)
        self.assertFalse(ok)
        self.assertIn("forbidden token", reason)

    def test_aggregate_weighted_instant_score(self):
        instant, valid_rate, elegance = validator.aggregate_weighted_instant_score(
            valid_flags=[1.0, 0.0],
            elegance_scores=[0.9, 0.2],
            difficulties=[2.0, 1.0],
            difficulty_weighting_enabled=True,
        )
        self.assertAlmostEqual(valid_rate, 2.0 / 3.0, places=6)
        self.assertAlmostEqual(elegance, (2.0 * 0.9 + 1.0 * 0.2) / 3.0, places=6)
        self.assertAlmostEqual(instant, valid_rate * elegance, places=6)

    def test_create_epistula_headers(self):
        body = b'{"hello":"world"}'
        headers = validator.create_epistula_headers(self._FakeHotkey(), body)
        self.assertEqual(headers["X-Epistula-Hotkey"], "5FakeHotkeyAddress")
        self.assertTrue(headers["X-Epistula-Timestamp"].isdigit())
        self.assertTrue(len(headers["X-Epistula-Signature"]) > 0)

    def test_verify_epistula_response_success(self):
        body = b'{"proof_code":"by simp"}'
        ts = str(1_700_000_000_000_000_000)
        with mock.patch.object(validator, "_verify_signature_with_hotkey", return_value=True):
            with mock.patch.object(validator.time, "time_ns", return_value=int(ts)):
                ok, reason = validator.verify_epistula_response(
                    response_body=body,
                    response_headers={
                        "X-Epistula-Timestamp": ts,
                        "X-Epistula-Signature": "abcd",
                        "X-Epistula-Hotkey": "5MinerHotkey",
                    },
                    expected_hotkey="5MinerHotkey",
                )
        self.assertTrue(ok)
        self.assertEqual(reason, "verified")

    def test_verify_epistula_response_hotkey_mismatch(self):
        ok, reason = validator.verify_epistula_response(
            response_body=b"{}",
            response_headers={
                "X-Epistula-Timestamp": "1",
                "X-Epistula-Signature": "abcd",
                "X-Epistula-Hotkey": "5Wrong",
            },
            expected_hotkey="5Expected",
        )
        self.assertFalse(ok)
        self.assertIn("does not match", reason)


if __name__ == "__main__":
    unittest.main()
