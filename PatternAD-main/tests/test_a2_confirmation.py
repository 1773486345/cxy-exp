import unittest

from scripts.a2.run_m1b_confirmation import _seed_pairs


class A2ConfirmationTest(unittest.TestCase):
    def test_seed_pairs_require_unique_matching_lists(self):
        self.assertEqual(
            _seed_pairs(
                {
                    "schema_version": 1,
                    "contract_seeds": [5102, 5103],
                    "model_seeds": [6202, 6203],
                }
            ),
            [(5102, 6202), (5103, 6203)],
        )
        with self.assertRaisesRegex(ValueError, "equally sized"):
            _seed_pairs(
                {"schema_version": 1, "contract_seeds": [5102], "model_seeds": [6202, 6203]}
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            _seed_pairs(
                {"schema_version": 1, "contract_seeds": [5102, 5102], "model_seeds": [6202, 6203]}
            )


if __name__ == "__main__":
    unittest.main()
