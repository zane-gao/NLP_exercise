import unittest

from verification.run_all import run_all_validations


class RunAllValidationTest(unittest.TestCase):
    def test_run_all_collects_both_lecture_validation_summaries(self):
        result = run_all_validations()

        self.assertTrue(result["passed"])
        self.assertIn("lecture02", result)
        self.assertIn("lecture05", result)
        self.assertEqual(result["lecture02"]["pages_expected"], "engineering")
        self.assertEqual(result["lecture05"]["pages_expected"], "engineering")


if __name__ == "__main__":
    unittest.main()
