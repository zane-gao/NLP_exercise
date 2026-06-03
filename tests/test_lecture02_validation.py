import unittest

from verification.lecture02_validation import (
    run_cbow_toy_validation,
    run_negative_sampling_cost_validation,
    run_subword_oov_validation,
)


class Lecture02ValidationTest(unittest.TestCase):
    def test_cbow_toy_training_lowers_loss_and_clusters_related_words(self):
        result = run_cbow_toy_validation(seed=7, epochs=80)

        self.assertLess(result["final_loss"], result["initial_loss"] * 0.9)
        self.assertIn("queen", result["nearest_neighbors"]["king"][:3])
        self.assertIn("dog", result["nearest_neighbors"]["cat"][:3])

    def test_negative_sampling_uses_far_fewer_output_scores_than_full_softmax(self):
        result = run_negative_sampling_cost_validation(vocab_size=5000, positive_pairs=120, negatives_per_pair=5)

        self.assertEqual(result["full_softmax_score_count"], 600000)
        self.assertEqual(result["negative_sampling_score_count"], 720)
        self.assertLess(
            result["negative_sampling_score_count"],
            result["full_softmax_score_count"] * 0.01,
        )

    def test_subword_oov_gets_vector_when_plain_vocab_lookup_fails(self):
        result = run_subword_oov_validation()

        self.assertIsNone(result["plain_lookup"])
        self.assertIsNotNone(result["subword_vector"])
        self.assertGreater(result["subword_similarity_to_related"], 0.45)
        self.assertGreater(
            result["subword_similarity_to_related"],
            result["subword_similarity_to_unrelated"],
        )


if __name__ == "__main__":
    unittest.main()
