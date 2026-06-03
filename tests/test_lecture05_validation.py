import unittest

from verification.lecture05_validation import (
    count_illegal_bio_transitions,
    extract_bio_entities,
    repair_bio_tags,
    run_ner_context_validations,
)


class Lecture05ValidationTest(unittest.TestCase):
    def test_bio_repair_removes_illegal_i_tags_and_preserves_entity_spans(self):
        repaired = repair_bio_tags(["O", "I-ORG", "I-ORG", "B-PER", "I-LOC"])

        self.assertEqual(repaired, ["O", "B-ORG", "I-ORG", "B-PER", "B-LOC"])
        self.assertEqual(count_illegal_bio_transitions(repaired), 0)
        self.assertEqual(
            extract_bio_entities(repaired),
            [(1, 2, "ORG"), (3, 3, "PER"), (4, 4, "LOC")],
        )

    def test_context_validations_show_each_added_context_source_helping(self):
        result = run_ner_context_validations()

        self.assertGreater(result["bidirectional_f1"], result["left_only_f1"])
        self.assertGreater(result["char_oov_f1"], result["no_char_oov_f1"])
        self.assertGreater(result["crf_illegal_before"], result["crf_illegal_after"])
        self.assertGreater(result["document_memory_f1"], result["no_memory_f1"])


if __name__ == "__main__":
    unittest.main()
