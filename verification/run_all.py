"""一键运行两讲的最小工程验证。"""

import json

from verification.lecture02_validation import (
    run_cbow_toy_validation,
    run_negative_sampling_cost_validation,
    run_subword_oov_validation,
)
from verification.lecture05_validation import run_ner_context_validations


def run_all_validations():
    lecture02_cbow = run_cbow_toy_validation(seed=7, epochs=80)
    lecture02_cost = run_negative_sampling_cost_validation()
    lecture02_oov = run_subword_oov_validation()
    lecture05 = run_ner_context_validations()

    lecture02_passed = (
        lecture02_cbow["final_loss"] < lecture02_cbow["initial_loss"] * 0.9
        and "queen" in lecture02_cbow["nearest_neighbors"]["king"][:3]
        and "dog" in lecture02_cbow["nearest_neighbors"]["cat"][:3]
        and lecture02_cost["negative_sampling_score_count"] < lecture02_cost["full_softmax_score_count"] * 0.01
        and lecture02_oov["plain_lookup"] is None
        and lecture02_oov["subword_similarity_to_related"] > lecture02_oov["subword_similarity_to_unrelated"]
    )
    lecture05_passed = (
        lecture05["bidirectional_f1"] > lecture05["left_only_f1"]
        and lecture05["char_oov_f1"] > lecture05["no_char_oov_f1"]
        and lecture05["crf_illegal_before"] > lecture05["crf_illegal_after"]
        and lecture05["document_memory_f1"] > lecture05["no_memory_f1"]
    )

    return {
        "passed": lecture02_passed and lecture05_passed,
        "lecture02": {
            "pages_expected": "engineering",
            "cbow": lecture02_cbow,
            "negative_sampling_cost": lecture02_cost,
            "subword_oov": lecture02_oov,
        },
        "lecture05": lecture05,
    }


def main():
    result = run_all_validations()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
