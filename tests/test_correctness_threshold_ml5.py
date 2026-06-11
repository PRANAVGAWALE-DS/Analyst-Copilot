# tests/test_correctness_threshold_ml5.py
import sys

sys.path.insert(0, "analyst_copilot")
from eval import _columns_semantically_match

print("=== ML-5: Correctness threshold tests ===")

cases = [
    # (generated_cols, ground_truth_cols, expect_correct, expect_partial, description)
    (
        {"amount", "region", "quarter", "count"},
        {"amount", "region", "quarter", "count"},
        True,
        False,
        "exact 4/4 match → correct",
    ),
    (
        {"amount", "region", "quarter", "count"},
        {"total_amount", "region", "quarter", "record_count"},
        True,
        False,
        "4/4 after normalisation → correct",
    ),
    (
        {"amount", "region"},
        {"amount", "region", "quarter", "count"},
        False,
        True,
        "2/4 = 50% → partial only, NOT correct",
    ),
    (
        {"amount", "region", "quarter"},
        {"amount", "region", "quarter", "count"},
        True,
        False,
        "3/4 = 75% → correct (at threshold)",
    ),
    (
        {"amount"},
        {"amount", "region", "quarter", "count"},
        False,
        False,
        "1/4 = 25% → neither",
    ),
    # Scalar result
    ({"count"}, {"total_count"}, True, False, "scalar bare aggregate → correct"),
]

all_pass = True
for gen, gt, exp_correct, exp_partial, desc in cases:
    got_correct, got_partial = _columns_semantically_match(gen, gt)
    ok = (got_correct == exp_correct) and (got_partial == exp_partial)
    all_pass = all_pass and ok
    print(f"  {'PASS' if ok else 'FAIL'}: {desc}")
    if not ok:
        print(
            f"    Expected correct={exp_correct} partial={exp_partial}, "
            f"got correct={got_correct} partial={got_partial}"
        )

print(f"\n{'All tests passed.' if all_pass else 'FAILURES — ML-5 fix incomplete.'}")
