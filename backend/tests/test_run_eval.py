from evaluation.run_eval import _confusions


def test_confusions_group_errors_by_expected_and_predicted_most_common_first():
    errors = [
        {"expected": "logistics", "predicted": "order_status"},
        {"expected": "refund", "predicted": "payment_issue"},
        {"expected": "refund", "predicted": "payment_issue"},
    ]
    assert _confusions(errors) == [
        {"expected": "refund", "predicted": "payment_issue", "count": 2},
        {"expected": "logistics", "predicted": "order_status", "count": 1},
    ]
    assert _confusions([]) == []
