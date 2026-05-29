from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate is_complaint precision/recall/F1 from support_agent log against dataset.json"
    )
    parser.add_argument(
        "--log",
        default="/Users/zheka/Documents/ML/agents/ccz_test/logs/support_agent_TICKET-LOCAL-001.log",
        help="Path to support_agent log file",
    )
    parser.add_argument(
        "--dataset",
        default="/Users/zheka/Documents/ML/agents/ccz_test/dataset/dataset.json",
        help="Path to dataset.json",
    )
    return parser.parse_args()


def load_expected(dataset_path: Path) -> dict[str, bool]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    expected: dict[str, bool] = {}
    for item in data:
        ticket_id = str(item.get("ticket_id", "")).strip()
        if not ticket_id:
            continue
        expected[ticket_id] = item.get("expected") == "complaint_escalation"
    return expected


def load_predicted_from_log(log_path: Path) -> dict[str, bool]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    # We only parse final snapshots to avoid duplicates from intermediate logs.
    # Example fragment:
    # Final trace: StateSnapshot(values={... 'ticket_id': 'TICKET-C001', ... 'is_complaint': True, ...})
    pattern = re.compile(
        r"Final trace:\s*StateSnapshot\(values=\{(?P<values>.*?)\}\s*,\s*next=",
        flags=re.DOTALL,
    )

    ticket_re = re.compile(r"'ticket_id':\s*'([^']+)'")
    complaint_re = re.compile(r"'is_complaint':\s*(True|False)")

    predicted: dict[str, bool] = {}
    for match in pattern.finditer(text):
        values_blob = match.group("values")
        ticket_m = ticket_re.search(values_blob)
        complaint_m = complaint_re.search(values_blob)
        if not ticket_m or not complaint_m:
            continue

        ticket_id = ticket_m.group(1)
        is_complaint = complaint_m.group(1) == "True"
        predicted[ticket_id] = is_complaint

    return predicted


def main() -> None:
    args = parse_args()

    log_path = Path(args.log)
    dataset_path = Path(args.dataset)

    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")
    if not dataset_path.exists():
        raise SystemExit(f"Dataset file not found: {dataset_path}")

    expected = load_expected(dataset_path)
    predicted = load_predicted_from_log(log_path)

    common_ticket_ids = sorted(set(expected) & set(predicted))
    if not common_ticket_ids:
        raise SystemExit(
            "No overlapping ticket_ids between log and dataset. "
            "Check log path or ticket_id format."
        )

    y_true: list[int] = []
    y_pred: list[int] = []
    for ticket_id in common_ticket_ids:
        y_true.append(1 if expected[ticket_id] else 0)
        y_pred.append(1 if predicted[ticket_id] else 0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    print("is_complaint metrics (positive class = True)")
    print(f"log_file={log_path}")
    print(f"dataset_file={dataset_path}")
    print(f"matched_samples={len(common_ticket_ids)}")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"precision={precision:.4f}")
    print(f"recall={recall:.4f}")
    print(f"f1={f1:.4f}")
    print(f"accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
