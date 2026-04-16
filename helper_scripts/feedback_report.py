#!/usr/bin/env python3
"""
feedback_report.py — Genius AI Feedback Report CLI

Query and display user feedback from the Chainlit SQLite database.
Shows: timestamp, user, question, answer, status (liked/not liked), comment.

Usage:
    python helper_scripts/feedback_report.py                        # All feedback
    python helper_scripts/feedback_report.py --user harnish@genius.ai   # Filter by user
    python helper_scripts/feedback_report.py --csv feedback_export.csv  # Export to CSV
    python helper_scripts/feedback_report.py --limit 50             # Limit results
    python helper_scripts/feedback_report.py --json                 # JSON output
"""

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

# Add parent directory to path so we can import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_layer import SQLiteDataLayer


def truncate(text: str, max_len: int = 80) -> str:
    """Truncate text for table display."""
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        return text[:max_len - 3] + "..."
    return text


def format_status(status: str) -> str:
    """Add emoji to feedback status."""
    if status == "liked":
        return "👍 Liked"
    elif status == "not liked":
        return "👎 Not Liked"
    return status


async def get_feedback(user: str | None, limit: int) -> list[dict]:
    """Fetch feedback records from the database."""
    dl = SQLiteDataLayer()
    try:
        return await dl.get_feedback_report(
            user_identifier=user,
            limit=limit,
        )
    finally:
        await dl.close()


def print_table(records: list[dict]):
    """Print feedback as a formatted console table."""
    if not records:
        print("\n📭 No feedback records found.\n")
        return

    print(f"\n📊 Feedback Report — {len(records)} record(s)\n")
    print("=" * 120)

    # Header
    print(
        f"{'Timestamp':<22} │ "
        f"{'User':<22} │ "
        f"{'Status':<14} │ "
        f"{'Question':<30} │ "
        f"{'Comment':<20}"
    )
    print("─" * 120)

    for r in records:
        timestamp = r["timestamp"][:19] if r["timestamp"] else "N/A"
        user = truncate(r["user"], 20)
        status = format_status(r["status"])
        question = truncate(r["question"], 28)
        comment = truncate(r["comment"] or "", 18)

        print(
            f"{timestamp:<22} │ "
            f"{user:<22} │ "
            f"{status:<14} │ "
            f"{question:<30} │ "
            f"{comment:<20}"
        )

    print("=" * 120)

    # Summary
    liked = sum(1 for r in records if r["status"] == "liked")
    not_liked = sum(1 for r in records if r["status"] == "not liked")
    commented = sum(1 for r in records if r.get("comment"))
    total = len(records)

    print(f"\n📈 Summary: {total} total | 👍 {liked} liked | 👎 {not_liked} not liked | 💬 {commented} with comments")

    if total > 0:
        satisfaction = (liked / total) * 100
        print(f"   Satisfaction rate: {satisfaction:.1f}%\n")


def export_csv(records: list[dict], filepath: str):
    """Export feedback records to a CSV file."""
    if not records:
        print("📭 No records to export.")
        return

    fieldnames = ["timestamp", "user", "question", "answer", "status", "value", "comment", "thread_id", "feedback_id"]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"✅ Exported {len(records)} feedback records to: {filepath}")


def export_json(records: list[dict]):
    """Print feedback as JSON to stdout."""
    print(json.dumps(records, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        description="Genius AI — Feedback Report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python helper_scripts/feedback_report.py
  python helper_scripts/feedback_report.py --user harnish@genius.ai
  python helper_scripts/feedback_report.py --csv feedback_export.csv
  python helper_scripts/feedback_report.py --json
  python helper_scripts/feedback_report.py --limit 20
        """,
    )
    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="Filter feedback by user identifier (e.g. harnish@genius.ai)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of feedback records to retrieve (default: 100)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        metavar="FILE",
        help="Export feedback to a CSV file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output feedback as JSON",
    )

    args = parser.parse_args()

    # Fetch feedback
    records = asyncio.run(get_feedback(user=args.user, limit=args.limit))

    # Output
    if args.csv:
        export_csv(records, args.csv)
    elif args.json:
        export_json(records)
    else:
        print_table(records)


if __name__ == "__main__":
    main()
