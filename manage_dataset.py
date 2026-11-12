import argparse
import os
import shutil
import time
from collections import Counter
import pandas as pd

CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "gestures.csv"))
DATA_DIR = os.path.dirname(CSV_PATH)


def load_dataset():
    if not os.path.isfile(CSV_PATH):
        print(f"[ERROR] Dataset not found at {CSV_PATH}")
        return None
    return pd.read_csv(CSV_PATH)


def show_stats(df):
    if df is None or len(df) == 0:
        print("Dataset is empty.")
        return

    print("=" * 55)
    print(f"VISION2BRAILLE DATASET SUMMARY: {len(df)} total samples")
    print("=" * 55)

    counts = df['label'].value_counts()
    all_letters = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
    missing = [c for c in all_letters if c not in counts]
    weak = [lbl for lbl, c in counts.items() if c < 75]
    words = [lbl for lbl in counts.keys() if len(lbl) > 1]

    print("\n[Alphabet A-Z]")
    for i, ch in enumerate(all_letters):
        cnt = counts.get(ch, 0)
        status = "OK" if cnt >= 80 else ("LOW" if cnt > 0 else "MISSING")
        print(f"  {ch}: {cnt:3d} ({status})", end="  ")
        if (i + 1) % 6 == 0:
            print()
    print("\n")

    if words:
        print("[Custom Word Signs]")
        for w in words:
            print(f"  {w}: {counts[w]} samples")
        print()

    if missing:
        print(f"[!] MISSING LETTERS ({len(missing)}): {', '.join(missing)}")
    if weak:
        print(f"[!] LOW SAMPLE LABELS (<75 samples): {', '.join(weak)}")
    if not missing and not weak:
        print("[+] All letters have healthy sample counts (>= 75).")
    print("=" * 55)


def backup_dataset():
    if not os.path.isfile(CSV_PATH):
        return None
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(DATA_DIR, f"gestures_backup_{ts}.csv")
    shutil.copyfile(CSV_PATH, backup_path)
    print(f"[BACKUP] Created safety backup: {backup_path}")
    return backup_path


def delete_label(label_to_delete):
    df = load_dataset()
    if df is None:
        return

    target = label_to_delete.strip().upper()
    matching = df[df['label'] == target]
    if len(matching) == 0:
        print(f"Label '{target}' was not found in dataset. Nothing deleted.")
        return

    backup_dataset()
    filtered_df = df[df['label'] != target]
    filtered_df.to_csv(CSV_PATH, index=False)
    print(f"[SUCCESS] Removed all {len(matching)} samples for '{target}'.")
    print(f"Dataset now has {len(filtered_df)} total samples.")
    print(f"You can now re-record clean samples for '{target}' using:")
    if len(target) == 1:
        print(f"   python data_collection/collect_gestures.py")
    else:
        print(f"   python data_collection/collect_gestures.py --label {target}")
    print("And then re-train with:")
    print("   python model/train_classifier.py")


def main():
    parser = argparse.ArgumentParser(description="Vision2Braille Dataset Health & Management Tool")
    parser.add_argument("--stats", action="store_true", help="Display per-label sample counts and health diagnostics")
    parser.add_argument("--delete", metavar="LABEL", help="Safely wipe all samples for a specific letter/word (with auto-backup)")
    parser.add_argument("--backup", action="store_true", help="Create an immediate backup of gestures.csv")
    args = parser.parse_args()

    if args.backup:
        backup_dataset()
    elif args.delete:
        delete_label(args.delete)
    else:
        # Default action: show stats
        df = load_dataset()
        show_stats(df)


if __name__ == "__main__":
    main()
