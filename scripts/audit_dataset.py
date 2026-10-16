from __future__ import annotations

import argparse
from collections import Counter

from clip_anomaly.config import load_config
from clip_anomaly.experiment import collect_selected_records, write_split_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    records = collect_selected_records(config)
    output = write_split_csv(records, config)
    counts = Counter((record.category, record.split, record.label) for record in records)
    print(f"wrote {len(records)} records to {output}")
    for key, count in sorted(counts.items()):
        print(f"category={key[0]} split={key[1]} label={key[2]} count={count}")


if __name__ == "__main__":
    main()
