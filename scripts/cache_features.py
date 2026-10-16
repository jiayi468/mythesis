from __future__ import annotations

import argparse

from clip_anomaly.config import load_config
from clip_anomaly.experiment import build_feature_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    output = build_feature_cache(load_config(args.config), args.batch_size)
    print(output)


if __name__ == "__main__":
    main()
