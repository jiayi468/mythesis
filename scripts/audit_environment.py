from __future__ import annotations

import argparse

from clip_anomaly.config import load_config
from clip_anomaly.experiment import write_environment_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    output = write_environment_report(load_config(args.config))
    print(output)


if __name__ == "__main__":
    main()
