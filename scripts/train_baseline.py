from __future__ import annotations

import argparse

from clip_anomaly.config import load_config
from clip_anomaly.experiment import feature_cache
from clip_anomaly.training import train_head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    payload = feature_cache(config).load()
    settings = config.training
    result = train_head(
        payload["features"],
        payload["masks"],
        [record["split"] for record in payload["records"]],
        payload["prototypes"],
        config.output_root / "experiments" / "checkpoints",
        int(config.model["hidden_dim"]),
        float(config.model["dropout"]),
        int(settings["epochs"]),
        int(settings["batch_size"]),
        float(settings["learning_rate"]),
        float(settings["weight_decay"]),
        int(settings["patience"]),
        float(settings["focal_gamma"]),
        config.seed,
    )
    print(f"best_epoch={result.best_epoch} best_val_loss={result.best_val_loss:.6f}")
    print(result.checkpoint_path)


if __name__ == "__main__":
    main()
