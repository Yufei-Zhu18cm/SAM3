import argparse
import os
from pathlib import Path

from eventbook.event_pipeline import load_yaml, predict_video


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="eventbook/config_h2o2101.yaml")
    parser.add_argument("--video_name", required=True)
    parser.add_argument(
        "--model_path",
        default="",
        help="Checkpoint path. If empty, prefer ./checkpoints/latest.joblib, otherwise fallback to config model.save_path",
    )
    parser.add_argument("--output_csv", required=True)
    return parser.parse_args()


def resolve_model_path(cfg: dict, cli_model_path: str) -> str:
    if cli_model_path:
        path = cli_model_path
    else:
        default_latest = "./checkpoints/latest.joblib"
        config_path = cfg.get("model", {}).get("save_path", "")
        if os.path.exists(default_latest):
            path = default_latest
        elif config_path and os.path.exists(config_path):
            path = config_path
        else:
            raise FileNotFoundError(
                "No model checkpoint found. Checked ./checkpoints/latest.joblib and config model.save_path"
            )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    return str(Path(path).resolve())


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    model_path = resolve_model_path(cfg, args.model_path)

    df = predict_video(
        cfg=cfg,
        video_name=args.video_name,
        model_path=model_path,
        output_csv=args.output_csv,
    )

    print(f"Prediction done. events={len(df)}")
    print(f"Model used: {model_path}")
    print(f"Saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
