from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CANONICAL_CSV_NAME = "need_predictions_geolocated_v2_final.csv"
CANONICAL_META_NAME = "need_predictions_geolocated_v2_final.meta.json"


def dashboard_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sibling_model_repo_root() -> Path:
    return dashboard_repo_root().parent / "afetYonetimi_colab"


def parse_args() -> argparse.Namespace:
    repo_root = dashboard_repo_root()
    default_source_repo = sibling_model_repo_root()
    default_dest_dir = repo_root / "data" / "predictions"

    parser = argparse.ArgumentParser(
        description="Copy the canonical final prediction CSV/meta pair into this dashboard repo."
    )
    parser.add_argument(
        "--source-repo",
        type=Path,
        default=default_source_repo,
        help="Path to the modeling repo that contains data/predictions/need_predictions_geolocated_v2_final.*",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=default_dest_dir,
        help="Destination directory inside the dashboard repo.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing destination files.",
    )
    return parser.parse_args()


def copy_file(src: Path, dst: Path, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination exists, rerun with --overwrite: {dst}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    source_repo = args.source_repo.expanduser().resolve()
    dest_dir = args.dest_dir.expanduser().resolve()

    src_csv = source_repo / "data" / "predictions" / CANONICAL_CSV_NAME
    src_meta = source_repo / "data" / "predictions" / CANONICAL_META_NAME
    dst_csv = dest_dir / CANONICAL_CSV_NAME
    dst_meta = dest_dir / CANONICAL_META_NAME

    copy_file(src_csv, dst_csv, args.overwrite)
    copy_file(src_meta, dst_meta, args.overwrite)

    print("Canonical prediction sync complete.")
    print(f"CSV : {dst_csv}")
    print(f"META: {dst_meta}")


if __name__ == "__main__":
    main()
