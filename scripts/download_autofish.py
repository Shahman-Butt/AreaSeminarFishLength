import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/raw/autofish")
    parser.add_argument("--repo-id", default="vapaau/autofish")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(out_dir),
    )
    print(path)


if __name__ == "__main__":
    main()
