"""Upload DISCO + CIPIC to ooshyun/fine-grained-soundscape."""
from huggingface_hub import HfApi
import argparse
from pathlib import Path


def upload(raw_dir: Path, repo_id: str = "ooshyun/fine-grained-soundscape"):
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    # Upload DISCO
    disco_dir = raw_dir / "disco_noises"
    if disco_dir.exists():
        print("Uploading DISCO ...")
        api.upload_folder(
            folder_path=str(disco_dir),
            path_in_repo="disco",
            repo_id=repo_id,
            repo_type="dataset",
        )

    # Upload CIPIC
    cipic_dir = raw_dir / "cipic-hrtf-database"
    if cipic_dir.exists():
        print("Uploading CIPIC ...")
        api.upload_folder(
            folder_path=str(cipic_dir),
            path_in_repo="cipic_hrtf",
            repo_id=repo_id,
            repo_type="dataset",
        )

    # Upload README
    readme = Path(__file__).parent / "README.md"
    if readme.exists():
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--repo_id", default="ooshyun/fine-grained-soundscape")
    args = parser.parse_args()
    upload(args.raw_dir, args.repo_id)
