from __future__ import annotations

import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = ROOT / "vendor"
COMMIT_MARKER = ".vendor_commit"

VENDORS = (
    {
        "name": "MatAnyone2",
        "path": ROOT / "vendor" / "MatAnyone2",
        "repo": "https://github.com/pq-yang/MatAnyone2",
        "commit": "4aed00afd91e108d8986b062b968678fd1c4f8f7",
        "archive_url": "https://github.com/pq-yang/MatAnyone2/archive/4aed00afd91e108d8986b062b968678fd1c4f8f7.tar.gz",
    },
    {
        "name": "segment-anything",
        "path": ROOT / "vendor" / "segment-anything",
        "repo": "https://github.com/facebookresearch/segment-anything",
        "commit": "dca509fe793f601edb92606367a655c15ac00fdf",
        "archive_url": "https://github.com/facebookresearch/segment-anything/archive/dca509fe793f601edb92606367a655c15ac00fdf.tar.gz",
    },
)


def _read_marker(repo_path: Path) -> str | None:
    marker = repo_path / COMMIT_MARKER
    if not marker.exists():
        return None
    return marker.read_text(encoding="utf-8").strip() or None


def _safe_extract_tar(archive_path: Path, destination: Path) -> Path:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        top_levels = {member.name.split("/", 1)[0] for member in members if member.name}
        if len(top_levels) != 1:
            raise RuntimeError(f"Unexpected archive layout in {archive_path.name}")

        for member in members:
            target_path = (destination / member.name).resolve()
            if destination.resolve() not in target_path.parents and target_path != destination.resolve():
                raise RuntimeError(f"Unsafe archive entry detected: {member.name}")

        try:
            archive.extractall(destination, filter="data")
        except TypeError:
            archive.extractall(destination)

    return destination / next(iter(top_levels))


def _download_file(url: str, destination: Path) -> None:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)


def ensure_vendor_checkout(vendor: dict[str, str | Path]) -> None:
    repo_path = Path(vendor["path"])
    commit = str(vendor["commit"])
    archive_url = str(vendor["archive_url"])

    if repo_path.exists() and _read_marker(repo_path) == commit:
        print(f"Using pinned {vendor['name']} snapshot at {repo_path}")
        return

    VENDOR_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="matanyone_vendor_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        archive_path = temp_dir / f"{vendor['name']}.tar.gz"
        _download_file(archive_url, archive_path)
        extracted_root = _safe_extract_tar(archive_path, temp_dir)

        if repo_path.exists():
            shutil.rmtree(repo_path)
        shutil.copytree(extracted_root, repo_path)
        (repo_path / COMMIT_MARKER).write_text(f"{commit}\n", encoding="utf-8")
        print(f"Installed pinned {vendor['name']} snapshot at {repo_path}")


if __name__ == "__main__":
    for vendor in VENDORS:
        ensure_vendor_checkout(vendor)
