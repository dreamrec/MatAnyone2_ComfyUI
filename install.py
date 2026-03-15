from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = ROOT / "vendor"
COMMIT_MARKER = ".vendor_commit"
DEMO_VIDEO_NAME = "matanyone2-demo-input.mp4"
DEMO_VIDEO_SOURCE = ROOT / "assets" / DEMO_VIDEO_NAME
DEMO_WORKFLOW_SOURCE = ROOT / "workflows" / "matanyone2_demo.json"
DEMO_WORKFLOW_NAMES = (
    "matanyone2_demo.json",
    "matanyone2_full_demo.json",
    "matanyone2_vhs_demo.json",
)

VENDORS = (
    {
        "name": "MatAnyone2",
        "path": ROOT / "vendor" / "MatAnyone2",
        "commit": "4aed00afd91e108d8986b062b968678fd1c4f8f7",
        "required_paths": (
            "matanyone2/config/eval_matanyone_config.yaml",
            "matanyone2/model/matanyone2.py",
        ),
    },
    {
        "name": "segment-anything",
        "path": ROOT / "vendor" / "segment-anything",
        "commit": "dca509fe793f601edb92606367a655c15ac00fdf",
        "required_paths": (
            "segment_anything/build_sam.py",
            "segment_anything/predictor.py",
        ),
    },
)


def _read_marker(repo_path: Path) -> str | None:
    marker = repo_path / COMMIT_MARKER
    if not marker.exists():
        return None
    return marker.read_text(encoding="utf-8").strip() or None


def _detect_input_directory() -> Path | None:
    try:
        import folder_paths  # type: ignore

        return Path(folder_paths.get_input_directory()).resolve()
    except Exception:
        pass

    for parent in ROOT.parents:
        if parent.name == "custom_nodes":
            return (parent.parent / "input").resolve()
    return None


def _detect_workflow_directory() -> Path | None:
    try:
        import folder_paths  # type: ignore

        return (Path(folder_paths.get_user_directory()).resolve() / "default" / "workflows")
    except Exception:
        pass

    for parent in ROOT.parents:
        if parent.name == "custom_nodes":
            return (parent.parent / "user" / "default" / "workflows").resolve()
    return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_demo_input_video() -> None:
    if not DEMO_VIDEO_SOURCE.exists():
        print(f"Bundled demo clip missing at {DEMO_VIDEO_SOURCE}, skipping input copy.")
        return

    input_dir = _detect_input_directory()
    if input_dir is None:
        print("Could not locate ComfyUI input directory, skipping demo clip copy.")
        return

    input_dir.mkdir(parents=True, exist_ok=True)
    destination = input_dir / DEMO_VIDEO_NAME
    if destination.exists():
        if _file_sha256(destination) == _file_sha256(DEMO_VIDEO_SOURCE):
            print(f"Using bundled demo clip at {destination}")
            return
        print(f"Refreshing bundled demo clip at {destination}")
    else:
        print(f"Copying bundled demo clip to {destination}")

    shutil.copy2(DEMO_VIDEO_SOURCE, destination)


def ensure_demo_workflows() -> None:
    if not DEMO_WORKFLOW_SOURCE.exists():
        print(f"Bundled demo workflow missing at {DEMO_WORKFLOW_SOURCE}, skipping workflow sync.")
        return

    workflow_dir = _detect_workflow_directory()
    if workflow_dir is None:
        print("Could not locate ComfyUI workflow directory, skipping demo workflow sync.")
        return

    workflow_dir.mkdir(parents=True, exist_ok=True)
    source_hash = _file_sha256(DEMO_WORKFLOW_SOURCE)

    for workflow_name in DEMO_WORKFLOW_NAMES:
        destination = workflow_dir / workflow_name
        if destination.exists():
            if _file_sha256(destination) == source_hash:
                print(f"Using bundled demo workflow at {destination}")
                continue
            print(f"Refreshing bundled demo workflow at {destination}")
        else:
            print(f"Copying bundled demo workflow to {destination}")
        shutil.copy2(DEMO_WORKFLOW_SOURCE, destination)


def ensure_vendor_bundle(vendor: dict[str, str | Path | tuple[str, ...]]) -> None:
    repo_path = Path(vendor["path"])
    commit = str(vendor["commit"])
    required_paths = tuple(str(path) for path in vendor["required_paths"])

    missing_paths = [str(repo_path / relative_path) for relative_path in required_paths if not (repo_path / relative_path).exists()]
    if missing_paths:
        missing = ", ".join(missing_paths)
        raise RuntimeError(
            f"Bundled {vendor['name']} sources are incomplete. Missing: {missing}. "
            "Reinstall the node package from the registry or GitHub."
        )

    marker = _read_marker(repo_path)
    if marker != commit:
        raise RuntimeError(
            f"Bundled {vendor['name']} sources do not match the pinned commit. "
            f"Expected {commit}, found {marker or 'missing marker'}."
        )

    print(f"Using bundled {vendor['name']} sources at {repo_path}")


if __name__ == "__main__":
    for vendor in VENDORS:
        ensure_vendor_bundle(vendor)
    ensure_demo_input_video()
    ensure_demo_workflows()
