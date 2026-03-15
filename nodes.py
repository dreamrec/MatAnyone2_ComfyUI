from __future__ import annotations

import base64
import io
import json
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.hub import download_url_to_file


ROOT = Path(__file__).resolve().parent
MATANYONE_VENDOR_ROOT = ROOT / "vendor" / "MatAnyone2"
SAM_VENDOR_ROOT = ROOT / "vendor" / "segment-anything"

MODEL_VARIANTS = {
    "MatAnyone 2": {
        "filename": "matanyone2.pth",
        "url": "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth",
    },
    "MatAnyone": {
        "filename": "matanyone.pth",
        "url": "https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth",
    },
}

SAM_VARIANTS = {
    "vit_h": {
        "filename": "sam_vit_h_4b8939.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    },
    "vit_l": {
        "filename": "sam_vit_l_0b3195.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    },
    "vit_b": {
        "filename": "sam_vit_b_01ec64.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    },
}

MASK_PREVIEW_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (87, 199, 133),
    (54, 128, 255),
    (255, 184, 77),
    (244, 91, 105),
)
POSITIVE_POINT_COLOR = (61, 220, 132)
NEGATIVE_POINT_COLOR = (255, 84, 112)
POINT_RADIUS = 7
MATANYONE_CATEGORY = "MatAnyone2"
MATANYONE_SAM_CATEGORY = "MatAnyone2/SAM"

_MATANYONE_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
_SAM_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
_MATANYONE_CACHE_LOCK = threading.Lock()
_SAM_CACHE_LOCK = threading.Lock()
_INTERACTIVE_SAM_SESSIONS: Dict[str, Dict[str, Any]] = {}
_INTERACTIVE_SAM_SESSIONS_LOCK = threading.Lock()
_INTERACTIVE_SAM_SESSION_TTL_SECONDS = 15 * 60


def _append_vendor_paths() -> None:
    for vendor_root in (MATANYONE_VENDOR_ROOT, SAM_VENDOR_ROOT):
        vendor_path = str(vendor_root)
        if vendor_root.exists() and vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)


def _require_matanyone() -> None:
    _append_vendor_paths()
    try:
        import cv2  # noqa: F401
        import hydra  # noqa: F401
        import imageio  # noqa: F401
        import matanyone2  # noqa: F401
        import omegaconf  # noqa: F401
        import PIL  # noqa: F401
        import tqdm  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "MatAnyone dependencies are not installed yet. Run `python install.py` "
            "from this custom node folder, then restart ComfyUI."
        ) from exc


def _require_segment_anything() -> None:
    _append_vendor_paths()
    try:
        import segment_anything  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Segment Anything is not installed yet. Run `python install.py` "
            "from this custom node folder, then restart ComfyUI."
        ) from exc


def _default_checkpoint_dir(subdir: str) -> Path:
    try:
        import folder_paths  # type: ignore

        return Path(folder_paths.models_dir) / subdir
    except Exception:
        return ROOT / "models" / subdir


def _ensure_download(checkpoint_path: str, url: str, subdir: str, filename: str) -> Path:
    if checkpoint_path.strip():
        resolved = Path(checkpoint_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved}")
        return resolved

    model_dir = _default_checkpoint_dir(subdir)
    model_dir.mkdir(parents=True, exist_ok=True)
    resolved = (model_dir / filename).resolve()
    if not resolved.exists():
        download_url_to_file(url, str(resolved), progress=True)
    return resolved


def _ensure_model_checkpoint(model_name: str, checkpoint_path: str) -> Path:
    variant = MODEL_VARIANTS[model_name]
    return _ensure_download(
        checkpoint_path=checkpoint_path,
        url=variant["url"],
        subdir="matanyone",
        filename=variant["filename"],
    )


def _ensure_sam_checkpoint(sam_model_type: str, checkpoint_path: str) -> Path:
    variant = SAM_VARIANTS[sam_model_type]
    return _ensure_download(
        checkpoint_path=checkpoint_path,
        url=variant["url"],
        subdir="sams",
        filename=variant["filename"],
    )


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    if device_name == "mps":
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS was requested, but no MPS device is available.")

    try:
        import comfy.model_management as model_management  # type: ignore

        device = model_management.get_torch_device()
        return device if isinstance(device, torch.device) else torch.device(device)
    except Exception:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


def _load_matanyone_model(checkpoint_path: Path, device: torch.device):
    _require_matanyone()

    from hydra import compose, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import open_dict

    from matanyone2.model.matanyone2 import MatAnyone2

    global_hydra = GlobalHydra.instance()
    if global_hydra.is_initialized():
        global_hydra.clear()

    try:
        initialize_config_module(
            version_base="1.3.2",
            config_module="matanyone2.config",
            job_name="matanyone_comfyui",
        )
        cfg = compose(config_name="eval_matanyone_config")
        with open_dict(cfg):
            cfg["weights"] = str(checkpoint_path)
            cfg["model"]["pretrained_resnet"] = False

        model = MatAnyone2(cfg, single_object=True).to(device).eval()
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
        model.load_weights(state_dict)
        return model
    finally:
        if global_hydra.is_initialized():
            global_hydra.clear()


def _get_cached_matanyone_model(model_name: str, checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    cache_key = (model_name, str(checkpoint_path), str(device))
    with _MATANYONE_CACHE_LOCK:
        cached = _MATANYONE_CACHE.get(cache_key)
        if cached is None:
            cached = {
                "model": _load_matanyone_model(checkpoint_path, device),
                "device": device,
                "checkpoint_path": str(checkpoint_path),
                "model_name": model_name,
            }
            _MATANYONE_CACHE[cache_key] = cached
    return cached


def _load_sam_model(sam_model_type: str, checkpoint_path: Path, device: torch.device):
    _require_segment_anything()

    from segment_anything import sam_model_registry

    model = sam_model_registry[sam_model_type](checkpoint=str(checkpoint_path))
    model.to(device=device)
    model.eval()
    return model


def _get_cached_sam_model(sam_model_type: str, checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    cache_key = (sam_model_type, str(checkpoint_path), str(device))
    with _SAM_CACHE_LOCK:
        cached = _SAM_CACHE.get(cache_key)
        if cached is None:
            cached = {
                "model": _load_sam_model(sam_model_type, checkpoint_path, device),
                "device": device,
                "checkpoint_path": str(checkpoint_path),
                "sam_model_type": sam_model_type,
            }
            _SAM_CACHE[cache_key] = cached
    return cached


def _prepare_image_batch(images: torch.Tensor) -> torch.Tensor:
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Expected IMAGE input shaped as [frames, height, width, 3].")
    return images.detach().float().cpu().clamp(0.0, 1.0).contiguous()


def _prepare_single_image(image: torch.Tensor) -> torch.Tensor:
    return _prepare_image_batch(image)[0:1]


def _resize_mask_to(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError("Expected MASK input shaped as [frames, height, width] or [height, width].")
    chosen = mask[0].detach().float().cpu().clamp(0.0, 1.0)
    if chosen.shape != (height, width):
        chosen = F.interpolate(
            chosen.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="nearest",
        )[0, 0]
    return chosen


def _prepare_binary_mask(
    mask: torch.Tensor,
    height: int,
    width: int,
    threshold: float = 0.5,
    invert_mask: bool = False,
    erode_kernel: int = 0,
    dilate_kernel: int = 0,
) -> torch.Tensor:
    chosen = _resize_mask_to(mask, height, width)
    if invert_mask:
        chosen = 1.0 - chosen
    mask_np = ((chosen >= threshold).float().numpy() * 255.0).astype(np.float32)

    if dilate_kernel > 0 or erode_kernel > 0:
        import cv2

        if dilate_kernel > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(dilate_kernel), int(dilate_kernel)))
            mask_np = cv2.dilate((mask_np != 0).astype(np.float32), kernel, iterations=1) * 255.0

        if erode_kernel > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(erode_kernel), int(erode_kernel)))
            mask_np = cv2.erode((mask_np == 255).astype(np.float32), kernel, iterations=1) * 255.0

    return torch.from_numpy(mask_np.copy()).to(torch.float32)


def _check_interrupt() -> None:
    try:
        import comfy.model_management as model_management  # type: ignore

        model_management.throw_exception_if_processing_interrupted()
    except Exception:
        return


def _soft_empty_cache() -> None:
    try:
        import comfy.model_management as model_management  # type: ignore

        model_management.soft_empty_cache()
    except Exception:
        return


def _get_progress_bar(total: int):
    try:
        import comfy.utils  # type: ignore

        return comfy.utils.ProgressBar(total)
    except Exception:
        return None


def _update_progress(progress_bar, current: int, total: int) -> None:
    if progress_bar is None:
        return
    if hasattr(progress_bar, "update_absolute"):
        progress_bar.update_absolute(current, total)
    elif hasattr(progress_bar, "update"):
        progress_bar.update(1)


def _empty_prompt() -> Dict[str, List[Any]]:
    return {"points": [], "labels": []}


def _clone_prompt(prompt: Optional[Dict[str, Any]]) -> Dict[str, List[Any]]:
    if prompt is None:
        return _empty_prompt()
    return {
        "points": [list(point) for point in prompt.get("points", [])],
        "labels": list(prompt.get("labels", [])),
    }


def _parse_label_token(token: str) -> int:
    value = token.strip().lower()
    if value in {"+", "1", "positive", "pos", "foreground", "fg", "true"}:
        return 1
    if value in {"-", "0", "negative", "neg", "background", "bg", "false"}:
        return 0
    raise ValueError(f"Unsupported label token: {token}")


def _parse_prompt_text(prompt_text: str) -> Dict[str, List[Any]]:
    prompt = _empty_prompt()
    for line in prompt_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = [part for part in re.split(r"[\s,]+", stripped) if part]
        if len(parts) < 3:
            raise ValueError(
                "Each prompt line must look like `x,y,+` or `x y negative`."
            )
        x = int(float(parts[0]))
        y = int(float(parts[1]))
        label = _parse_label_token(parts[2])
        prompt["points"].append([x, y])
        prompt["labels"].append(label)
    return prompt


def _prompt_to_numpy(prompt: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    if not prompt.get("points"):
        raise ValueError("The prompt is empty. Add at least one positive or negative point.")
    points = np.asarray(prompt["points"], dtype=np.float32)
    labels = np.asarray(prompt["labels"], dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Prompt points must have shape [N, 2].")
    if labels.ndim != 1 or labels.shape[0] != points.shape[0]:
        raise ValueError("Prompt labels must match the number of points.")
    return points, labels


def _make_preview(
    image: torch.Tensor,
    masks: Sequence[torch.Tensor],
    opacity: float,
    prompt: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    image_batch = _prepare_single_image(image)
    base = (image_batch[0].numpy() * 255.0).round().astype(np.uint8)

    for index, mask in enumerate(masks):
        color = np.asarray(MASK_PREVIEW_COLORS[index % len(MASK_PREVIEW_COLORS)], dtype=np.float32)
        mask_np = mask.detach().float().cpu().numpy()
        if mask_np.shape != base.shape[:2]:
            mask_np = F.interpolate(
                torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0),
                size=base.shape[:2],
                mode="nearest",
            )[0, 0].numpy()
        alpha = np.clip(mask_np, 0.0, 1.0) * opacity
        base = np.clip(base * (1.0 - alpha[..., None]) + color * alpha[..., None], 0, 255).astype(np.uint8)

    if prompt and prompt.get("points"):
        import cv2

        for point, label in zip(prompt["points"], prompt["labels"]):
            color = POSITIVE_POINT_COLOR if int(label) == 1 else NEGATIVE_POINT_COLOR
            cv2.circle(base, (int(point[0]), int(point[1])), POINT_RADIUS, color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(base, (int(point[0]), int(point[1])), POINT_RADIUS + 2, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    return torch.from_numpy(base.astype(np.float32) / 255.0).unsqueeze(0)


def _normalize_mask_index(mask_choice: str, scores: np.ndarray) -> int:
    if mask_choice == "best":
        return int(np.argmax(scores))
    return min(max(int(mask_choice), 0), len(scores) - 1)


def _merge_mask_list(masks: Iterable[torch.Tensor], threshold: float) -> torch.Tensor:
    valid_masks = [mask for mask in masks if mask is not None]
    if not valid_masks:
        raise ValueError("At least one mask is required.")

    first_mask = valid_masks[0]
    if first_mask.ndim == 2:
        first_mask = first_mask.unsqueeze(0)
    height, width = first_mask.shape[-2:]

    merged = torch.zeros((height, width), dtype=torch.float32)
    for mask in valid_masks:
        resized = _resize_mask_to(mask, height, width)
        merged = torch.maximum(merged, resized)

    return (merged >= threshold).float().unsqueeze(0)


def _image_tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    return (_prepare_single_image(image)[0].numpy() * 255.0).round().astype(np.uint8)


def _numpy_image_to_tensor(image_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image_np.astype(np.float32) / 255.0).unsqueeze(0)


def _save_temp_preview(image_np: np.ndarray, prefix: str = "matanyone_preview") -> str:
    """Save image as temp file and return a /view URL instead of a data URL.

    Falls back to base64 data URL if ComfyUI folder_paths is unavailable.
    """
    from PIL import Image

    try:
        import folder_paths  # type: ignore

        temp_dir = folder_paths.get_temp_directory()
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        filename = f"{prefix}_{uuid.uuid4().hex[:12]}.png"
        filepath = Path(temp_dir) / filename
        Image.fromarray(image_np).save(str(filepath), format="PNG")
        return f"/view?filename={filename}&type=temp"
    except Exception:
        buffer = io.BytesIO()
        Image.fromarray(image_np).save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"


def _encode_png_data_url(image_np: np.ndarray) -> str:
    """Legacy alias — now routes through temp file saving."""
    return _save_temp_preview(image_np)


def _decode_image_data_url(image_data: str) -> np.ndarray:
    from PIL import Image

    if not image_data:
        raise ValueError("Missing image data.")

    # Support /view URL references (avoids base64 in request body)
    if image_data.startswith("/view?") or image_data.startswith("view?"):
        try:
            import folder_paths  # type: ignore
            from urllib.parse import parse_qs, urlparse

            parsed = urlparse(image_data if "?" in image_data else f"/{image_data}")
            params = parse_qs(parsed.query)
            filename = params.get("filename", [""])[0]
            file_type = params.get("type", ["temp"])[0]
            subfolder = params.get("subfolder", [""])[0]

            if file_type == "temp":
                base_dir = folder_paths.get_temp_directory()
            elif file_type == "input":
                base_dir = folder_paths.get_input_directory()
            elif file_type == "output":
                base_dir = folder_paths.get_output_directory()
            else:
                base_dir = folder_paths.get_temp_directory()

            filepath = Path(base_dir) / subfolder / filename if subfolder else Path(base_dir) / filename
            with Image.open(str(filepath)) as img:
                return np.asarray(img.convert("RGB"))
        except Exception:
            pass  # Fall through to base64 decoding

    encoded = image_data.split(",", 1)[1] if image_data.startswith("data:") else image_data
    raw = base64.b64decode(encoded)
    with Image.open(io.BytesIO(raw)) as image:
        return np.asarray(image.convert("RGB"))


def _empty_editor_state() -> Dict[str, Any]:
    return {"targets": [], "active_index": 0}


def _normalize_editor_target(target: Optional[Dict[str, Any]], index: int) -> Dict[str, Any]:
    prompt = _clone_prompt(target)
    mask_choice = str((target or {}).get("mask_choice", "best"))
    if mask_choice not in {"best", "0", "1", "2"}:
        mask_choice = "best"
    name = str((target or {}).get("name") or f"Mask {index + 1}")
    return {
        "name": name,
        "mask_choice": mask_choice,
        "points": prompt["points"],
        "labels": prompt["labels"],
    }


def _normalize_editor_state(editor_state: Optional[Any]) -> Dict[str, Any]:
    if isinstance(editor_state, str):
        stripped = editor_state.strip()
        if not stripped:
            return _empty_editor_state()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError("The saved interactive editor state is not valid JSON.") from exc
    elif isinstance(editor_state, dict):
        parsed = editor_state
    elif editor_state is None:
        return _empty_editor_state()
    else:
        raise ValueError("Unsupported interactive editor state.")

    raw_targets = parsed.get("targets", [])
    if not isinstance(raw_targets, list):
        raise ValueError("Interactive editor state must contain a `targets` list.")

    targets = [_normalize_editor_target(target if isinstance(target, dict) else None, index) for index, target in enumerate(raw_targets)]
    active_index = int(parsed.get("active_index", 0) or 0)
    if targets:
        active_index = min(max(active_index, 0), len(targets) - 1)
    else:
        active_index = 0
    return {"targets": targets, "active_index": active_index}


def _editor_state_to_json(editor_state: Dict[str, Any]) -> str:
    return json.dumps(_normalize_editor_state(editor_state), indent=2, sort_keys=True)


def _editor_state_prompt_count(editor_state: Dict[str, Any]) -> int:
    return sum(len(target.get("points", [])) for target in editor_state.get("targets", []))


def _editor_state_target_count(editor_state: Dict[str, Any], include_empty: bool = False) -> int:
    if include_empty:
        return len(editor_state.get("targets", []))
    return sum(1 for target in editor_state.get("targets", []) if target.get("points"))


def _predict_masks_for_editor_targets(
    predictor,
    image_batch: torch.Tensor,
    editor_state: Dict[str, Any],
    preview_opacity: float,
) -> Dict[str, Any]:
    targets = editor_state.get("targets", [])
    active_index = min(max(int(editor_state.get("active_index", 0) or 0), 0), max(len(targets) - 1, 0))
    selected_masks: List[torch.Tensor] = []
    active_candidates: List[Dict[str, Any]] = []

    for target_index, target in enumerate(targets):
        if not target.get("points"):
            continue

        points, labels = _prompt_to_numpy(target)
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
            return_logits=False,
        )

        selected_index = _normalize_mask_index(str(target.get("mask_choice", "best")), scores)
        selected_mask = torch.from_numpy(masks[selected_index].astype(np.float32)).unsqueeze(0)
        selected_masks.append(selected_mask)

        if target_index == active_index:
            for candidate_index, score in enumerate(scores):
                candidate_mask = torch.from_numpy(masks[candidate_index].astype(np.float32)).unsqueeze(0)
                candidate_preview = _make_preview(
                    image_batch,
                    [candidate_mask[0]],
                    float(preview_opacity),
                    prompt=target,
                )
                active_candidates.append(
                    {
                        "index": int(candidate_index),
                        "score": float(score),
                        "preview_url": _encode_png_data_url(_image_tensor_to_numpy(candidate_preview)),
                        "selected": candidate_index == selected_index,
                    }
                )

    active_prompt = targets[active_index] if targets and active_index < len(targets) else None
    merged_preview = _make_preview(
        image_batch,
        [mask[0] for mask in selected_masks],
        float(preview_opacity),
        prompt=active_prompt if active_prompt and active_prompt.get("points") else active_prompt,
    )

    return {
        "selected_masks": selected_masks,
        "preview": merged_preview,
        "active_candidates": active_candidates,
    }


def _run_interactive_editor_node(
    sam_model: Dict[str, Any],
    image: torch.Tensor,
    editor_state: Dict[str, Any],
    preview_opacity: float,
    merge_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _require_segment_anything()

    from segment_anything import SamPredictor

    image_batch = _prepare_single_image(image)
    predictor = SamPredictor(sam_model["model"])
    predictor.set_image(_image_tensor_to_numpy(image_batch))

    prediction = _predict_masks_for_editor_targets(
        predictor=predictor,
        image_batch=image_batch,
        editor_state=editor_state,
        preview_opacity=preview_opacity,
    )

    if not prediction["selected_masks"]:
        # First queue: no masks yet — return blank mask and raw frame preview
        h, w = image_batch.shape[1], image_batch.shape[2]
        blank_mask = torch.zeros((1, h, w), dtype=torch.float32)
        return blank_mask, image_batch

    merged_mask = _merge_mask_list(prediction["selected_masks"], threshold=float(merge_threshold))
    return merged_mask, prediction["preview"]


def _cleanup_interactive_sam_sessions() -> None:
    now = time.time()
    with _INTERACTIVE_SAM_SESSIONS_LOCK:
        expired = [
            session_id
            for session_id, session in _INTERACTIVE_SAM_SESSIONS.items()
            if now - float(session.get("last_access", session.get("created_at", now))) > _INTERACTIVE_SAM_SESSION_TTL_SECONDS
        ]
        for session_id in expired:
            _INTERACTIVE_SAM_SESSIONS.pop(session_id, None)


def _create_interactive_sam_session(
    image_data: str,
    sam_model_type: str,
    checkpoint_path: str,
    device: str,
) -> Dict[str, Any]:
    _require_segment_anything()

    from segment_anything import SamPredictor

    _cleanup_interactive_sam_sessions()
    image_np = _decode_image_data_url(image_data)
    resolved_checkpoint = _ensure_sam_checkpoint(sam_model_type, checkpoint_path)
    resolved_device = _resolve_device(device)
    sam_model = _get_cached_sam_model(sam_model_type, resolved_checkpoint, resolved_device)

    predictor = SamPredictor(sam_model["model"])
    predictor.set_image(image_np)

    session_id = uuid.uuid4().hex
    now = time.time()
    session = {
        "id": session_id,
        "created_at": now,
        "last_access": now,
        "predictor": predictor,
        "image_np": image_np,
        "width": int(image_np.shape[1]),
        "height": int(image_np.shape[0]),
        "lock": threading.Lock(),
    }
    with _INTERACTIVE_SAM_SESSIONS_LOCK:
        _INTERACTIVE_SAM_SESSIONS[session_id] = session
    return session


def _get_interactive_sam_session(session_id: str) -> Dict[str, Any]:
    _cleanup_interactive_sam_sessions()
    with _INTERACTIVE_SAM_SESSIONS_LOCK:
        session = _INTERACTIVE_SAM_SESSIONS.get(session_id)
        if session is None:
            raise KeyError(f"Interactive SAM session not found: {session_id}")
        session["last_access"] = time.time()
        return session


def _close_interactive_sam_session(session_id: str) -> None:
    with _INTERACTIVE_SAM_SESSIONS_LOCK:
        _INTERACTIVE_SAM_SESSIONS.pop(session_id, None)


class MatAnyoneModelLoader:
    DESCRIPTION = "Loads a MatAnyone or MatAnyone2 checkpoint and auto-downloads it when no path is provided."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(MODEL_VARIANTS.keys()), {"default": "MatAnyone 2"}),
                "checkpoint_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Leave blank to auto-download the selected checkpoint",
                    },
                ),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("MATANYONE_MODEL",)
    RETURN_NAMES = ("matanyone_model",)
    FUNCTION = "load_model"
    CATEGORY = MATANYONE_CATEGORY

    def load_model(self, model_name: str, checkpoint_path: str, device: str):
        resolved_checkpoint = _ensure_model_checkpoint(model_name, checkpoint_path)
        resolved_device = _resolve_device(device)
        return (_get_cached_matanyone_model(model_name, resolved_checkpoint, resolved_device),)


class MatAnyoneSAMLoader:
    DESCRIPTION = "Loads a Segment Anything checkpoint for scripted or reusable SAM refinement."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model_type": (list(SAM_VARIANTS.keys()), {"default": "vit_h"}),
                "checkpoint_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Leave blank to auto-download the selected SAM checkpoint",
                    },
                ),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("MATANYONE_SAM",)
    RETURN_NAMES = ("sam_model",)
    FUNCTION = "load_sam"
    CATEGORY = MATANYONE_SAM_CATEGORY

    def load_sam(self, sam_model_type: str, checkpoint_path: str, device: str):
        resolved_checkpoint = _ensure_sam_checkpoint(sam_model_type, checkpoint_path)
        resolved_device = _resolve_device(device)
        return (_get_cached_sam_model(sam_model_type, resolved_checkpoint, resolved_device),)


class MatAnyoneSliceFrames:
    DESCRIPTION = "Slices an IMAGE batch down to the frame range you want to edit or matte."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "end_frame_exclusive": ("INT", {"default": -1, "min": -1, "max": 999999}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("images", "start_frame_used", "end_frame_used")
    FUNCTION = "slice_frames"
    CATEGORY = MATANYONE_CATEGORY

    def slice_frames(self, images: torch.Tensor, start_frame: int, end_frame_exclusive: int):
        batch = _prepare_image_batch(images)
        frame_count = batch.shape[0]
        if frame_count == 0:
            raise ValueError("No frames were provided.")
        start_index = min(max(int(start_frame), 0), frame_count - 1)
        if end_frame_exclusive < 0:
            end_index = frame_count
        else:
            end_index = min(max(int(end_frame_exclusive), start_index + 1), frame_count)
        return (batch[start_index:end_index].contiguous(), start_index, end_index)


class MatAnyoneSelectFrame:
    DESCRIPTION = "Extracts one frame from an IMAGE batch for interactive masking or prompt setup."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 999999}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "frame_index_used")
    FUNCTION = "select_frame"
    CATEGORY = MATANYONE_CATEGORY

    def select_frame(self, images: torch.Tensor, frame_index: int):
        batch = _prepare_image_batch(images)
        if batch.shape[0] == 0:
            raise ValueError("No frames were provided.")
        resolved_index = min(max(int(frame_index), 0), batch.shape[0] - 1)
        return (batch[resolved_index:resolved_index + 1].contiguous(), resolved_index)


class MatAnyonePromptStart:
    DESCRIPTION = "Creates an empty SAM prompt container for point-based edits."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("MATANYONE_PROMPT",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "start"
    CATEGORY = MATANYONE_SAM_CATEGORY

    def start(self):
        return (_empty_prompt(),)


class MatAnyonePromptFromText:
    DESCRIPTION = "Parses x,y,+/- lines into a reusable SAM prompt structure."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "512,320,+\n540,318,-",
                    },
                )
            }
        }

    RETURN_TYPES = ("MATANYONE_PROMPT", "INT")
    RETURN_NAMES = ("prompt", "point_count")
    FUNCTION = "from_text"
    CATEGORY = MATANYONE_SAM_CATEGORY

    def from_text(self, prompt_text: str):
        prompt = _parse_prompt_text(prompt_text)
        return (prompt, len(prompt["points"]))


class MatAnyoneAddPoint:
    DESCRIPTION = "Adds one positive or negative point to an existing SAM prompt."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("MATANYONE_PROMPT",),
                "x": ("INT", {"default": 512, "min": 0, "max": 16384}),
                "y": ("INT", {"default": 320, "min": 0, "max": 16384}),
                "label": (["positive", "negative"], {"default": "positive"}),
            }
        }

    RETURN_TYPES = ("MATANYONE_PROMPT", "INT")
    RETURN_NAMES = ("prompt", "point_count")
    FUNCTION = "add_point"
    CATEGORY = MATANYONE_SAM_CATEGORY

    def add_point(self, prompt: Dict[str, Any], x: int, y: int, label: str):
        updated = _clone_prompt(prompt)
        updated["points"].append([int(x), int(y)])
        updated["labels"].append(1 if label == "positive" else 0)
        return (updated, len(updated["points"]))


class MatAnyoneSAMRefine:
    DESCRIPTION = "Runs SAM against a frame and prompt, returning the chosen mask, preview, logits, and score."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model": ("MATANYONE_SAM",),
                "image": ("IMAGE",),
                "prompt": ("MATANYONE_PROMPT",),
                "multimask_output": ("BOOLEAN", {"default": True}),
                "mask_choice": (["best", "0", "1", "2"], {"default": "best"}),
                "use_previous_logits": ("BOOLEAN", {"default": True}),
                "preview_opacity": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "previous_logits": ("MATANYONE_SAM_LOGITS",),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "MATANYONE_SAM_LOGITS", "FLOAT")
    RETURN_NAMES = ("mask", "preview", "logits", "score")
    FUNCTION = "refine"
    CATEGORY = MATANYONE_SAM_CATEGORY

    def refine(
        self,
        sam_model: Dict[str, Any],
        image: torch.Tensor,
        prompt: Dict[str, Any],
        multimask_output: bool,
        mask_choice: str,
        use_previous_logits: bool,
        preview_opacity: float,
        previous_logits: Optional[Dict[str, Any]] = None,
    ):
        _require_segment_anything()

        from segment_anything import SamPredictor

        image_batch = _prepare_single_image(image)
        image_np = (image_batch[0].numpy() * 255.0).round().astype(np.uint8)
        points, labels = _prompt_to_numpy(prompt)

        predictor = SamPredictor(sam_model["model"])
        predictor.set_image(image_np)

        mask_input = None
        if previous_logits is not None and use_previous_logits:
            mask_input = previous_logits.get("low_res")
            if isinstance(mask_input, torch.Tensor):
                mask_input = mask_input.detach().cpu().numpy()
            if mask_input is not None and mask_input.ndim == 2:
                mask_input = mask_input[None, :, :]

        masks, scores, low_res_masks = predictor.predict(
            point_coords=points,
            point_labels=labels,
            mask_input=mask_input,
            multimask_output=bool(multimask_output),
            return_logits=False,
        )

        selected_index = _normalize_mask_index(mask_choice, scores)
        selected_mask = torch.from_numpy(masks[selected_index].astype(np.float32)).unsqueeze(0)
        preview = _make_preview(image_batch, [selected_mask[0]], float(preview_opacity), prompt=prompt)
        logits = {
            "low_res": low_res_masks[selected_index].copy(),
            "low_res_candidates": low_res_masks.copy(),
            "scores": scores.copy(),
            "selected_index": int(selected_index),
        }
        return (selected_mask, preview, logits, float(scores[selected_index]))


class MatAnyoneMergeMasks:
    DESCRIPTION = "Combines up to four masks into one thresholded output mask."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_a": ("MASK",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "mask_b": ("MASK",),
                "mask_c": ("MASK",),
                "mask_d": ("MASK",),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "merge"
    CATEGORY = MATANYONE_SAM_CATEGORY

    def merge(
        self,
        mask_a: torch.Tensor,
        threshold: float,
        mask_b: Optional[torch.Tensor] = None,
        mask_c: Optional[torch.Tensor] = None,
        mask_d: Optional[torch.Tensor] = None,
    ):
        return (_merge_mask_list([mask_a, mask_b, mask_c, mask_d], threshold),)


class MatAnyonePreviewMasks:
    DESCRIPTION = "Draws one or more mask overlays on the source image for quick inspection."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask_a": ("MASK",),
                "opacity": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "mask_b": ("MASK",),
                "mask_c": ("MASK",),
                "mask_d": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "preview"
    CATEGORY = MATANYONE_SAM_CATEGORY

    def preview(
        self,
        image: torch.Tensor,
        mask_a: torch.Tensor,
        opacity: float,
        mask_b: Optional[torch.Tensor] = None,
        mask_c: Optional[torch.Tensor] = None,
        mask_d: Optional[torch.Tensor] = None,
    ):
        image_batch = _prepare_single_image(image)
        height, width = image_batch.shape[1:3]
        masks = []
        for mask in (mask_a, mask_b, mask_c, mask_d):
            if mask is None:
                continue
            masks.append(_resize_mask_to(mask, height, width))
        return (_make_preview(image_batch, masks, float(opacity)),)


class MatAnyoneInteractiveSAM:
    DESCRIPTION = "Opens the built-in multi-target SAM editor and returns the merged first-frame mask."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sam_model_type": (list(SAM_VARIANTS.keys()), {"default": "vit_h"}),
                "checkpoint_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Leave blank to auto-download the selected SAM checkpoint",
                    },
                ),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "editor_state": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Managed by the interactive editor",
                    },
                ),
                "preview_opacity": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "merge_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING", "INT")
    RETURN_NAMES = ("mask", "preview", "editor_state", "target_count")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = MATANYONE_SAM_CATEGORY

    def run(
        self,
        image: torch.Tensor,
        sam_model_type: str,
        checkpoint_path: str,
        device: str,
        editor_state: str,
        preview_opacity: float,
        merge_threshold: float,
    ):
        import folder_paths
        from PIL import Image as PILImage

        normalized_state = _normalize_editor_state(editor_state)
        image_batch = _prepare_single_image(image)

        # Save input frame as temp preview for the editor UI
        temp_dir = folder_paths.get_temp_directory()
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        frame_filename = f"matanyone_frame_{uuid.uuid4().hex[:12]}.png"
        frame_np = _image_tensor_to_numpy(image_batch)
        PILImage.fromarray(frame_np).save(str(Path(temp_dir) / frame_filename), format="PNG")
        ui_images = [{"filename": frame_filename, "type": "temp", "subfolder": ""}]

        # Check if any targets have actual points — if not, skip SAM entirely
        has_points = any(
            len(t.get("points", []) if isinstance(t, dict) else []) > 0
            for t in normalized_state.get("targets", [])
        )

        if not has_points:
            # First pass: return blank mask + raw frame, no SAM loading needed
            h, w = image_batch.shape[1], image_batch.shape[2]
            blank_mask = torch.zeros((1, h, w), dtype=torch.float32)
            result = (
                blank_mask,
                image_batch,
                _editor_state_to_json(normalized_state),
                0,
            )
            return {"ui": {"images": ui_images, "has_masks": [False]}, "result": result}

        # Second pass: load SAM and compute masks
        resolved_checkpoint = _ensure_sam_checkpoint(sam_model_type, checkpoint_path)
        resolved_device = _resolve_device(device)
        sam_model = _get_cached_sam_model(sam_model_type, resolved_checkpoint, resolved_device)
        merged_mask, preview = _run_interactive_editor_node(
            sam_model=sam_model,
            image=image,
            editor_state=normalized_state,
            preview_opacity=float(preview_opacity),
            merge_threshold=float(merge_threshold),
        )

        result = (
            merged_mask,
            preview,
            _editor_state_to_json(normalized_state),
            _editor_state_target_count(normalized_state),
        )
        return {"ui": {"images": ui_images, "has_masks": [True]}, "result": result}


class MatAnyoneMatte:
    DESCRIPTION = "Propagates the first-frame mask through the clip and returns foreground, alpha, and preview outputs."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "matanyone_model": ("MATANYONE_MODEL",),
                "images": ("IMAGE",),
                "first_frame_mask": ("MASK",),
                "warmup_iterations": ("INT", {"default": 10, "min": 0, "max": 128}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "erode_kernel": ("INT", {"default": 10, "min": 0, "max": 256}),
                "dilate_kernel": ("INT", {"default": 10, "min": 0, "max": 256}),
                "max_internal_size": ("INT", {"default": -1, "min": -1, "max": 4096}),
                "preview_red": ("INT", {"default": 120, "min": 0, "max": 255}),
                "preview_green": ("INT", {"default": 255, "min": 0, "max": 255}),
                "preview_blue": ("INT", {"default": 155, "min": 0, "max": 255}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("foreground", "alpha", "preview")
    FUNCTION = "run"
    CATEGORY = MATANYONE_CATEGORY

    def run(
        self,
        matanyone_model: Dict[str, Any],
        images: torch.Tensor,
        first_frame_mask: torch.Tensor,
        warmup_iterations: int,
        mask_threshold: float,
        invert_mask: bool,
        erode_kernel: int,
        dilate_kernel: int,
        max_internal_size: int,
        preview_red: int,
        preview_green: int,
        preview_blue: int,
    ):
        _require_matanyone()

        from matanyone2.inference.inference_core import InferenceCore

        frames = _prepare_image_batch(images)
        frame_count, height, width, _ = frames.shape
        if frame_count == 0:
            raise ValueError("No frames were provided.")

        init_mask = _prepare_binary_mask(
            first_frame_mask,
            height=height,
            width=width,
            threshold=float(mask_threshold),
            invert_mask=bool(invert_mask),
            erode_kernel=int(erode_kernel),
            dilate_kernel=int(dilate_kernel),
        )

        model = matanyone_model["model"]
        device = matanyone_model["device"]

        processor = InferenceCore(model, cfg=model.cfg, device=device)
        processor.max_internal_size = int(max_internal_size)

        frame_tensors = frames.permute(0, 3, 1, 2).contiguous()
        first_frame = frame_tensors[0].to(device)
        mask_tensor = init_mask.to(device)

        progress_bar = _get_progress_bar(frame_count)
        alphas: List[torch.Tensor] = []

        with torch.inference_mode():
            _check_interrupt()
            processor.step(first_frame, mask_tensor, objects=[1])

            output_prob = None
            for _ in range(int(warmup_iterations) + 1):
                output_prob = processor.step(first_frame, first_frame_pred=True)

            if output_prob is None:
                raise RuntimeError("MatAnyone warmup did not produce an alpha matte.")

            first_alpha = processor.output_prob_to_mask(output_prob).detach().float().cpu().clamp(0.0, 1.0)
            alphas.append(first_alpha)
            _update_progress(progress_bar, 1, frame_count)

            for frame_index in range(1, frame_count):
                _check_interrupt()
                output_prob = processor.step(frame_tensors[frame_index].to(device))
                alpha = processor.output_prob_to_mask(output_prob).detach().float().cpu().clamp(0.0, 1.0)
                alphas.append(alpha)
                _update_progress(progress_bar, frame_index + 1, frame_count)

        alpha_batch = torch.stack(alphas, dim=0).contiguous()
        alpha_rgb = alpha_batch.unsqueeze(-1)
        foreground = frames * alpha_rgb

        preview_bg = torch.tensor(
            [preview_red, preview_green, preview_blue],
            dtype=frames.dtype,
        ).view(1, 1, 1, 3) / 255.0
        preview = foreground + preview_bg * (1.0 - alpha_rgb)

        _soft_empty_cache()
        return (foreground, alpha_batch, preview)


try:
    from aiohttp import web
    from server import PromptServer
except Exception:
    PromptServer = None
else:
    @PromptServer.instance.routes.post("/matanyone2/interactive/create_session")
    async def matanyone_create_session(request):
        try:
            payload = await request.json()
            session = _create_interactive_sam_session(
                image_data=str(payload.get("image_data", "")),
                sam_model_type=str(payload.get("sam_model_type", "vit_h")),
                checkpoint_path=str(payload.get("checkpoint_path", "")),
                device=str(payload.get("device", "auto")),
            )
            return web.json_response(
                {
                    "session_id": session["id"],
                    "width": session["width"],
                    "height": session["height"],
                }
            )
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)


    @PromptServer.instance.routes.post("/matanyone2/interactive/predict")
    async def matanyone_predict_interactive(request):
        try:
            payload = await request.json()
            session = _get_interactive_sam_session(str(payload.get("session_id", "")))
            editor_state = _normalize_editor_state(payload.get("editor_state"))
            preview_opacity = float(payload.get("preview_opacity", 0.65))
            image_batch = _numpy_image_to_tensor(session["image_np"])

            with session["lock"]:
                prediction = _predict_masks_for_editor_targets(
                    predictor=session["predictor"],
                    image_batch=image_batch,
                    editor_state=editor_state,
                    preview_opacity=preview_opacity,
                )

            return web.json_response(
                {
                    "state": editor_state,
                    "preview_url": _encode_png_data_url(_image_tensor_to_numpy(prediction["preview"])),
                    "active_candidates": prediction["active_candidates"],
                    "target_count": _editor_state_target_count(editor_state),
                    "point_count": _editor_state_prompt_count(editor_state),
                }
            )
        except KeyError as exc:
            return web.json_response({"error": str(exc)}, status=404)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)


    @PromptServer.instance.routes.post("/matanyone2/interactive/close_session")
    async def matanyone_close_session(request):
        try:
            payload = await request.json()
            _close_interactive_sam_session(str(payload.get("session_id", "")))
            return web.json_response({"ok": True})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)


NODE_CLASS_MAPPINGS = {
    "MatAnyoneModelLoader": MatAnyoneModelLoader,
    "MatAnyoneSAMLoader": MatAnyoneSAMLoader,
    "MatAnyoneSliceFrames": MatAnyoneSliceFrames,
    "MatAnyoneSelectFrame": MatAnyoneSelectFrame,
    "MatAnyonePromptStart": MatAnyonePromptStart,
    "MatAnyonePromptFromText": MatAnyonePromptFromText,
    "MatAnyoneAddPoint": MatAnyoneAddPoint,
    "MatAnyoneSAMRefine": MatAnyoneSAMRefine,
    "MatAnyoneMergeMasks": MatAnyoneMergeMasks,
    "MatAnyonePreviewMasks": MatAnyonePreviewMasks,
    "MatAnyoneInteractiveSAM": MatAnyoneInteractiveSAM,
    "MatAnyoneMatte": MatAnyoneMatte,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MatAnyoneModelLoader": "MatAnyone2 Model Loader",
    "MatAnyoneSAMLoader": "MatAnyone2 SAM Loader",
    "MatAnyoneSliceFrames": "MatAnyone2 Slice Frames",
    "MatAnyoneSelectFrame": "MatAnyone2 Select Frame",
    "MatAnyonePromptStart": "MatAnyone2 Prompt Start",
    "MatAnyonePromptFromText": "MatAnyone2 Prompt From Text",
    "MatAnyoneAddPoint": "MatAnyone2 Add Point",
    "MatAnyoneSAMRefine": "MatAnyone2 SAM Refine",
    "MatAnyoneMergeMasks": "MatAnyone2 Merge Masks",
    "MatAnyonePreviewMasks": "MatAnyone2 Preview Masks",
    "MatAnyoneInteractiveSAM": "MatAnyone2 Interactive SAM",
    "MatAnyoneMatte": "MatAnyone2 Matte",
}
