from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "camera": {
        "source": 0,
        "fps": 30,
        "width": 1280,
        "height": 720,
        "reconnect_interval_seconds": 5,
    },
    "recognition": {
        "confirm_threshold": 0.60,
        "suspect_threshold": 0.40,
        "vote_window": 5,
        "vote_confirm_count": 3,
        "max_faces_per_frame": 3,
        "detection_fps": 8,
        "embedding_interval_seconds": 1.5,
        "model_name": "buffalo_l",
        "det_size": 320,
        "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    },
    "curfew": {
        "enabled": True,
        "curfew_time": "22:30",
        "grace_period_minutes": 10,
        "duplicate_window_minutes": 10,
    },
    "event": {
        "stranger_duplicate_window_minutes": 5,
        "snapshot_retention_days": 7,
        "enable_led_alert": True,
        "enable_buzzer_alert": False,
    },
    "storage": {
        "database_path": "data/runtime/face_access.db",
        "snapshot_dir": "data/snapshots",
        "registered_face_dir": "data/registered_faces",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path = "config/config.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return DEFAULT_CONFIG

    try:
        import yaml
    except ImportError:
        loaded = _load_simple_yaml(config_path)
        return _deep_merge(DEFAULT_CONFIG, loaded)

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    return _deep_merge(DEFAULT_CONFIG, loaded)


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    current_list_key: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue

        if not raw_line.startswith(" ") and line.endswith(":"):
            section_name = line[:-1].strip()
            current_section = {}
            result[section_name] = current_section
            current_list_key = None
            continue

        if current_section is None or ":" not in line:
            stripped = line.strip()
            if current_section is not None and current_list_key and stripped.startswith("- "):
                current_section[current_list_key].append(_parse_scalar(stripped[2:].strip()))
            continue

        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value == "":
            current_section[key] = []
            current_list_key = key
        else:
            current_section[key] = _parse_scalar(raw_value)
            current_list_key = None

    return result


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
