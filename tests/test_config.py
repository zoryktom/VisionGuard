"""Config loader tests."""

from visionguard.config import load_config, repo_root, resolve_path


def test_load_default_config() -> None:
    cfg = load_config()
    assert cfg.inference.imgsz == 640
    assert cfg.api.port == 8000
    assert cfg.capture.fps == 30


def test_overrides_merge() -> None:
    cfg = load_config(overrides={"inference": {"backend": "dummy", "confidence": 0.5}})
    assert cfg.inference.backend == "dummy"
    assert cfg.inference.confidence == 0.5
    assert cfg.inference.imgsz == 640


def test_resolve_path_relative() -> None:
    path = resolve_path("configs/default.yaml")
    assert path.exists()
    assert repo_root() in path.parents or path.parent == repo_root()
