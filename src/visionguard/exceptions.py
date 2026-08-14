"""Domain exceptions."""


class VisionGuardError(Exception):
    """Base error for the VisionGuard runtime."""


class ConfigError(VisionGuardError):
    """Invalid or missing configuration."""


class CaptureError(VisionGuardError):
    """Video source could not be opened or read."""


class InferenceError(VisionGuardError):
    """Detector backend failed."""


class TrainingError(VisionGuardError):
    """Training or export pipeline failed."""
