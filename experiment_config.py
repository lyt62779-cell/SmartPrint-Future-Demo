"""Configuration and environment provenance for reproducible experiments."""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    mode: str = "fusion"
    dpi_scale: float = 3.0
    ocr_low_confidence: float = 0.90
    difference_threshold: int = 25
    blur_kernel: int = 5
    morphology_kernel: int = 3
    morphology_iterations: int = 1
    min_area: float = 30.0
    merge_gap: int = 5
    orb_max_features: int = 3000
    orb_keep_match_ratio: float = 0.25
    orb_min_matches: int = 8
    ransac_threshold: float = 3.0
    orb_min_inlier_ratio: float = 0.20
    max_rotation_degrees: float = 10.0
    min_scale: float = 0.85
    max_scale: float = 1.15
    max_roi_pixels: int = 20_000_000
    ocr_batch_size: int = 8
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in {"visual", "ocr", "fusion"}:
            raise ValueError("mode must be visual, ocr, or fusion")
        for name in ("dpi_scale", "min_area", "ransac_threshold", "max_rotation_degrees", "min_scale", "max_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, minimum in (("difference_threshold", 0), ("blur_kernel", 1), ("morphology_kernel", 1), ("morphology_iterations", 0), ("merge_gap", 0), ("orb_max_features", 100), ("orb_min_matches", 3), ("max_roi_pixels", 1), ("ocr_batch_size", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.difference_threshold > 255:
            raise ValueError("difference_threshold must be <= 255")
        for name in ("blur_kernel", "morphology_kernel"):
            if getattr(self, name) % 2 != 1:
                raise ValueError(f"{name} must be odd")
        for name in ("orb_keep_match_ratio", "orb_min_inlier_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.min_scale > self.max_scale:
            raise ValueError("min_scale must be <= max_scale")
        if self.ocr_low_confidence != 0.90 or isinstance(self.ocr_low_confidence, bool):
            raise ValueError("ocr_low_confidence is fixed at the implemented multilingual_ocr value 0.90")
        if not isinstance(self.extra, dict):
            raise ValueError("extra must be metadata, not algorithm overrides")
        json.dumps(self.extra, allow_nan=False)

    def alignment_options(self) -> dict[str, Any]:
        self.validate()
        return {
            "max_features": self.orb_max_features,
            "keep_match_ratio": self.orb_keep_match_ratio,
            "min_matches": self.orb_min_matches,
            "ransac_threshold": self.ransac_threshold,
            "min_inlier_ratio": self.orb_min_inlier_ratio,
            "max_rotation_degrees": self.max_rotation_degrees,
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
        }

    def execution_parameters(self) -> dict[str, Any]:
        """Record fixed implementation choices separately from editable settings."""
        from region_separator import RegionSeparatorConfig

        return {
            "alignment": self.alignment_options(),
            "matcher": {"method": "BFMatcher", "norm": "NORM_HAMMING", "cross_check": True},
            "ransac": {"method": "estimateAffinePartial2D", "max_iters": 3000, "confidence": 0.99, "refine_iters": 10},
            "ocr": {"detection_model": "PP-OCRv6_medium_det", "recognition_model": "PP-OCRv6_medium_rec", "enable_mkldnn": False, "doc_orientation": True, "doc_unwarping": False, "textline_orientation": False, "low_confidence": 0.90, "western_review_confidence": 0.97, "ambiguous_latin_review_confidence": 0.94, "review_batch_limit": 8, "panel_batch_size": self.ocr_batch_size},
            "template_search": {"rotations": [0, 90, 180, 270], "scales": [0.82, 0.90, 0.96, 1.0, 1.04, 1.10, 1.18], "gray_weight": 0.65, "edge_weight": 0.35, "candidate_threshold": 0.38},
            "fusion": "legacy_rule_OR_with_movement_warning",
            "cnn_batch_size": 32,
            "region_separator": asdict(RegionSeparatorConfig()),
            "movement": {"translation_ratio": 0.005, "translation_pixels_without_size": 3.0, "rotation_degrees": 0.5, "scale_change": 0.01},
            "ssim": {"gaussian_kernel": 11, "gaussian_sigma": 1.5, "grid": [4, 4], "worst_fraction": 0.25, "anomaly_threshold": 0.95, "reference_score_weights": {"global": 0.30, "local": 0.30, "area": 0.25, "severity": 0.15}},
            "timing_semantics": {"ocr_s": "CNN routing, cropping, primary prediction, multilingual review, and record mapping", "model_initialization_s": "CNN readiness and PaddleOCR construction; secondary model lazy initialization remains in ocr_prediction_and_review_s", "ocr_prediction_and_review_s": "primary prediction and multilingual review including any lazy secondary model initialization; not pure inference"},
            "input_coordinates": "zero-based pages; xyxy PDF page points or original image pixels",
            "extra_role": "metadata only; never algorithm parameter overrides",
        }

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_value(project_dir: Path, args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_dir), *args],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def source_fingerprint(project_dir: str | Path) -> dict[str, Any]:
    project = Path(project_dir).resolve()
    files = sorted({*project.glob("*.py"), *project.glob("requirements*.txt")})
    hashes = {path.name: file_sha256(path) for path in files}
    return {"sha256": stable_hash(hashes), "files": hashes, "scope": "top-level Python implementation and requirements files"}


def collect_environment(project_dir: str | Path) -> dict[str, Any]:
    project = Path(project_dir).resolve()
    return {
        "git_commit": _git_value(project, ["rev-parse", "HEAD"]),
        "git_branch": _git_value(project, ["branch", "--show-current"]),
        "git_status": _git_value(project, ["status", "--porcelain"]),
        "source_fingerprint": source_fingerprint(project),
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "gpu": _gpu_name(),
        "ram_mb": _ram_mb(),
        "packages": {
            "paddleocr": _package_version("paddleocr"),
            "paddlex": _package_version("paddlex"),
            "pymupdf": _package_version("PyMuPDF"),
            "opencv": _package_version("opencv-python"),
            "numpy": _package_version("numpy"),
            "paddlepaddle": _package_version("paddlepaddle"),
            "paddlepaddle_gpu": _package_version("paddlepaddle-gpu"),
            "pillow": _package_version("Pillow"),
            "psutil": _package_version("psutil"),
        },
        "cnn": _cnn_environment(project),
        "paddlex_pdf": {"render_scale_environment": os.environ.get("PADDLE_PDX_PDF_RENDER_SCALE"), "minimum_render_scale_environment": os.environ.get("PADDLE_PDX_PDF_MIN_RENDER_SCALE"), "effective_render_scale": None, "effective_max_image_pixels": None, "note": "Single-file GUI PDF rendering is library-controlled. Effective library values are not inferred without runtime observation; experiment ROI rendering uses config.dpi_scale."},
    }


def _cnn_environment(project: Path) -> dict[str, Any]:
    interpreter = Path(os.environ.get("CNN_CLASSIFIER_PYTHON", sys.executable))
    checkpoint = project / "model_cache" / "cnn_region_classifier" / "best_efficientnet_b0_classifier.pth"
    packages = None
    if interpreter.is_file():
        script = "import importlib.metadata as m,json,sys; print(json.dumps({'python':sys.version,'torch':m.version('torch'),'torchvision':m.version('torchvision')}))"
        try:
            packages = json.loads(subprocess.check_output([str(interpreter), "-c", script], text=True, stderr=subprocess.DEVNULL, timeout=15))
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return {"python_executable": str(interpreter), "packages": packages, "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint) if checkpoint.is_file() else None}


def _gpu_name() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _ram_mb() -> float | None:
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().total / 1024**2, 2)
    except ImportError:
        return None


def write_experiment_config(
    path: str | Path,
    config: ExperimentConfig,
    project_dir: str | Path,
    *,
    provenance: dict[str, Any] | None = None,
    expected_common_identity: str | None = None,
) -> dict[str, Any]:
    payload = {"config": config.to_dict(), "execution_parameters": config.execution_parameters(), "environment": collect_environment(project_dir), "provenance": provenance}
    environment = payload["environment"]
    identity = {"config": payload["config"], "execution_parameters": payload["execution_parameters"], "provenance": provenance, "source_fingerprint": environment["source_fingerprint"], "packages": environment["packages"], "cnn": environment["cnn"], "python": environment["python"], "python_executable": environment["python_executable"], "paddlex_pdf": environment.get("paddlex_pdf")}
    payload["identity_sha256"] = stable_hash(identity)
    common = dict(identity, config={key: value for key, value in payload["config"].items() if key != "mode"})
    payload["common_identity_sha256"] = stable_hash(common)
    if expected_common_identity is not None and expected_common_identity != payload["common_identity_sha256"]:
        raise ValueError("Cross-mode inputs, ground truth, non-mode configuration, or code differ; choose a new output directory")
    target = Path(path)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("identity_sha256") != payload["identity_sha256"]:
            raise ValueError("Existing run configuration differs; choose a new output directory")
        return existing
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    return payload

