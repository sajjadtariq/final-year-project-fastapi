"""EfficientNet-B3 adapter with checkpoint-defined preprocessing and thresholds."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision.models import efficientnet_b3

from ..errors import ModelExecutionError
from .preprocessing import build_efficientnet_transform

CLASS_TO_IDX = {
    "nevus": 0,
    "melanoma": 1,
    "bcc": 2,
    "acne": 3,
    "eczema": 4,
    "psoriasis": 5,
}
IDX_TO_CLASS = {index: name for name, index in CLASS_TO_IDX.items()}


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ModelExecutionError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise ModelExecutionError("MPS was requested but is unavailable")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EfficientNetClassifier:
    """Load the selected checkpoint once and classify prepared PIL images."""

    classes = tuple(CLASS_TO_IDX)

    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        if not checkpoint_path.is_file():
            raise ModelExecutionError(
                f"EfficientNet checkpoint not found: {checkpoint_path}"
            )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        required = {
            "model_state_dict",
            "class_to_idx",
            "configuration",
            "image_size",
            "decision_thresholds",
        }
        missing = required - checkpoint.keys()
        if missing:
            raise ModelExecutionError(
                f"EfficientNet checkpoint is missing keys: {sorted(missing)}"
            )
        if checkpoint["class_to_idx"] != CLASS_TO_IDX:
            raise ModelExecutionError("EfficientNet class mapping is incompatible")

        configuration = checkpoint["configuration"]
        model = efficientnet_b3(weights=None)
        input_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=float(configuration["model"]["dropout"]), inplace=True),
            nn.Linear(input_features, len(CLASS_TO_IDX)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        self.device = device
        self.model = model.to(device).eval()
        self.image_size = int(checkpoint["image_size"])
        self.thresholds = {
            key: float(value)
            for key, value in checkpoint["decision_thresholds"].items()
        }
        self.transform = build_efficientnet_transform(self.image_size)
        self.model_version = str(checkpoint.get("model_version", "untracked"))
        registry = checkpoint.get("registry_version")
        self.registry_version = None if registry is None else str(registry)
        self.mlflow_run_id = checkpoint.get("mlflow_run_id")

    def predict(self, image: Image.Image) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            values = torch.softmax(self.model(tensor), dim=1)[0].cpu().tolist()
        probabilities = {
            IDX_TO_CLASS[index]: float(value)
            for index, value in enumerate(values)
        }
        raw_class = max(probabilities, key=probabilities.get)
        confidence = probabilities[raw_class]
        threshold = float(
            self.thresholds.get(raw_class, self.thresholds["default"])
        )
        accepted = confidence >= threshold
        result = {
            "accepted": accepted,
            "class_name": raw_class if accepted else None,
            "raw_predicted_class": raw_class,
            "confidence": confidence,
            "threshold_used": threshold,
            "probabilities": probabilities,
        }
        elapsed_ms = (time.perf_counter() - started) * 1000
        return result, elapsed_ms

    def warmup(self) -> None:
        tensor = torch.zeros(
            (1, 3, self.image_size, self.image_size),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            self.model(tensor)


# File summary:
# - select_device resolves explicit or automatic CPU/CUDA/MPS execution.
# - EfficientNetClassifier validates the checkpoint, loads it once, applies its
#   stored thresholds, and returns independent probabilities and timing.
