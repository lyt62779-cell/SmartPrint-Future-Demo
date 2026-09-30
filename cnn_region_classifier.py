"""工业区域 CNN 二分类推理与业务路由。

本模块只负责把上游已经产生的候选区域分成两类：

    0 -> document_text -> 后续可送入现有 OCR
    1 -> image_content -> 后续可送入图片内容处理流程

它不会调用 OCR，也不会调用图片差分，更不会修改二者的实现。当前模型是
区域分类器而不是目标检测器，因此调用方必须提供候选框 ``[x, y, w, h]``。

主 PaddleOCR 环境未安装 PyTorch；为避免改动现有环境，本模块启动一个使用
本机既有 PyTorch 环境的常驻子进程。模型只加载一次，后续区域采用批量推理。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Sequence


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    BASE_DIR
    / "model_cache"
    / "cnn_region_classifier"
    / "best_efficientnet_b0_classifier.pth"
)
DEFAULT_RUNTIME_PYTHON = Path(os.environ.get("CNN_CLASSIFIER_PYTHON", sys.executable))

CLASS_NAMES = ("document_text", "image_content")
DOCUMENT_TEXT_CLASS_ID = 0
IMAGE_CONTENT_CLASS_ID = 1
INPUT_SIZE = 224
INFERENCE_BATCH_SIZE = 32
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

Box = tuple[int, int, int, int]


def _normalize_box(box: Sequence[int | float]) -> Box:
    """把候选框统一为 ``(x, y, width, height)`` 并拒绝空区域。"""

    if len(box) != 4:
        raise ValueError(f"区域框必须包含 4 个数值，收到：{box}")
    x, y, width, height = (int(round(float(value))) for value in box)
    if width <= 0 or height <= 0:
        raise ValueError(f"区域框宽高必须大于 0，收到：{box}")
    return x, y, width, height


def route_predictions(
    predictions: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """按固定类别顺序把预测结果路由为文字组和图片组。"""

    routed: dict[str, list[dict[str, Any]]] = {
        "document_text": [],
        "image_content": [],
    }
    for prediction in predictions:
        class_id = int(prediction["class_id"])
        class_name = str(prediction["class_name"])
        if class_id not in (DOCUMENT_TEXT_CLASS_ID, IMAGE_CONTENT_CLASS_ID):
            raise ValueError(f"模型返回未知类别编号：{class_id}")
        expected_name = CLASS_NAMES[class_id]
        if class_name != expected_name:
            raise ValueError(
                f"类别顺序不一致：class_id={class_id} 应为 {expected_name}，"
                f"实际为 {class_name}"
            )
        routed[class_name].append(dict(prediction))
    return routed


class CNNRegionClassifier:
    """通过常驻 PyTorch 子进程执行 EfficientNet-B0 区域分类。"""

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        runtime_python: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        configured_runtime = runtime_python or os.environ.get(
            "CNN_CLASSIFIER_PYTHON"
        )
        self.runtime_python = Path(
            configured_runtime or DEFAULT_RUNTIME_PYTHON
        )
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"不支持的推理设备：{device}")
        self.device = device
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def _start_worker(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"CNN 最佳检查点不存在：{self.checkpoint_path}"
            )
        if self.checkpoint_path.name != "best_efficientnet_b0_classifier.pth":
            raise ValueError(
                "只允许使用 best_efficientnet_b0_classifier.pth，"
                f"收到：{self.checkpoint_path.name}"
            )
        if not self.runtime_python.is_file():
            raise FileNotFoundError(
                f"CNN PyTorch 运行时不存在：{self.runtime_python}"
            )

        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONUTF8"] = "1"
        command = [
            str(self.runtime_python),
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            "--checkpoint",
            str(self.checkpoint_path.resolve()),
            "--device",
            self.device,
        ]
        self._process = subprocess.Popen(
            command,
            cwd=str(BASE_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=environment,
            creationflags=(
                subprocess.CREATE_NO_WINDOW
                if os.name == "nt"
                else 0
            ),
        )
        ready = self._read_response()
        if not ready.get("ready"):
            self.close()
            raise RuntimeError(f"CNN 分类工作进程启动失败：{ready}")

    def _read_response(self) -> dict[str, Any]:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("CNN 分类工作进程尚未启动")
        line = self._process.stdout.readline()
        if not line:
            stderr = ""
            if self._process.stderr is not None:
                stderr = self._process.stderr.read().strip()
            raise RuntimeError(
                "CNN 分类工作进程意外退出"
                + (f"：{stderr}" if stderr else "")
            )
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(
                f"CNN 区域分类失败：{response['error_type']}: "
                f"{response['error']}"
            )
        return response

    def ensure_ready(self) -> None:
        """Load the persistent model before timing inference separately."""
        with self._lock:
            self._start_worker()

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._start_worker()
            if self._process is None or self._process.stdin is None:
                raise RuntimeError("CNN 分类工作进程不可用")
            self._process.stdin.write(
                json.dumps(payload, ensure_ascii=False) + "\n"
            )
            self._process.stdin.flush()
            return self._read_response()

    def predict_regions(
        self,
        image_path: str | Path,
        regions: Sequence[Sequence[int | float]],
    ) -> list[dict[str, Any]]:
        """批量分类同一张图上的候选区域。

        ``regions`` 坐标格式固定为 ``[x, y, width, height]``。返回结果
        保留 bbox、类别编号、类别名称、置信度和两类 softmax 概率。
        """

        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"待分类图片不存在：{path}")
        normalized_regions = [
            list(_normalize_box(region))
            for region in regions
        ]
        if not normalized_regions:
            return []
        response = self._request(
            {
                "command": "predict_regions",
                "image_path": str(path.resolve()),
                "regions": normalized_regions,
            }
        )
        predictions = list(response["predictions"])
        if len(predictions) != len(normalized_regions):
            raise RuntimeError(
                "CNN 返回区域数量不一致："
                f"输入 {len(normalized_regions)}，输出 {len(predictions)}"
            )
        return predictions

    def classify_and_route(
        self,
        image_path: str | Path,
        regions: Sequence[Sequence[int | float]],
    ) -> dict[str, list[dict[str, Any]]]:
        """分类候选框并返回 OCR 组和图片内容组。"""

        return route_predictions(
            self.predict_regions(image_path, regions)
        )

    def close(self) -> None:
        """关闭常驻工作进程；不删除模型或任何业务数据。"""

        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write('{"command":"shutdown"}\n')
                    process.stdin.flush()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()

    def __enter__(self) -> "CNNRegionClassifier":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _build_worker_model(checkpoint_path: Path, device_name: str):
    """在 PyTorch 工作进程内按训练结构加载模型，不下载预训练权重。"""

    import torch
    from torch import nn
    from torchvision.models import efficientnet_b0

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但当前没有可用 CUDA GPU")
        device = torch.device("cuda")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # weights=None 对应 build_model(pretrained=False)，不会联网下载或加载
    # ImageNet 权重。分类头严格保持 Linear(1280, 2)。
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    if in_features != 1280:
        raise RuntimeError(
            f"EfficientNet-B0 分类头输入应为 1280，实际为 {in_features}"
        )
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 2),
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    metadata = checkpoint.get("metadata", {})
    checkpoint_classes = tuple(
        metadata.get("class_names", CLASS_NAMES)
    )
    if checkpoint_classes != CLASS_NAMES:
        raise ValueError(
            f"检查点类别顺序不兼容：{checkpoint_classes}"
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, device


def _preprocess_worker_image(image):
    """严格复现训练时的 RGB、等比例白边、Tensor 和 ImageNet 归一化。"""

    import numpy as np
    import torch
    from PIL import Image

    image = image.convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"非法区域尺寸：{image.size}")
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    resized_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (255, 255, 255))
    canvas.paste(
        resized,
        (
            (INPUT_SIZE - resized_size[0]) // 2,
            (INPUT_SIZE - resized_size[1]) // 2,
        ),
    )

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).contiguous()
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def _worker_predict_regions(
    model,
    device,
    image_path: Path,
    regions: Sequence[Sequence[int | float]],
    batch_size: int = INFERENCE_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """工作进程内一次打开原图，按有界批次预测并保持候选顺序。"""

    import torch
    from PIL import Image

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    predictions: list[dict[str, Any]] = []
    with Image.open(image_path) as opened:
        source = opened.convert("RGB")
        source_width, source_height = source.size
        for start in range(0, len(regions), batch_size):
            tensors = []
            clipped_boxes: list[Box] = []
            for raw_box in regions[start:start + batch_size]:
                x, y, width, height = _normalize_box(raw_box)
                left = max(0, min(x, source_width))
                top = max(0, min(y, source_height))
                right = max(0, min(x + width, source_width))
                bottom = max(0, min(y + height, source_height))
                if right <= left or bottom <= top:
                    raise ValueError(f"区域框超出图片或为空：{raw_box}")
                clipped_boxes.append(
                    (left, top, right - left, bottom - top)
                )
                tensors.append(
                    _preprocess_worker_image(source.crop((left, top, right, bottom)))
                )

            batch = torch.stack(tensors).to(device)
            with torch.inference_mode():
                logits = model(batch)
                probabilities = torch.softmax(logits, dim=1).cpu()
            for box, probability in zip(clipped_boxes, probabilities):
                class_id = int(probability.argmax().item())
                predictions.append(
                    {
                        "bbox": list(box),
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "confidence": float(probability[class_id].item()),
                        "probabilities": {
                            CLASS_NAMES[index]: float(probability[index].item())
                            for index in range(len(CLASS_NAMES))
                        },
                    }
                )
            del batch, logits, probabilities, tensors
        source.close()
    return predictions


def _worker_main(checkpoint: Path, device_name: str) -> None:
    """以 JSON Lines 协议服务主 PaddleOCR 进程。"""

    try:
        model, device = _build_worker_model(checkpoint, device_name)
        print(
            json.dumps(
                {
                    "ready": True,
                    "device": str(device),
                    "class_names": list(CLASS_NAMES),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return

    for line in __import__("sys").stdin:
        try:
            request = json.loads(line)
            command = request.get("command")
            if command == "shutdown":
                break
            if command != "predict_regions":
                raise ValueError(f"未知工作进程命令：{command}")
            predictions = _worker_predict_regions(
                model,
                device,
                Path(request["image_path"]),
                request["regions"],
            )
            response = {"predictions": predictions}
        except Exception as exc:
            response = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        print(json.dumps(response, ensure_ascii=False), flush=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EfficientNet-B0 工业区域二分类与路由",
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument(
        "--region",
        action="append",
        default=[],
        metavar="X,Y,W,H",
        help="候选框，可重复传入",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--runtime-python",
        type=Path,
        default=DEFAULT_RUNTIME_PYTHON,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _parse_region(text: str) -> Box:
    try:
        return _normalize_box(text.split(","))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"区域格式应为 X,Y,W,H，收到：{text}"
        ) from exc


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.worker:
        _worker_main(args.checkpoint, args.device)
        return
    if args.image is None or not args.region:
        raise SystemExit("请提供 --image 和至少一个 --region X,Y,W,H")
    regions = [_parse_region(region) for region in args.region]
    with CNNRegionClassifier(
        checkpoint_path=args.checkpoint,
        runtime_python=args.runtime_python,
        device=args.device,
    ) as classifier:
        result = classifier.classify_and_route(args.image, regions)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


