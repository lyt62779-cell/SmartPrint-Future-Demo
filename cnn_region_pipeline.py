"""原候选框检测、CNN 分类和后续业务模块之间的轻量适配层。

本文件不调用 OCR，也不调用图片差分。它只完成三件事：

1. 调用原项目候选框检测接口；
2. 把候选框交给独立 EfficientNet-B0 分类器；
3. 为后续 OCR 准备文字 patch，并用图片区域坐标过滤原 PDF 差分结果。

这样 OCR 和差分模块仍保持原实现，只接收经过路由后的输入。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from cnn_region_classifier import CNNRegionClassifier
from region_separator import RegionSeparatorConfig, detect_candidate_regions


def classify_detected_regions(
    image_path: str | Path,
    classifier: CNNRegionClassifier,
    config: RegionSeparatorConfig | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """检测原有候选框并由 CNN 分成文字区域和图片内容区域。"""

    path = Path(image_path)
    candidates = detect_candidate_regions(path, config)
    if not candidates:
        raise RuntimeError(f"原候选区域检测未找到有效区域：{path.name}")
    return classifier.classify_and_route(path, candidates)


def consolidate_document_text_regions(
    predictions: Sequence[dict[str, Any]],
    image_size: tuple[int, int] | None = None,
    containment_threshold: float = 0.82,
    iou_threshold: float = 0.55,
    maximum_panel_area_ratio: float = 0.45,
) -> list[dict[str, Any]]:
    """把 CNN 文字候选整理为段落级区域。

    原候选检测可能同时生成整个文字段落和段落内部的多级小框，CNN 又会
    把它们都判为 ``document_text``。这里按面积从大到小保留区域：小框
    绝大部分已被大框包含时，只保留大框。互不包含的多个文字段仍保留，
    覆盖大半输入图且包含多个小文字框的候选通常是“整版误框”，会先被
    删除；真正的段落大框仍可保留。随后把相邻、同方向、尺寸相近的行框
    聚合成段落。每个完整文字段最终只进行一次 OCR。
    """

    if not 0.0 <= containment_threshold <= 1.0:
        raise ValueError("containment_threshold 必须在 0 到 1 之间")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold 必须在 0 到 1 之间")
    if not 0.0 < maximum_panel_area_ratio <= 1.0:
        raise ValueError("maximum_panel_area_ratio 必须在 0 到 1 之间")

    def coordinates(
        prediction: dict[str, Any],
    ) -> tuple[float, float, float, float]:
        if prediction.get("class_name") != "document_text":
            raise ValueError("文字区域列表包含非 document_text 预测")
        x, y, width, height = (
            float(value) for value in prediction["bbox"]
        )
        if width <= 0 or height <= 0:
            raise ValueError("CNN 文字区域宽高必须大于 0")
        return x, y, width, height

    ordered = sorted(
        (dict(prediction) for prediction in predictions),
        key=lambda prediction: (
            coordinates(prediction)[2] * coordinates(prediction)[3]
        ),
        reverse=True,
    )
    if image_size is not None:
        image_width, image_height = image_size
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image_size 宽高必须大于 0")
        image_area = float(image_width * image_height)
        filtered_ordered: list[dict[str, Any]] = []
        for prediction in ordered:
            x, y, width, height = coordinates(prediction)
            contained_children = 0
            for child in ordered:
                if child is prediction:
                    continue
                cx, cy, child_width, child_height = coordinates(child)
                child_area = child_width * child_height
                intersection_width = max(
                    0.0,
                    min(x + width, cx + child_width) - max(x, cx),
                )
                intersection_height = max(
                    0.0,
                    min(y + height, cy + child_height) - max(y, cy),
                )
                if (
                    intersection_width * intersection_height
                    / max(child_area, 1.0)
                    >= containment_threshold
                ):
                    contained_children += 1
            area_ratio = width * height / image_area
            is_whole_panel = (
                area_ratio >= maximum_panel_area_ratio
                and contained_children >= 2
            )
            if not is_whole_panel:
                filtered_ordered.append(prediction)
        ordered = filtered_ordered
    kept: list[dict[str, Any]] = []
    for prediction in ordered:
        x, y, width, height = coordinates(prediction)
        area = width * height
        is_nested = False
        for accepted in kept:
            ax, ay, accepted_width, accepted_height = coordinates(accepted)
            intersection_width = max(
                0.0,
                min(x + width, ax + accepted_width) - max(x, ax),
            )
            intersection_height = max(
                0.0,
                min(y + height, ay + accepted_height) - max(y, ay),
            )
            intersection = intersection_width * intersection_height
            accepted_area = accepted_width * accepted_height
            union = area + accepted_area - intersection
            containment = intersection / max(area, 1.0)
            iou = intersection / max(union, 1.0)
            if containment >= containment_threshold or iou >= iou_threshold:
                is_nested = True
                break
        if not is_nested:
            kept.append(prediction)
    def should_join(
        first: dict[str, Any],
        second: dict[str, Any],
    ) -> bool:
        """判断两个原始小框是否属于同一行或同一段落。"""

        x1, y1, width1, height1 = coordinates(first)
        x2, y2, width2, height2 = coordinates(second)
        horizontal_overlap = max(
            0.0,
            min(x1 + width1, x2 + width2) - max(x1, x2),
        )
        vertical_overlap = max(
            0.0,
            min(y1 + height1, y2 + height2) - max(y1, y2),
        )
        horizontal_gap = max(
            0.0,
            max(x1, x2) - min(x1 + width1, x2 + width2),
        )
        vertical_gap = max(
            0.0,
            max(y1, y2) - min(y1 + height1, y2 + height2),
        )
        height_ratio = max(height1, height2) / max(
            min(height1, height2),
            1.0,
        )
        width_ratio = max(width1, width2) / max(
            min(width1, width2),
            1.0,
        )

        # 横排段落：相邻行高度相近、横向有明显重叠、行距不大。
        stacked_lines = (
            height_ratio <= 2.5
            and horizontal_overlap / max(min(width1, width2), 1.0) >= 0.20
            and vertical_gap <= max(height1, height2) * 0.90
        )
        # 同一行被拆成多个框，或文字整体旋转 90° 时的对称判断。
        neighboring_fragments = (
            width_ratio <= 2.5
            and vertical_overlap / max(min(height1, height2), 1.0) >= 0.20
            and horizontal_gap <= min(width1, width2) * 0.35
        )
        return stacked_lines or neighboring_fragments

    # 在“原始保留框”之间建立邻接图，再按连通分量求并集。邻接关系只
    # 比较原始框，避免合并框越变越大后把旁边无关版面继续吞进去。
    visited: set[int] = set()
    consolidated: list[dict[str, Any]] = []
    for start_index in range(len(kept)):
        if start_index in visited:
            continue
        stack = [start_index]
        component: list[int] = []
        visited.add(start_index)
        while stack:
            current = stack.pop()
            component.append(current)
            for candidate_index in range(len(kept)):
                if candidate_index in visited:
                    continue
                if should_join(kept[current], kept[candidate_index]):
                    visited.add(candidate_index)
                    stack.append(candidate_index)

        members = [kept[index] for index in component]
        if len(members) == 1:
            consolidated.append(members[0])
            continue
        member_boxes = [coordinates(member) for member in members]
        left = min(box[0] for box in member_boxes)
        top = min(box[1] for box in member_boxes)
        right = max(box[0] + box[2] for box in member_boxes)
        bottom = max(box[1] + box[3] for box in member_boxes)
        merged = dict(max(members, key=lambda item: float(item["confidence"])))
        merged["bbox"] = [
            int(round(left)),
            int(round(top)),
            int(round(right - left)),
            int(round(bottom - top)),
        ]
        merged["confidence"] = max(
            float(member["confidence"]) for member in members
        )
        merged["merged_region_count"] = len(members)
        consolidated.append(merged)

    consolidated.sort(
        key=lambda prediction: (
            coordinates(prediction)[1],
            coordinates(prediction)[0],
        )
    )
    return consolidated


def crop_document_text_patches(
    image_path: str | Path,
    predictions: Sequence[dict[str, Any]],
    output_dir: str | Path,
    prefix: str,
) -> list[dict[str, Any]]:
    """按 CNN 文字框裁剪 patch，供原 OCR 批量识别。

    返回项同时保留 patch 路径和原图 bbox，OCR 完成后调用方可把 patch
    内部坐标平移回完整框选区域坐标。
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    patches: list[dict[str, Any]] = []
    with Image.open(image_path) as opened:
        source = opened.convert("RGB")
        for index, prediction in enumerate(predictions, start=1):
            if prediction.get("class_name") != "document_text":
                raise ValueError("文字 patch 列表包含非 document_text 预测")
            x, y, width, height = (
                int(value) for value in prediction["bbox"]
            )
            patch_path = output / f"{prefix}_text_{index:04d}.png"
            source.crop((x, y, x + width, y + height)).save(patch_path)
            patches.append(
                {
                    "path": patch_path,
                    "bbox": (x, y, width, height),
                    "prediction": dict(prediction),
                }
            )
    return patches


def filter_pdf_difference_boxes(
    image_result: dict[str, Any],
    original_predictions: Sequence[dict[str, Any]],
    comparison_predictions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """只保留落在 CNN 图片内容区域内的原 PDF 差异框。

    ``compare_images`` 仍比较由两份原 PDF 直接渲染出的完整框选区域；CNN
    patch 不参与差分。该函数仅利用 CNN 定位坐标过滤差异框，使文字变化
    交给 OCR，图片内容变化交给原差分模块。

    差分模块会先把比较图缩放到原图尺寸，因此比较侧候选框也按相同的
    ``resize_scale`` 映射到差分坐标。两侧区域取并集，可以保留图片新增或
    缺失时只存在于一侧的差异。
    """

    result = dict(image_result)
    alignment = result.get("alignment_offset", {})
    resize_scale = (
        alignment.get("resize_scale", {})
        if isinstance(alignment, dict)
        else {}
    )
    scale_x = float(resize_scale.get("x", 1.0))
    scale_y = float(resize_scale.get("y", 1.0))

    region_boxes: list[tuple[float, float, float, float]] = []
    for prediction in original_predictions:
        x, y, width, height = prediction["bbox"]
        region_boxes.append((float(x), float(y), float(width), float(height)))
    for prediction in comparison_predictions:
        x, y, width, height = prediction["bbox"]
        region_boxes.append(
            (
                float(x) * scale_x,
                float(y) * scale_y,
                float(width) * scale_x,
                float(height) * scale_y,
            )
        )

    def overlaps_image_region(box: dict[str, Any]) -> bool:
        x = float(box["x"])
        y = float(box["y"])
        width = float(box["width"])
        height = float(box["height"])
        box_area = max(width * height, 1.0)
        box_center = (x + width / 2.0, y + height / 2.0)
        for rx, ry, rw, rh in region_boxes:
            intersection_width = max(
                0.0,
                min(x + width, rx + rw) - max(x, rx),
            )
            intersection_height = max(
                0.0,
                min(y + height, ry + rh) - max(y, ry),
            )
            intersection = intersection_width * intersection_height
            region_area = max(rw * rh, 1.0)
            center_inside = (
                rx <= box_center[0] <= rx + rw
                and ry <= box_center[1] <= ry + rh
            )
            if (
                center_inside
                or intersection / box_area >= 0.10
                or intersection / region_area >= 0.10
            ):
                return True
        return False

    all_boxes = list(result.get("difference_boxes", []))
    filtered_boxes = [box for box in all_boxes if overlaps_image_region(box)]
    result["difference_boxes_before_cnn_filter"] = len(all_boxes)
    result["difference_boxes"] = filtered_boxes
    result["has_difference"] = bool(filtered_boxes)
    result["image_content_region_counts"] = (
        len(original_predictions),
        len(comparison_predictions),
    )
    return result
