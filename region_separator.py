"""
基于 OpenCV 的文字区域 / 非文字图片区域分离模块。

本模块只做区域分类，不调用 OCR，也不调用图片差分。处理流程：

    输入图片
        -> 灰度化
        -> Otsu 二值化（同时考虑深色字和浅色字）
        -> MSER 稳定区域检测
        -> 连通域分析
        -> 相邻区域合并
        -> 根据面积、宽高比、边缘密度等特征分类

公开接口 `separate_regions()` 返回：

    {
        "text_regions": [[x, y, w, h], ...],
        "image_regions": [[x, y, w, h], ...],
    }

第一阶段采用传统图像算法，适合后续分别把 text_regions 交给 OCR，
把 image_regions 交给图片检测模块，但本文件本身不会调用它们。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np


ImageInput: TypeAlias = str | Path | np.ndarray
Box: TypeAlias = tuple[int, int, int, int]


@dataclass(frozen=True)
class RegionSeparatorConfig:
    """区域分离参数。

    参数都按图片尺寸归一化或给出较保守的默认值，避免只适用于某一种
    固定分辨率。工业稿件差异较大时，可以在调用处传入自定义配置。
    """

    # MSER 检测的最小/最大区域面积，占整图面积的比例。
    mser_min_area_ratio: float = 0.00001
    mser_max_area_ratio: float = 0.08

    # 连通域太小通常是扫描噪声；太大通常是整块背景。
    component_min_area_ratio: float = 0.000005
    component_max_area_ratio: float = 0.65

    # 初始字符候选的最大面积和最大高度。
    character_max_area_ratio: float = 0.04
    character_max_height_ratio: float = 0.12

    # 合并同一文字行时，允许的水平间距 = 字符高度 * 此系数。
    text_horizontal_gap_factor: float = 2.2
    text_vertical_overlap_ratio: float = 0.30

    # 最终图片区域必须达到的最小面积。
    image_min_area_ratio: float = 0.001
    image_max_area_ratio: float = 0.45

    # 边缘密度 = Canny 边缘像素数 / 区域像素数。
    text_min_edge_density: float = 0.015
    image_min_edge_density: float = 0.008


def load_image(image: ImageInput) -> np.ndarray:
    """读取 BGR 图片并检查有效性。

    使用 ``np.fromfile + cv2.imdecode``，可正常读取 Windows 中文路径。
    NumPy 数组输入会复制一份，避免本模块意外修改调用方的原图。
    """

    if isinstance(image, np.ndarray):
        loaded = image.copy()
    else:
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"图片不存在：{path}")
        encoded = np.fromfile(path, dtype=np.uint8)
        loaded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if loaded is None or loaded.size == 0:
        raise ValueError("无法读取有效图片")
    if loaded.ndim == 2:
        return cv2.cvtColor(loaded, cv2.COLOR_GRAY2BGR)
    if loaded.ndim == 3 and loaded.shape[2] == 4:
        return cv2.cvtColor(loaded, cv2.COLOR_BGRA2BGR)
    if loaded.ndim != 3 or loaded.shape[2] != 3:
        raise ValueError(f"不支持的图片形状：{loaded.shape}")
    return loaded


def preprocess_image(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """灰度化并生成深色前景、浅色前景两种二值图。

    包装稿既可能是白底黑字，也可能是深色底白字。只使用一种二值极性
    会漏掉其中一类文字，因此对 Otsu 结果同时保留正向和反向二值图。
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 彩色包装稿中，整块红色/蓝色背景在全局 Otsu 阈值下容易和文字粘成
    # 一个大区域。自适应阈值依据局部邻域计算门槛，能把红底白字、浅底
    # 黑字分别提取出来。
    minimum_side = min(gray.shape)
    block_size = max(15, round(minimum_side / 30))
    if block_size % 2 == 0:
        block_size += 1
    dark_foreground = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        7,
    )
    light_foreground = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        7,
    )
    return gray, dark_foreground, light_foreground


def _clip_box(box: Box, width: int, height: int) -> Box | None:
    x, y, w, h = box
    x1 = max(0, min(int(x), width))
    y1 = max(0, min(int(y), height))
    x2 = max(0, min(int(x + w), width))
    y2 = max(0, min(int(y + h), height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def _intersection_area(first: Box, second: Box) -> int:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    return max(0, x2 - x1) * max(0, y2 - y1)


def _iou(first: Box, second: Box) -> float:
    intersection = _intersection_area(first, second)
    if intersection == 0:
        return 0.0
    first_area = first[2] * first[3]
    second_area = second[2] * second[3]
    return intersection / max(first_area + second_area - intersection, 1)


def _union_box(first: Box, second: Box) -> Box:
    x1 = min(first[0], second[0])
    y1 = min(first[1], second[1])
    x2 = max(first[0] + first[2], second[0] + second[2])
    y2 = max(first[1] + first[3], second[1] + second[3])
    return x1, y1, x2 - x1, y2 - y1


def _deduplicate_boxes(boxes: list[Box]) -> list[Box]:
    """删除 MSER 常见的多层嵌套框，保留覆盖更完整的区域。"""

    kept: list[Box] = []
    for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        area = box[2] * box[3]
        duplicate = False
        for accepted in kept:
            intersection = _intersection_area(box, accepted)
            containment = intersection / max(area, 1)
            if _iou(box, accepted) >= 0.72 or containment >= 0.88:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


def detect_mser_boxes(
    gray: np.ndarray,
    config: RegionSeparatorConfig,
) -> list[Box]:
    """使用 MSER 检测灰度稳定区域。

    MSER 会寻找在多个灰度阈值下仍保持形状稳定的区域。字符内部通常具有
    稳定的亮暗结构，因此它适合在复杂包装背景中生成文字候选框。
    """

    height, width = gray.shape
    image_area = width * height
    minimum_area = max(3, round(image_area * config.mser_min_area_ratio))
    maximum_area = max(
        minimum_area + 1,
        round(image_area * config.mser_max_area_ratio),
    )

    boxes: list[Box] = []
    for source in (gray, cv2.bitwise_not(gray)):
        detector = cv2.MSER_create()
        detector.setMinArea(minimum_area)
        detector.setMaxArea(maximum_area)
        regions, _boxes = detector.detectRegions(source)
        for points in regions:
            x, y, w, h = cv2.boundingRect(points.reshape(-1, 1, 2))
            clipped = _clip_box((x, y, w, h), width, height)
            if clipped is not None:
                boxes.append(clipped)
    return _deduplicate_boxes(boxes)


def detect_connected_component_boxes(
    binary_images: tuple[np.ndarray, np.ndarray],
    config: RegionSeparatorConfig,
) -> list[Box]:
    """对两种极性的二值图执行连通域分析。

    连通域把相邻前景像素归为一个区域，可补充 MSER 漏掉的字符笔画、
    图标边缘和大面积图形区域。
    """

    height, width = binary_images[0].shape
    image_area = width * height
    minimum_area = max(
        2,
        round(image_area * config.component_min_area_ratio),
    )
    maximum_area = round(image_area * config.component_max_area_ratio)
    boxes: list[Box] = []

    for binary in binary_images:
        polarity_boxes: list[Box] = []
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if not minimum_area <= int(area) <= maximum_area:
                continue
            clipped = _clip_box(
                (int(x), int(y), int(w), int(h)),
                width,
                height,
            )
            if clipped is not None:
                polarity_boxes.append(clipped)

        # 连通域在像素层面互不重叠，即使两个不规则区域的外接矩形互相
        # 包含，也不能按矩形包含关系删除，否则大色块会吞掉内部文字。
        boxes.extend(polarity_boxes)
    return boxes


def _edge_density(edges: np.ndarray, box: Box) -> float:
    x, y, w, h = box
    region = edges[y : y + h, x : x + w]
    return float(cv2.countNonZero(region) / max(region.size, 1))


def _foreground_balance(binary: np.ndarray, box: Box) -> float:
    """返回前景比例与背景比例中的较小值。

    文字区域通常同时包含笔画和背景；纯色块的该值接近 0。取较小值后，
    深色字和浅色字可使用相同阈值。
    """

    x, y, w, h = box
    region = binary[y : y + h, x : x + w]
    foreground_ratio = float(cv2.countNonZero(region) / max(region.size, 1))
    return min(foreground_ratio, 1.0 - foreground_ratio)


def _is_character_candidate(
    box: Box,
    image_shape: tuple[int, int],
    edges: np.ndarray,
    binary: np.ndarray,
    config: RegionSeparatorConfig,
) -> bool:
    """根据面积、宽高比、边缘密度筛选初始字符候选。"""

    image_height, image_width = image_shape
    x, y, width, height = box
    del x, y
    area_ratio = width * height / (image_width * image_height)
    aspect_ratio = width / max(height, 1)
    edge_density = _edge_density(edges, box)
    balance = _foreground_balance(binary, box)
    gray_region = edges[
        box[1] : box[1] + box[3],
        box[0] : box[0] + box[2],
    ]
    local_edge_variation = float(np.std(gray_region))
    return (
        2 <= width
        and 3 <= height <= image_height * config.character_max_height_ratio
        and area_ratio <= config.character_max_area_ratio
        and 0.08 <= aspect_ratio <= 12.0
        and edge_density >= config.text_min_edge_density
        and (balance >= 0.008 or local_edge_variation >= 20.0)
    )


def _should_merge_text(
    first: Box,
    second: Box,
    config: RegionSeparatorConfig,
) -> bool:
    """判断两个字符框是否可能属于同一文字行。"""

    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    vertical_overlap = max(
        0,
        min(ay + ah, by + bh) - max(ay, by),
    )
    overlap_ratio = vertical_overlap / max(min(ah, bh), 1)
    horizontal_gap = max(0, max(ax, bx) - min(ax + aw, bx + bw))
    maximum_gap = max(ah, bh) * config.text_horizontal_gap_factor
    center_difference = abs((ay + ah / 2) - (by + bh / 2))
    same_line = (
        overlap_ratio >= config.text_vertical_overlap_ratio
        and center_difference <= max(ah, bh) * 0.8
    )
    if same_line and horizontal_gap <= maximum_gap:
        return True

    # 包装展开图中常有旋转 90° 的侧面文字。对称地检查同一文字列，
    # 让竖排字符也能合并成一个文字区域。
    horizontal_overlap = max(
        0,
        min(ax + aw, bx + bw) - max(ax, bx),
    )
    horizontal_overlap_ratio = horizontal_overlap / max(min(aw, bw), 1)
    vertical_gap = max(0, max(ay, by) - min(ay + ah, by + bh))
    center_x_difference = abs((ax + aw / 2) - (bx + bw / 2))
    same_column = (
        horizontal_overlap_ratio >= config.text_vertical_overlap_ratio
        and center_x_difference <= max(aw, bw) * 0.8
    )
    return (
        same_column
        and vertical_gap <= max(aw, bw) * config.text_horizontal_gap_factor
    )


def merge_text_boxes(
    boxes: list[Box],
    config: RegionSeparatorConfig,
) -> list[Box]:
    """按水平文字行和竖直文字列分别做受约束的贪心合并。

    图像内部的小边缘会形成复杂的连通网络。若单纯使用传递闭包，只要
    中间存在一串相邻图形，最终就可能把整页串成一个“文字框”。这里
    限制合并后区域在短轴方向不能明显变厚，从结构上阻断这种传播。
    """

    if not boxes:
        return []

    def axis_gap(first: Box, second: Box, horizontal: bool) -> int:
        if horizontal:
            return max(
                0,
                max(first[0], second[0])
                - min(first[0] + first[2], second[0] + second[2]),
            )
        return max(
            0,
            max(first[1], second[1])
            - min(first[1] + first[3], second[1] + second[3]),
        )

    def can_join(group: Box, item: Box, horizontal: bool) -> bool:
        if horizontal:
            overlap = max(
                0,
                min(group[1] + group[3], item[1] + item[3])
                - max(group[1], item[1]),
            )
            overlap_ratio = overlap / max(min(group[3], item[3]), 1)
            gap = axis_gap(group, item, True)
            unit = max(3, min(group[3], item[3]))
            combined = _union_box(group, item)
            short_axis_stable = combined[3] <= max(group[3], item[3]) * 1.6
        else:
            overlap = max(
                0,
                min(group[0] + group[2], item[0] + item[2])
                - max(group[0], item[0]),
            )
            overlap_ratio = overlap / max(min(group[2], item[2]), 1)
            gap = axis_gap(group, item, False)
            unit = max(3, min(group[2], item[2]))
            combined = _union_box(group, item)
            short_axis_stable = combined[2] <= max(group[2], item[2]) * 1.6
        return (
            overlap_ratio >= config.text_vertical_overlap_ratio
            and gap <= unit * config.text_horizontal_gap_factor
            and short_axis_stable
        )

    def group_axis(horizontal: bool) -> list[Box]:
        ordered = sorted(
            boxes,
            key=(
                (lambda box: (box[0], box[1]))
                if horizontal
                else (lambda box: (box[1], box[0]))
            ),
        )
        groups: list[tuple[Box, int]] = []
        for item in ordered:
            best_index: int | None = None
            best_gap: int | None = None
            for index, (group, _count) in enumerate(groups):
                if not can_join(group, item, horizontal):
                    continue
                gap = axis_gap(group, item, horizontal)
                if best_gap is None or gap < best_gap:
                    best_index = index
                    best_gap = gap
            if best_index is None:
                groups.append((item, 1))
            else:
                group, count = groups[best_index]
                groups[best_index] = (_union_box(group, item), count + 1)

        # 至少两个字符级证据才能组成文字行，减少照片纹理误报。
        return [group for group, count in groups if count >= 2]

    # 暂不在水平组与竖直组之间做“包含即去重”。一个错误的竖直大框
    # 可能包含正确的水平文字行；下一阶段会先按区域特征过滤，再去重。
    return group_axis(True) + group_axis(False)


def _classify_text_regions(
    boxes: list[Box],
    gray: np.ndarray,
    binary: np.ndarray,
    config: RegionSeparatorConfig,
) -> list[Box]:
    """再次使用区域特征过滤合并后的文字行。"""

    edges = cv2.Canny(gray, 60, 160)
    image_height, image_width = gray.shape
    image_area = image_width * image_height
    result: list[Box] = []
    for box in boxes:
        _x, _y, width, height = box
        area_ratio = width * height / image_area
        aspect_ratio = width / max(height, 1)
        edge_density = _edge_density(edges, box)
        balance = _foreground_balance(binary, box)

        # 文字行可以很宽，但一般不会覆盖整张图的大部分面积；笔画还会
        # 形成一定边缘密度，并在区域内同时保留前景与背景。
        shape_is_text_like = (
            aspect_ratio >= 2.5
            or aspect_ratio <= 0.5
            or area_ratio <= 0.005
        )
        if (
            area_ratio <= 0.02
            and 0.08 <= aspect_ratio <= 60.0
            and height <= image_height * 0.30
            and edge_density >= config.text_min_edge_density
            and balance >= 0.008
            and shape_is_text_like
        ):
            result.append(box)
    return _deduplicate_boxes(result)


def _merge_image_boxes(
    boxes: list[Box],
    image_shape: tuple[int, int],
) -> list[Box]:
    """合并重叠或距离很近的非文字图形区域。"""

    image_height, image_width = image_shape
    gap_x = max(3, round(image_width * 0.008))
    gap_y = max(3, round(image_height * 0.008))
    merged = [
        box
        for box in boxes
        if (
            box[2] * box[3]
            / max(image_width * image_height, 1)
            <= 0.45
        )
    ]
    changed = True
    while changed:
        changed = False
        output: list[Box] = []
        while merged:
            current = merged.pop()
            index = 0
            while index < len(merged):
                other = merged[index]
                expanded = (
                    current[0] - gap_x,
                    current[1] - gap_y,
                    current[2] + gap_x * 2,
                    current[3] + gap_y * 2,
                )
                if _intersection_area(expanded, other) > 0:
                    combined = _union_box(current, other)
                    combined_ratio = (
                        combined[2]
                        * combined[3]
                        / max(image_width * image_height, 1)
                    )
                    if combined_ratio <= 0.35:
                        current = combined
                        merged.pop(index)
                        changed = True
                    else:
                        index += 1
                else:
                    index += 1
            output.append(current)
        merged = output
    return _deduplicate_boxes(merged)


def detect_image_regions(
    gray: np.ndarray,
    binary_images: tuple[np.ndarray, np.ndarray],
    component_boxes: list[Box],
    text_regions: list[Box],
    config: RegionSeparatorConfig,
) -> list[Box]:
    """在排除文字边缘后寻找非文字图片区域。

    先把已识别的文字框从边缘图中遮掉，再通过形态学闭运算连接图片内部
    的相邻边缘。最后结合大连通域候选，并用面积、宽高比、边缘密度过滤。
    """

    image_height, image_width = gray.shape
    image_area = image_width * image_height
    edges = cv2.Canny(gray, 60, 160)
    original_edges = edges.copy()

    text_mask = np.zeros_like(gray)
    padding = max(2, round(min(image_width, image_height) * 0.003))
    for x, y, width, height in text_regions:
        cv2.rectangle(
            text_mask,
            (max(0, x - padding), max(0, y - padding)),
            (
                min(image_width - 1, x + width + padding),
                min(image_height - 1, y + height + padding),
            ),
            255,
            -1,
        )
    edges[text_mask > 0] = 0

    kernel_width = max(3, round(image_width * 0.012))
    kernel_height = max(3, round(image_height * 0.012))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_width, kernel_height),
    )
    connected_edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )
    connected_edges = cv2.dilate(
        connected_edges,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    contours, _hierarchy = cv2.findContours(
        connected_edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates: list[Box] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        candidates.append((x, y, width, height))

    # 大连通域常对应照片、色块、图标或插图，作为边缘轮廓的补充。
    candidates.extend(
        box
        for box in component_boxes
        if box[2] * box[3] / image_area >= config.image_min_area_ratio
    )
    candidates = [
        box
        for box in candidates
        if (
            config.image_min_area_ratio
            <= box[2] * box[3] / image_area
            <= config.image_max_area_ratio
        )
    ]
    candidates = _merge_image_boxes(
        candidates,
        gray.shape,
    )

    result: list[Box] = []
    for box in candidates:
        _x, _y, width, height = box
        area_ratio = width * height / image_area
        aspect_ratio = width / max(height, 1)
        edge_density = _edge_density(original_edges, box)

        # 如果候选几乎完全落在一个文字框内部，它仍是文字结构，不应再
        # 作为图片区域重复输出。
        contained_by_text = any(
            _intersection_area(box, text_box)
            / max(width * height, 1)
            >= 0.65
            for text_box in text_regions
        )
        if (
            not contained_by_text
            and area_ratio >= config.image_min_area_ratio
            and area_ratio <= config.image_max_area_ratio
            and 0.05 <= aspect_ratio <= 20.0
            and (
                edge_density >= config.image_min_edge_density
                or area_ratio >= 0.02
            )
        ):
            result.append(box)
    return _deduplicate_boxes(result)


def detect_candidate_regions(
    image: ImageInput,
    config: RegionSeparatorConfig | None = None,
) -> list[list[int]]:
    """复用原检测链路，只输出尚未分类的候选区域。

    该接口保留原有 MSER、连通域和区域合并算法，但不使用
    ``_classify_text_regions`` 给候选框贴文字/图片标签。文字行形状候选和
    图形轮廓候选会被合并到同一列表，最终类别统一交给外部 CNN 判断。

    返回坐标格式固定为 ``[x, y, width, height]``。不同候选框可以存在包含
    关系，例如产品图片内部仍可能包含一个独立文字行；这里不能因为大框
    包含小框就删除小框，否则 CNN 将没有机会分别判断它们。
    """

    settings = config or RegionSeparatorConfig()
    source = load_image(image)
    gray, dark_foreground, light_foreground = preprocess_image(source)
    edges = cv2.Canny(gray, 60, 160)

    mser_boxes = detect_mser_boxes(gray, settings)
    component_boxes = detect_connected_component_boxes(
        (dark_foreground, light_foreground),
        settings,
    )
    character_candidates = _deduplicate_boxes(
        [
            box
            for box in mser_boxes + component_boxes
            if _is_character_candidate(
                box,
                gray.shape,
                edges,
                dark_foreground,
                settings,
            )
        ]
    )
    line_candidates = merge_text_boxes(character_candidates, settings)

    # 传入空的 text_regions，意味着只复用原图形轮廓候选生成步骤，不根据
    # 旧的文字分类结果遮挡或排除任何区域。它们和文字行候选最终都由 CNN
    # 统一分类。
    graphic_candidates = detect_image_regions(
        gray,
        (dark_foreground, light_foreground),
        component_boxes,
        [],
        settings,
    )

    candidates: list[Box] = []
    for box in sorted(
        line_candidates + graphic_candidates,
        key=lambda item: (item[1], item[0], item[2] * item[3]),
    ):
        # 只去除高度重合的重复候选，保留大框包含小框的情况。
        if any(_iou(box, accepted) >= 0.88 for accepted in candidates):
            continue
        candidates.append(box)
    return [list(box) for box in candidates]


def separate_regions(
    image: ImageInput,
    config: RegionSeparatorConfig | None = None,
) -> dict[str, list[list[int]]]:
    """把输入图片划分为文字区域和非文字图片区域。

    返回坐标全部采用原图像素坐标，格式为 ``[x, y, width, height]``。
    """

    settings = config or RegionSeparatorConfig()
    source = load_image(image)
    gray, dark_foreground, light_foreground = preprocess_image(source)
    edges = cv2.Canny(gray, 60, 160)

    mser_boxes = detect_mser_boxes(gray, settings)
    component_boxes = detect_connected_component_boxes(
        (dark_foreground, light_foreground),
        settings,
    )

    # MSER 和连通域各有优缺点：前者擅长稳定字符结构，后者擅长完整笔画。
    # 合并两类候选后，先做字符级过滤，再合并成文字行。
    # 必须先按字符特征过滤，再去除重复框。浅色前景二值图可能产生一个
    # 覆盖整页的背景连通域；若先做嵌套框去重，它会错误吞掉内部所有
    # MSER 字符候选。
    character_candidates = _deduplicate_boxes(
        [
            box
            for box in mser_boxes + component_boxes
            if _is_character_candidate(
                box,
                gray.shape,
                edges,
                dark_foreground,
                settings,
            )
        ]
    )
    merged_text = merge_text_boxes(character_candidates, settings)
    text_regions = _classify_text_regions(
        merged_text,
        gray,
        dark_foreground,
        settings,
    )
    image_regions = detect_image_regions(
        gray,
        (dark_foreground, light_foreground),
        component_boxes,
        text_regions,
        settings,
    )

    text_regions.sort(key=lambda box: (box[1], box[0]))
    image_regions.sort(key=lambda box: (box[1], box[0]))
    return {
        "text_regions": [list(box) for box in text_regions],
        "image_regions": [list(box) for box in image_regions],
    }


def visualize_regions(
    image: ImageInput,
    regions: dict[str, list[list[int]]],
) -> np.ndarray:
    """生成调试预览：绿色框为文字，蓝色框为非文字图片。"""

    output = load_image(image)
    for index, (x, y, width, height) in enumerate(
        regions["text_regions"],
        start=1,
    ):
        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            (0, 200, 0),
            2,
        )
        cv2.putText(
            output,
            f"T{index}",
            (x, max(15, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 160, 0),
            1,
            cv2.LINE_AA,
        )
    for index, (x, y, width, height) in enumerate(
        regions["image_regions"],
        start=1,
    ):
        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            (255, 80, 0),
            2,
        )
        cv2.putText(
            output,
            f"I{index}",
            (x, max(15, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 80, 0),
            1,
            cv2.LINE_AA,
        )
    return output


def save_image(path: str | Path, image: np.ndarray) -> None:
    """保存图片，兼容 Windows 中文路径。"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    extension = output.suffix or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise ValueError(f"无法编码输出图片：{output}")
    encoded.tofile(output)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 OpenCV 分离文字区域和非文字图片区域",
    )
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument(
        "--visual",
        help="可选：保存带文字框/图片框的调试预览",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    result = separate_regions(args.image)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.visual:
        save_image(
            args.visual,
            visualize_regions(args.image, result),
        )


if __name__ == "__main__":
    main()
