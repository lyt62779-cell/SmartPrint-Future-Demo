"""
图片结构相似度与综合评分模块。

SSIM（结构相似度）同时比较局部亮度、对比度和结构。与单纯像素差相比，
它对轻微亮度变化和压缩噪声更宽容，更适合判断两张图的整体结构是否一致。

本模块使用 OpenCV + NumPy 直接实现 SSIM，不依赖 scikit-image。
"""

from __future__ import annotations

import cv2
import numpy as np


def calculate_ssim_map(
    image_a: np.ndarray,
    image_b: np.ndarray,
) -> np.ndarray:
    """计算逐像素 SSIM 图。

    返回图中每个像素都带有一个局部结构相似度。后续既可以对整张图求平均
    得到全局 SSIM，也可以按网格统计得到局部 SSIM。
    """

    if image_a is None or image_b is None:
        raise ValueError("SSIM 输入图片不能为空")
    if image_a.shape != image_b.shape:
        raise ValueError("计算 SSIM 前，两张图片必须具有相同尺寸")
    if image_a.ndim != 2 or image_b.ndim != 2:
        raise ValueError("SSIM 输入必须是灰度图")

    array_a = image_a.astype(np.float64)
    array_b = image_b.astype(np.float64)

    # 8 位图片动态范围 L=255，K1=0.01、K2=0.03 是经典 SSIM 参数。
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mean_a = cv2.GaussianBlur(array_a, (11, 11), 1.5)
    mean_b = cv2.GaussianBlur(array_b, (11, 11), 1.5)

    mean_a_sq = mean_a * mean_a
    mean_b_sq = mean_b * mean_b
    mean_ab = mean_a * mean_b

    variance_a = cv2.GaussianBlur(array_a * array_a, (11, 11), 1.5) - mean_a_sq
    variance_b = cv2.GaussianBlur(array_b * array_b, (11, 11), 1.5) - mean_b_sq
    covariance_ab = cv2.GaussianBlur(array_a * array_b, (11, 11), 1.5) - mean_ab

    numerator = (2 * mean_ab + c1) * (2 * covariance_ab + c2)
    denominator = (mean_a_sq + mean_b_sq + c1) * (
        variance_a + variance_b + c2
    )

    # 理论上分母大于 0；maximum 是为数值计算增加额外保护。
    return numerator / np.maximum(
        denominator,
        np.finfo(np.float64).eps,
    )


def calculate_ssim(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """计算整张图片的全局 SSIM。

    工业图片正常情况下通常位于 0 到 1：
        1 表示结构完全一致；
        越接近 0 表示整体结构差异越大。
    """

    return round(float(np.mean(calculate_ssim_map(image_a, image_b))), 8)


def calculate_local_ssim(
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    grid_rows: int = 4,
    grid_columns: int = 4,
    worst_fraction: float = 0.25,
    anomaly_threshold: float = 0.95,
) -> tuple[float, list[dict[str, int | float | str]]]:
    """分块计算局部 SSIM，并定位结构异常网格。

    图片默认划分为 4×4 网格，每个区域独立计算平均 SSIM。local_ssim
    取最差 25% 网格的平均值，而不是全部网格平均，因此小范围缺陷不会被
    大量正常背景稀释。

    返回：
        local_ssim: 最差若干网格的平均结构相似度。
        anomaly_regions: 低于 anomaly_threshold 的网格坐标和严重等级。
    """

    if grid_rows <= 0 or grid_columns <= 0:
        raise ValueError("局部 SSIM 网格行列数必须大于 0")
    if not 0 < worst_fraction <= 1:
        raise ValueError("worst_fraction 必须在 0 到 1 之间")
    if not -1 <= anomaly_threshold <= 1:
        raise ValueError("anomaly_threshold 必须在 -1 到 1 之间")

    ssim_map = calculate_ssim_map(image_a, image_b)
    height, width = ssim_map.shape
    row_edges = np.linspace(0, height, grid_rows + 1, dtype=int)
    column_edges = np.linspace(0, width, grid_columns + 1, dtype=int)

    region_scores: list[float] = []
    anomaly_regions: list[dict[str, int | float | str]] = []
    for row in range(grid_rows):
        for column in range(grid_columns):
            top, bottom = row_edges[row], row_edges[row + 1]
            left, right = column_edges[column], column_edges[column + 1]
            if bottom <= top or right <= left:
                continue

            region_score = float(np.mean(ssim_map[top:bottom, left:right]))
            region_score = float(np.clip(region_score, -1.0, 1.0))
            region_scores.append(region_score)
            if region_score >= anomaly_threshold:
                continue

            if region_score < 0.50:
                severity = "critical"
            elif region_score < 0.75:
                severity = "high"
            elif region_score < 0.90:
                severity = "medium"
            else:
                severity = "low"

            anomaly_regions.append(
                {
                    "x": int(left),
                    "y": int(top),
                    "width": int(right - left),
                    "height": int(bottom - top),
                    "ssim": round(region_score, 8),
                    "severity_level": severity,
                }
            )

    if not region_scores:
        return 1.0, []

    worst_count = max(
        1,
        int(np.ceil(len(region_scores) * worst_fraction)),
    )
    local_score = float(np.mean(sorted(region_scores)[:worst_count]))
    anomaly_regions.sort(key=lambda region: float(region["ssim"]))
    return round(local_score, 8), anomaly_regions


def calculate_box_area_rate(
    difference_boxes: list[dict[str, int | float]],
    image_shape: tuple[int, ...],
) -> float:
    """计算所有变化框覆盖面积占图片面积的比例。

    使用掩码合并重叠框，避免多个相交矩形被重复计算。
    """

    height, width = image_shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("图片尺寸无效")

    box_mask = np.zeros((height, width), dtype=np.uint8)
    for box in difference_boxes:
        x = max(0, int(box["x"]))
        y = max(0, int(box["y"]))
        right = min(width, x + max(0, int(box["width"])))
        bottom = min(height, y + max(0, int(box["height"])))
        if right > x and bottom > y:
            box_mask[y:bottom, x:right] = 1

    return float(np.count_nonzero(box_mask) / box_mask.size)


def calculate_image_score(
    global_ssim: float,
    local_ssim: float,
    difference_rate: float,
    difference_boxes: list[dict[str, int | float]],
    image_shape: tuple[int, ...],
    severity_level: str,
) -> float:
    """融合全局/局部 SSIM、差异面积和严重程度，生成 0~100 评分。

    权重设计：
        30% 全局 SSIM：反映整体结构一致性；
        30% 局部 SSIM：提高对局部缺陷的敏感度；
        25% 差异面积：融合真实差异像素与变化框覆盖范围；
        15% 区域严重程度：重大缺失或替换会进一步降低评分。

    这些权重是清晰可调的工程初始值，后续可使用真实工业样本标定。
    """

    global_score = float(np.clip(global_ssim, 0.0, 1.0))
    local_score = float(np.clip(local_ssim, 0.0, 1.0))
    box_area_rate = calculate_box_area_rate(difference_boxes, image_shape)
    combined_area_rate = (
        0.70 * float(np.clip(difference_rate, 0.0, 1.0))
        + 0.30 * float(np.clip(box_area_rate, 0.0, 1.0))
    )
    area_score = 1.0 - combined_area_rate
    severity_scores = {
        "none": 1.0,
        "low": 0.90,
        "medium": 0.70,
        "high": 0.40,
        "critical": 0.0,
    }
    if severity_level not in severity_scores:
        raise ValueError(f"未知严重等级: {severity_level}")

    score = 100.0 * (
        0.30 * global_score
        + 0.30 * local_score
        + 0.25 * area_score
        + 0.15 * severity_scores[severity_level]
    )
    return round(float(np.clip(score, 0.0, 100.0)), 2)
