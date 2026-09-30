"""
图片自动对齐模块。

算法原理：
1. ORB 在两张灰度图中寻找角点等稳定特征；
2. ORB 为每个特征点生成二进制描述子；
3. 使用汉明距离匹配两张图的描述子；
4. 用 RANSAC 排除错误匹配，并估计“旋转 + 缩放 + 平移”仿射矩阵；
5. 将图片 B 校正到图片 A 的坐标系。

本模块只依赖 OpenCV 和 NumPy，不调用任何 OCR 功能。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _to_gray(image: np.ndarray) -> np.ndarray:
    """将输入图片转换为灰度图，供 ORB 提取特征。"""

    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError("图片必须是灰度图或 BGR/BGRA 彩色图")
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError("不支持的图片通道数")


def _identity_result(
    image_b: np.ndarray,
    reason: str,
    matched_features: int = 0,
    inlier_features: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """特征不足时返回未经仿射校正的图片和明确的失败信息。"""

    return image_b.copy(), {
        "x": 0.0,
        "y": 0.0,
        "rotation_degrees": 0.0,
        "scale": 1.0,
        "matched_features": int(matched_features),
        "inlier_features": int(inlier_features),
        "inlier_ratio": round(
            inlier_features / matched_features,
            4,
        )
        if matched_features
        else 0.0,
        "success": False,
        "reason": reason,
    }


def align_images(
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    max_features: int = 3000,
    keep_match_ratio: float = 0.25,
    min_matches: int = 8,
    ransac_threshold: float = 3.0,
    min_inlier_ratio: float = 0.20,
    max_rotation_degrees: float = 10.0,
    min_scale: float = 0.85,
    max_scale: float = 1.15,
) -> tuple[np.ndarray, dict[str, Any]]:
    """使用 ORB 将图片 B 对齐到图片 A。

    参数：
        image_a: 基准图片。
        image_b: 待校正图片，建议先调整到与 A 相同尺寸。
        max_features: ORB 最多提取的特征点数量。
        keep_match_ratio: 按匹配质量排序后保留的比例。
        min_matches: 估计矩阵所需的最少匹配数量。
        ransac_threshold: RANSAC 判断内点时允许的像素误差。
        min_inlier_ratio: RANSAC 内点占保留匹配的最低比例。
        max_rotation_degrees: 允许自动校正的最大旋转角度。
        min_scale/max_scale: 允许自动校正的缩放范围。

    返回：
        aligned_b: 已映射到 A 坐标系的图片 B。
        alignment_offset: 平移、旋转、缩放和匹配状态。

    alignment_offset 中的 x/y 是“把 B 对齐到 A”所应用的平移量，
    单位为像素；正 x 表示向右移动，正 y 表示向下移动。
    """

    if image_a is None or image_b is None:
        raise ValueError("输入图片不能为空")
    if image_a.size == 0 or image_b.size == 0:
        raise ValueError("输入图片不能是空数组")
    if max_features < 100:
        raise ValueError("max_features 不能小于 100")
    if not 0 < keep_match_ratio <= 1:
        raise ValueError("keep_match_ratio 必须在 0 到 1 之间")
    if min_matches < 3:
        raise ValueError("min_matches 不能小于 3")
    if not 0 < min_inlier_ratio <= 1:
        raise ValueError("min_inlier_ratio 必须在 0 到 1 之间")
    if max_rotation_degrees <= 0:
        raise ValueError("max_rotation_degrees 必须大于 0")
    if not 0 < min_scale <= max_scale:
        raise ValueError("缩放范围参数无效")

    height_a, width_a = image_a.shape[:2]
    if image_b.shape[:2] != (height_a, width_a):
        image_b = cv2.resize(
            image_b,
            (width_a, height_a),
            interpolation=cv2.INTER_AREA,
        )

    gray_a = _to_gray(image_a)
    gray_b = _to_gray(image_b)

    # ORB 对旋转和一定程度的缩放具有鲁棒性，并且不需要深度学习模型。
    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints_a, descriptors_a = orb.detectAndCompute(gray_a, None)
    keypoints_b, descriptors_b = orb.detectAndCompute(gray_b, None)

    if descriptors_a is None or descriptors_b is None:
        return _identity_result(image_b, "图片纹理不足，未检测到可用 ORB 描述子")

    # crossCheck=True 要求 A->B 与 B->A 互相都是最佳匹配，降低误匹配。
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(
        matcher.match(descriptors_b, descriptors_a),
        key=lambda match: match.distance,
    )

    keep_count = min(
        len(matches),
        max(min_matches, int(round(len(matches) * keep_match_ratio))),
    )
    good_matches = matches[:keep_count]
    if len(good_matches) < min_matches:
        return _identity_result(
            image_b,
            f"有效特征匹配不足：{len(good_matches)} < {min_matches}",
            len(good_matches),
        )

    # 源点来自 B，目标点来自 A，因此矩阵方向是 B -> A。
    points_b = np.float32(
        [keypoints_b[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    points_a = np.float32(
        [keypoints_a[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)

    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        points_b,
        points_a,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=3000,
        confidence=0.99,
        refineIters=10,
    )
    if matrix is None:
        return _identity_result(
            image_b,
            "无法根据特征匹配计算稳定的仿射矩阵",
            len(good_matches),
        )

    # 仿射矩阵形式：
    # [a, -b, tx]
    # [b,  a, ty]
    # scale=sqrt(a²+b²)，rotation=atan2(b,a)。
    a = float(matrix[0, 0])
    b = float(matrix[1, 0])
    scale = float(np.sqrt(a * a + b * b))
    rotation_degrees = float(np.degrees(np.arctan2(b, a)))
    offset_x = float(matrix[0, 2])
    offset_y = float(matrix[1, 2])

    inlier_count = (
        int(np.count_nonzero(inlier_mask)) if inlier_mask is not None else 0
    )
    inlier_ratio = inlier_count / len(good_matches)

    # 本模块只解决扫描造成的少量平移、旋转和比例误差。如果两张图是版面
    # 重排、局部拼接或完全不同，ORB 可能从重复图形中得到少量错误匹配。
    # 对异常矩阵主动回退，比使用错误矩阵扭曲整张图片更安全。
    rejection_reasons: list[str] = []
    if inlier_ratio < min_inlier_ratio:
        rejection_reasons.append(
            f"RANSAC 内点比例过低: {inlier_ratio:.1%} < {min_inlier_ratio:.1%}"
        )
    if abs(rotation_degrees) > max_rotation_degrees:
        rejection_reasons.append(
            f"旋转角度异常: {rotation_degrees:.2f}°"
        )
    if not min_scale <= scale <= max_scale:
        rejection_reasons.append(f"缩放比例异常: {scale:.4f}")
    if rejection_reasons:
        return _identity_result(
            image_b,
            "；".join(rejection_reasons),
            len(good_matches),
            inlier_count,
        )

    aligned_b = cv2.warpAffine(
        image_b,
        matrix,
        (width_a, height_a),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    alignment_offset: dict[str, Any] = {
        "x": round(offset_x, 3),
        "y": round(offset_y, 3),
        "rotation_degrees": round(rotation_degrees, 4),
        "scale": round(scale, 6),
        "matched_features": len(good_matches),
        "inlier_features": inlier_count,
        "inlier_ratio": round(inlier_ratio, 4),
        "success": True,
        "reason": "",
    }
    return aligned_b, alignment_offset
