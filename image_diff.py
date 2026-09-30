"""
工业图片差异检测主模块。

统一处理流程：
读取并检查图片
-> BGR 格式和尺寸统一
-> ORB 自动对齐
-> 灰度化与高斯滤波
-> absdiff 像素差分
-> 阈值与形态学去噪
-> 轮廓定位
-> SSIM 与综合评分
-> 红框可视化并保存。

本模块完全独立，不导入 PaddleOCR，也不调用任何 OCR 代码。未来 OCR 主程序
只需导入 compare_images(image_a, image_b)，即可获得结构化图片检测结果。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from image_align import align_images
from image_similarity import (
    calculate_image_score,
    calculate_local_ssim,
    calculate_ssim,
)

ImageInput = str | Path | np.ndarray


def read_image(image_path: str | Path) -> np.ndarray:
    """读取图片并检查有效性，兼容 Windows 中文路径。

    cv2.imread 在部分 Windows 环境中无法正确处理中文路径，所以先用
    NumPy 读取文件字节，再使用 cv2.imdecode 解码为 BGR 图片。
    """

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在: {path}")

    image_bytes = np.fromfile(str(path), dtype=np.uint8)
    if image_bytes.size == 0:
        raise ValueError(f"图片文件为空: {path}")

    image = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise ValueError(f"无法解码图片，请检查图片是否有效: {path}")
    return normalize_bgr(image)


def normalize_bgr(image: np.ndarray) -> np.ndarray:
    """将灰度、BGR 或 BGRA 图片统一为 OpenCV 使用的三通道 BGR。

    说明：
        OpenCV 默认颜色顺序是 BGR。本接口接收的 NumPy 彩色数组也按
        OpenCV 约定视为 BGR；文件输入会由 OpenCV 解码，因此不会出现
        PIL 常见的 RGB/BGR 颠倒问题。
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("图片必须是路径或 NumPy 数组")
    if image.size == 0:
        raise ValueError("图片数组不能为空")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError("图片数组维度无效")
    if image.shape[2] == 3:
        return image.copy()
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    raise ValueError(f"不支持的图片通道数: {image.shape[2]}")


def load_image(image: ImageInput) -> np.ndarray:
    """统一接收图片路径或 NumPy 图片，方便未来从其他模块直接调用。"""

    if isinstance(image, np.ndarray):
        return normalize_bgr(image)
    if isinstance(image, (str, Path)):
        return read_image(image)
    raise TypeError("image_a/image_b 必须是路径、Path 或 NumPy 数组")


def save_image(output_path: str | Path, image: np.ndarray) -> None:
    """保存图片并兼容 Windows 中文路径。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lower() or ".jpg"

    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise ValueError(f"无法编码输出图片，格式可能不受支持: {extension}")
    encoded.tofile(str(path))


def validate_odd_kernel(value: int, name: str) -> int:
    """检查高斯核或形态学核是否为正奇数。"""

    if value <= 0 or value % 2 == 0:
        raise ValueError(f"{name} 必须是大于 0 的奇数，例如 3、5、7")
    return value


def resize_to_reference(
    image_a: np.ndarray,
    image_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """以图片 A 为基准自动调整 B 的尺寸。

    尺寸统一是像素比较和特征对齐的前提。返回 resize_scale 便于了解
    图片 B 在进入 ORB 对齐前经历的宽高缩放比例。
    """

    height_a, width_a = image_a.shape[:2]
    height_b, width_b = image_b.shape[:2]
    if min(height_a, width_a, height_b, width_b) <= 0:
        raise ValueError("图片宽高必须大于 0")

    scale_info = {
        "x": round(width_a / width_b, 6),
        "y": round(height_a / height_b, 6),
    }
    if (height_a, width_a) == (height_b, width_b):
        return image_a, image_b, scale_info

    # 缩小时 INTER_AREA 通常能较好地减少锯齿；放大时使用 INTER_LINEAR。
    interpolation = (
        cv2.INTER_AREA
        if width_b > width_a or height_b > height_a
        else cv2.INTER_LINEAR
    )
    resized_b = cv2.resize(
        image_b,
        (width_a, height_a),
        interpolation=interpolation,
    )
    return image_a, resized_b, scale_info


def preprocess_gray(image: np.ndarray, blur_kernel: int = 5) -> np.ndarray:
    """执行灰度转换和高斯滤波降噪。

    灰度化降低颜色通道带来的计算量，使差分关注亮度与结构。
    高斯滤波抑制扫描颗粒、压缩噪声和轻微亮度波动；核越大，抗噪越强，
    但也越可能忽略细小变化，工业样本应根据最小缺陷尺寸进行标定。
    """

    kernel = validate_odd_kernel(blur_kernel, "blur_kernel")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (kernel, kernel), 0)


def create_difference_mask(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    *,
    threshold_value: int = 25,
    morphology_kernel: int = 3,
    morphology_iterations: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """计算绝对差异图，并通过阈值和形态学操作生成二值掩码。

    absdiff：
        逐像素计算 |A-B|，值越大代表该位置变化越明显。

    threshold：
        只保留超过阈值的显著差异，较小亮度波动变为黑色背景。

    形态学开运算：
        先腐蚀再膨胀，去除孤立小噪点。

    形态学闭运算：
        先膨胀再腐蚀，连接同一缺陷中相邻的碎片和小孔。
    """

    if gray_a.shape != gray_b.shape:
        raise ValueError("执行像素差分前，两张灰度图尺寸必须一致")
    if not 0 <= threshold_value <= 255:
        raise ValueError("threshold_value 必须在 0 到 255 之间")
    if morphology_iterations < 0:
        raise ValueError("morphology_iterations 不能小于 0")

    absolute_difference = cv2.absdiff(gray_a, gray_b)
    _, binary_mask = cv2.threshold(
        absolute_difference,
        threshold_value,
        255,
        cv2.THRESH_BINARY,
    )

    if morphology_iterations > 0:
        kernel_size = validate_odd_kernel(
            morphology_kernel,
            "morphology_kernel",
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_size, kernel_size),
        )
        binary_mask = cv2.morphologyEx(
            binary_mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=morphology_iterations,
        )
        binary_mask = cv2.morphologyEx(
            binary_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=morphology_iterations,
        )

    return absolute_difference, binary_mask


def find_difference_boxes(
    binary_mask: np.ndarray,
    *,
    min_area: float = 30.0,
    merge_gap: int = 5,
) -> list[dict[str, Any]]:
    """通过外轮廓寻找变化区域，并返回矩形坐标。

    min_area 用于过滤残余噪声。merge_gap 会先轻微膨胀掩码，使距离很近的
    差异碎片合并为一个框；最终坐标会限制在图片有效范围内。
    """

    if min_area < 0:
        raise ValueError("min_area 不能小于 0")
    if merge_gap < 0:
        raise ValueError("merge_gap 不能小于 0")

    contour_mask = binary_mask
    if merge_gap > 0:
        size = merge_gap * 2 + 1
        merge_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (size, size),
        )
        contour_mask = cv2.dilate(binary_mask, merge_kernel, iterations=1)

    contours, _ = cv2.findContours(
        contour_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    height, width = binary_mask.shape[:2]
    boxes: list[dict[str, Any]] = []
    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < min_area:
            continue

        x, y, box_width, box_height = cv2.boundingRect(contour)

        # 膨胀仅用于合并轮廓，这里向内还原 merge_gap，并限制坐标范围。
        left = max(0, x + merge_gap)
        top = max(0, y + merge_gap)
        right = min(width, x + box_width - merge_gap)
        bottom = min(height, y + box_height - merge_gap)
        if right <= left or bottom <= top:
            left, top = max(0, x), max(0, y)
            right = min(width, x + box_width)
            bottom = min(height, y + box_height)

        # area 使用原始二值差异掩码在框内的真实白色像素数量。
        changed_area = int(np.count_nonzero(binary_mask[top:bottom, left:right]))
        # 轮廓合并时经过膨胀，轮廓面积可能大于真实差异面积。因此再按真实
        # 白色像素数量过滤一次，避免稀疏噪点被误判为有效区域。
        if changed_area < min_area:
            continue
        area_ratio = changed_area / binary_mask.size

        # 面积占比以整张图片为基准。小区域更可能是文字、划痕或小缺陷；
        # 覆盖 1% 以上的大区域更可能是图片替换、元素缺失或大块色差。
        if area_ratio < 0.01:
            region_type = "small"
            possible_change = "可能文字变化或小缺陷"
        else:
            region_type = "large"
            possible_change = "可能图片替换或内容缺失"

        if area_ratio < 0.0005:
            region_severity = "low"
        elif area_ratio < 0.005:
            region_severity = "medium"
        elif area_ratio < 0.02:
            region_severity = "high"
        else:
            region_severity = "critical"

        boxes.append(
            {
                "x": int(left),
                "y": int(top),
                "width": int(right - left),
                "height": int(bottom - top),
                "area": changed_area,
                "area_ratio": round(area_ratio, 8),
                "region_type": region_type,
                "possible_change": possible_change,
                "severity_level": region_severity,
            }
        )

    boxes.sort(key=lambda box: (int(box["y"]), int(box["x"])))
    return boxes


def calculate_overall_severity(
    difference_boxes: list[dict[str, Any]],
    difference_rate: float,
    local_ssim: float,
) -> str:
    """综合差异面积、最严重区域和局部 SSIM 得到总体严重等级。"""

    if not difference_boxes:
        return "none"

    severity_order = {
        "none": 0,
        "low": 1,
        "medium": 2,
        "high": 3,
        "critical": 4,
    }
    region_level = max(
        (
            str(box["severity_level"])
            for box in difference_boxes
        ),
        key=lambda level: severity_order[level],
    )

    if difference_rate >= 0.05 or local_ssim < 0.50:
        metric_level = "critical"
    elif difference_rate >= 0.02 or local_ssim < 0.70:
        metric_level = "high"
    elif difference_rate >= 0.005 or local_ssim < 0.85:
        metric_level = "medium"
    else:
        metric_level = "low"

    return max(
        (region_level, metric_level),
        key=lambda level: severity_order[level],
    )


def create_difference_heatmap(
    aligned_image_b: np.ndarray,
    absolute_difference: np.ndarray,
    binary_mask: np.ndarray,
) -> np.ndarray:
    """生成差异热力图：蓝色较弱、黄色较强、红色最强。

    先将绝对差异归一化并使用 TURBO 伪彩色映射，再只在二值差异区域内
    叠加到图片 B。未超过阈值的位置保持原图，便于快速定位真实异常。
    """

    normalized = cv2.normalize(
        absolute_difference,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)
    color_heat = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    blended = cv2.addWeighted(
        aligned_image_b,
        0.45,
        color_heat,
        0.55,
        0,
    )
    heatmap = aligned_image_b.copy()
    heatmap[binary_mask > 0] = blended[binary_mask > 0]
    return heatmap


def create_difference_pixel_map(
    aligned_image_b: np.ndarray,
    binary_mask: np.ndarray,
) -> np.ndarray:
    """将每个确认存在差异的像素标为纯红色。

    红点来自阈值处理和形态学去噪后的二值掩码，因此不会把所有轻微扫描
    噪声都涂红。未变化像素保持图片 B 原样，便于逐像素核对。
    """

    pixel_map = aligned_image_b.copy()
    pixel_map[binary_mask > 0] = (0, 0, 255)
    return pixel_map


def create_visualization(
    aligned_image_b: np.ndarray,
    binary_mask: np.ndarray,
    difference_boxes: list[dict[str, Any]],
    *,
    image_score: float,
    global_ssim: float,
    local_ssim: float,
    severity_level: str,
) -> np.ndarray:
    """在已对齐的图片 B 上画红框，并显示变化数量和评分。"""

    # 先逐像素标红，再画矩形和编号。这样既能看出精确差异形状，也能快速
    # 找到差异区域在整张图中的位置。
    result = create_difference_pixel_map(aligned_image_b, binary_mask)
    for index, box in enumerate(difference_boxes, start=1):
        x = int(box["x"])
        y = int(box["y"])
        width = int(box["width"])
        height = int(box["height"])
        cv2.rectangle(
            result,
            (x, y),
            (x + width - 1, y + height - 1),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            result,
            f"#{index}",
            (x, max(20, y - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # 顶部使用实心底色，保证统计文字在深色或复杂图片上仍清晰可读。
    header_height = 36
    overlay = result.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (result.shape[1], min(header_height, result.shape[0])),
        (0, 0, 0),
        -1,
    )
    result = cv2.addWeighted(overlay, 0.65, result, 0.35, 0)
    summary = (
        f"Changes: {len(difference_boxes)}  "
        f"Score: {image_score:.2f}  "
        f"G-SSIM: {global_ssim:.4f}  L-SSIM: {local_ssim:.4f}  "
        f"Severity: {severity_level}"
    )
    cv2.putText(
        result,
        summary,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def compare_images(
    image_a: ImageInput,
    image_b: ImageInput,
    output_path: str | Path = "results/diff_result.jpg",
    *,
    threshold_value: int = 25,
    blur_kernel: int = 5,
    morphology_kernel: int = 3,
    morphology_iterations: int = 1,
    min_area: float = 30.0,
    merge_gap: int = 5,
    enable_alignment: bool = True,
    alignment_options: dict[str, Any] | None = None,
    prepared_pair: tuple | None = None,
) -> dict[str, Any]:
    """统一图片比较接口。

    image_a/image_b 可传入文件路径或 OpenCV NumPy 图片。默认先将 B 调整为
    A 的尺寸，再通过 ORB 自动校正平移、少量旋转和扫描偏移，最后执行差分、
    SSIM、综合评分和可视化。

    返回值包含未来 OCR 模块所需的固定字段，同时增加 has_difference 和
    result_path，便于当前图片检测阶段独立使用。
    """

    started = time.perf_counter()
    if prepared_pair is not None:
        reference, aligned_candidate, alignment_offset = prepared_pair
        if (
            not isinstance(reference, np.ndarray)
            or not isinstance(aligned_candidate, np.ndarray)
            or reference.shape != aligned_candidate.shape
            or reference.ndim != 3
            or reference.shape[2] != 3
            or reference.size == 0
            or reference.dtype != np.uint8
            or aligned_candidate.dtype != np.uint8
        ):
            raise ValueError("prepared_pair requires equally sized uint8 BGR arrays")
        alignment_offset = dict(alignment_offset)
        load_seconds = alignment_seconds = 0.0
    else:
        load_started = time.perf_counter()
        reference = load_image(image_a)
        candidate = load_image(image_b)
        reference, candidate, resize_scale = resize_to_reference(
            reference,
            candidate,
        )
        load_seconds = time.perf_counter() - load_started
        alignment_started = time.perf_counter()
        if enable_alignment:
            aligned_candidate, alignment_offset = align_images(
                reference, candidate, **(alignment_options or {}),
            )
        else:
            aligned_candidate = candidate.copy()
            alignment_offset = {
                "x": 0.0,
                "y": 0.0,
                "rotation_degrees": 0.0,
                "scale": 1.0,
                "matched_features": 0,
                "success": False,
                "reason": "自动对齐已关闭",
            }
        alignment_offset["resize_scale"] = resize_scale
        alignment_seconds = time.perf_counter() - alignment_started

    preprocess_started = time.perf_counter()
    gray_a = preprocess_gray(reference, blur_kernel)
    gray_b = preprocess_gray(aligned_candidate, blur_kernel)
    preprocess_seconds = time.perf_counter() - preprocess_started
    visual_started = time.perf_counter()
    absolute_difference, binary_mask = create_difference_mask(
        gray_a,
        gray_b,
        threshold_value=threshold_value,
        morphology_kernel=morphology_kernel,
        morphology_iterations=morphology_iterations,
    )

    different_pixels = int(cv2.countNonZero(binary_mask))
    difference_rate = float(different_pixels / binary_mask.size)
    difference_boxes = find_difference_boxes(
        binary_mask,
        min_area=min_area,
        merge_gap=merge_gap,
    )

    global_ssim = calculate_ssim(gray_a, gray_b)
    local_ssim, local_ssim_regions = calculate_local_ssim(gray_a, gray_b)
    severity_level = calculate_overall_severity(
        difference_boxes,
        difference_rate,
        local_ssim,
    )
    image_score = calculate_image_score(
        global_ssim,
        local_ssim,
        difference_rate,
        difference_boxes,
        gray_a.shape,
        severity_level,
    )
    has_difference = bool(difference_boxes)
    visual_seconds = time.perf_counter() - visual_started

    report_started = time.perf_counter()
    output = Path(output_path)
    visualized = create_visualization(
        aligned_candidate,
        binary_mask,
        difference_boxes,
        image_score=image_score,
        global_ssim=global_ssim,
        local_ssim=local_ssim,
        severity_level=severity_level,
    )
    save_image(output, visualized)

    # 同时保存灰度绝对差异图和二值掩码，便于研究阈值及误检原因。
    diff_map_path = output.with_name(f"{output.stem}_map.png")
    mask_path = output.with_name(f"{output.stem}_mask.png")
    if output.stem == "diff_result":
        heatmap_path = output.with_name("difference_heatmap.jpg")
        difference_pixels_path = output.with_name("difference_pixels.png")
    else:
        page_prefix = output.stem.removesuffix("_diff_result")
        heatmap_path = output.with_name(
            f"{page_prefix}_difference_heatmap.jpg"
        )
        difference_pixels_path = output.with_name(
            f"{page_prefix}_difference_pixels.png"
        )

    heatmap = create_difference_heatmap(
        aligned_candidate,
        absolute_difference,
        binary_mask,
    )
    difference_pixels = create_difference_pixel_map(
        aligned_candidate,
        binary_mask,
    )
    save_image(diff_map_path, absolute_difference)
    save_image(mask_path, binary_mask)
    save_image(heatmap_path, heatmap)
    save_image(difference_pixels_path, difference_pixels)
    report_seconds = time.perf_counter() - report_started

    return {
        "has_difference": has_difference,
        "image_score": image_score,
        "global_ssim": global_ssim,
        "local_ssim": local_ssim,
        # 保留旧字段，确保现有 PDF 汇总和 GUI 调用不会中断。
        "ssim_score": global_ssim,
        "difference_rate": round(difference_rate, 8),
        "difference_boxes": difference_boxes,
        "local_ssim_regions": local_ssim_regions,
        "severity_level": severity_level,
        "alignment_offset": alignment_offset,
        "result_path": str(output.resolve()),
        "heatmap_path": str(heatmap_path.resolve()),
        "difference_pixels_path": str(difference_pixels_path.resolve()),
        "timing": {
            "load_s": round(load_seconds, 6),
            "preprocess_s": round(preprocess_seconds, 6),
            "alignment_s": round(alignment_seconds, 6),
            "visual_s": round(visual_seconds, 6),
            "report_s": round(report_seconds, 6),
            "total_s": round(time.perf_counter() - started, 6),
        },
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="ORB 对齐 + OpenCV 差分 + SSIM 图片比较"
    )
    parser.add_argument("image_a", help="基准图片 A 的路径")
    parser.add_argument("image_b", help="待比较图片 B 的路径")
    parser.add_argument(
        "-o",
        "--output",
        default="results/diff_result.jpg",
        help="可视化结果路径，默认 results/diff_result.jpg",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=25,
        help="像素差阈值 0~255，默认 25",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=5,
        help="高斯滤波核大小，默认 5",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=30.0,
        help="最小变化区域面积，默认 30",
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="关闭 ORB 自动对齐，直接比较尺寸统一后的图片",
    )
    return parser


def main() -> int:
    """命令行入口，输出 JSON 结构化结果。"""

    args = build_argument_parser().parse_args()
    try:
        result = compare_images(
            args.image_a,
            args.image_b,
            output_path=args.output,
            threshold_value=args.threshold,
            blur_kernel=args.blur_kernel,
            min_area=args.min_area,
            enable_alignment=not args.no_align,
        )
    except (FileNotFoundError, TypeError, ValueError, OSError, cv2.error) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
