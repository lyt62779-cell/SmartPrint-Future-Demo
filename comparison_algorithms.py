"""Shared legacy comparison algorithms extracted without changing decision rules."""
from __future__ import annotations
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Protocol
from PIL import Image
from cnn_region_classifier import CNNRegionClassifier
from multilingual_ocr import MultilingualOCRProcessor, OCRRecord

class SearchPane(Protocol):
    path: Path | None
    page_count: int
    selection: tuple[float, float, float, float] | None

    def render_search_page(self, page_index: int) -> tuple[Image.Image, tuple[float, float]]:
        ...

class ComparisonAlgorithms:
    def _find_all_matches(
        self,
        template_path: Path,
        template_pane: SearchPane | None = None,
        search_pane: SearchPane | None = None,
        *,
        seed_page_index: int | None = None,
        seed_selection: tuple[float, float, float, float] | None = None,
    ) -> list[dict[str, object]]:
        import cv2
        import numpy as np

        template_pane = template_pane or self.original_pane
        search_pane = search_pane or self.compare_pane
        if template_pane.selection is None or search_pane.path is None:
            raise ValueError("缺少模板框选或目标图片")
        template_image = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if template_image is None:
            raise ValueError("无法读取框选模板区域")
        template_width_ref = (
            template_pane.selection[2] - template_pane.selection[0]
        )
        template_height_ref = (
            template_pane.selection[3] - template_pane.selection[1]
        )
        rotations = {
            0: template_image,
            90: cv2.rotate(template_image, cv2.ROTATE_90_COUNTERCLOCKWISE),
            180: cv2.rotate(template_image, cv2.ROTATE_180),
            270: cv2.rotate(template_image, cv2.ROTATE_90_CLOCKWISE),
        }
        scale_candidates = (0.82, 0.90, 0.96, 1.0, 1.04, 1.10, 1.18)
        candidates: list[dict[str, object]] = []
        page_count = max(search_pane.page_count, 1)
        for page_index in range(page_count):
            search_pil, reference_size = search_pane.render_search_page(
                page_index
            )
            search_gray = cv2.cvtColor(np.array(search_pil), cv2.COLOR_RGB2GRAY)
            search_edges = cv2.Canny(search_gray, 60, 160)
            pixels_per_ref_x = search_gray.shape[1] / reference_size[0]
            pixels_per_ref_y = search_gray.shape[0] / reference_size[1]
            for rotation, rotated_template in rotations.items():
                if rotation in (0, 180):
                    expected_width = template_width_ref * pixels_per_ref_x
                    expected_height = template_height_ref * pixels_per_ref_y
                else:
                    expected_width = template_height_ref * pixels_per_ref_x
                    expected_height = template_width_ref * pixels_per_ref_y
                for scale_factor in scale_candidates:
                    width = max(20, round(expected_width * scale_factor))
                    height = max(20, round(expected_height * scale_factor))
                    if width >= search_gray.shape[1] or height >= search_gray.shape[0]:
                        continue
                    interpolation = cv2.INTER_AREA if width < rotated_template.shape[1] else cv2.INTER_CUBIC
                    resized = cv2.resize(rotated_template, (width, height), interpolation=interpolation)
                    resized = cv2.equalizeHist(resized)
                    template_edges = cv2.Canny(resized, 60, 160)
                    gray_result = cv2.matchTemplate(search_gray, resized, cv2.TM_CCOEFF_NORMED)
                    edge_result = cv2.matchTemplate(search_edges, template_edges, cv2.TM_CCOEFF_NORMED)
                    combined = gray_result * 0.65 + edge_result * 0.35
                    local_maxima = combined >= cv2.dilate(combined, np.ones((17, 17), dtype=np.uint8))
                    locations = np.argwhere(local_maxima & (combined >= 0.38))
                    if locations.size == 0:
                        continue
                    ranked = sorted(
                        ((float(combined[y, x]), int(x), int(y)) for y, x in locations),
                        reverse=True,
                    )[:12]
                    for score, x, y in ranked:
                        candidates.append(
                            {
                                "page_index": page_index,
                                "rotation": rotation,
                                "score": score,
                                "selection": (
                                    x / pixels_per_ref_x,
                                    y / pixels_per_ref_y,
                                    (x + width) / pixels_per_ref_x,
                                    (y + height) / pixels_per_ref_y,
                                ),
                            }
                        )
        seed_match: dict[str, object] | None = None
        if seed_page_index is not None and seed_selection is not None:
            # 在重排图内反查其他副本时，用户当前框选本身必须作为第一个
            # 可靠实例保留下来；其余实例仍由模板匹配得分决定。
            seed_match = {
                "page_index": seed_page_index,
                "rotation": 0,
                "score": 1.0,
                "selection": seed_selection,
            }
            candidates.append(seed_match)
        return self._select_match_candidates(candidates, seed_match)

    @classmethod
    def _select_match_candidates(
        cls,
        candidates: list[dict[str, object]],
        seed_match: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        """按置信度和重叠率筛选自动定位结果。

        当模板来自重排图本身时，seed_match 的得分必然接近 100%。门槛
        改由最佳的“其他位置”决定，避免漏掉存在轻微印刷差异的另外
        1～3 个副本。
        """

        if not candidates:
            raise ValueError("未能可靠找到对应版面，请扩大框选范围后重试")
        candidates.sort(key=lambda item: float(item["score"]), reverse=True)
        best_score = float(candidates[0]["score"])
        if best_score < 0.38:
            score_text = f"{best_score:.2%}"
            raise ValueError(
                f"未能可靠找到对应版面（最高匹配度 {score_text}），"
                "请扩大框选范围后重试"
            )

        threshold_anchor = best_score
        if seed_match is not None:
            other_scores = [
                float(candidate["score"])
                for candidate in candidates
                if cls._match_overlap(candidate, seed_match) < 0.30
            ]
            if other_scores:
                # 自身匹配固定接近 100%，不能让它把其他轻微变化的副本
                # 门槛抬得过高。使用“最佳其他位置”估计多实例门槛。
                threshold_anchor = max(other_scores)
        minimum_score = max(0.45, threshold_anchor * 0.76)
        matches: list[dict[str, object]] = []
        for candidate in candidates:
            if float(candidate["score"]) < minimum_score:
                break
            if any(
                cls._match_overlap(candidate, accepted) >= 0.30
                for accepted in matches
            ):
                continue
            matches.append(candidate)
            if len(matches) >= 8:
                break
        matches.sort(
            key=lambda item: (
                int(item["page_index"]),
                item["selection"][1],
                item["selection"][0],
            )
        )
        return matches

    @staticmethod
    def _match_overlap(first: dict[str, object], second: dict[str, object]) -> float:
        if int(first["page_index"]) != int(second["page_index"]):
            return 0.0
        ax1, ay1, ax2, ay2 = first["selection"]
        bx1, by1, bx2, by2 = second["selection"]
        intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
        intersection = intersection_width * intersection_height
        if intersection <= 0:
            return 0.0
        first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return intersection / max(first_area + second_area - intersection, 1.0)

    @staticmethod
    def _normalize_region_orientation(
        region_path: Path,
        rotation: int,
        index: int,
    ) -> Path:
        """把自动定位到的旋转版面恢复为原图方向，供图片模块比较。

        match.rotation 表示“原图模板为了匹配重排图所旋转的角度”，因此
        恢复重排区域时需要执行相反方向的旋转。
        """

        if rotation == 0:
            return region_path

        transpose_by_rotation = {
            90: Image.Transpose.ROTATE_270,
            180: Image.Transpose.ROTATE_180,
            270: Image.Transpose.ROTATE_90,
        }
        if rotation not in transpose_by_rotation:
            raise ValueError(f"不支持的自动定位旋转角度: {rotation}")

        normalized_path = region_path.with_name(
            f"visual_normalized_{index}_{time.time_ns()}.png"
        )
        with Image.open(region_path) as source:
            normalized = source.convert("RGB").transpose(
                transpose_by_rotation[rotation]
            )
            normalized.save(normalized_path)
        return normalized_path

    @staticmethod
    def _map_difference_box_to_page(
        match: dict[str, object],
        box: dict[str, object],
        analyzed_size: tuple[int, int],
    ) -> tuple[float, float, float, float]:
        """将图片模块的像素框映射回重排 PDF/图片的原画布坐标。

        图片模块比较的是已经转正的裁剪图，而 UI 展示的是重排图原方向。
        因此先根据 0/90/180/270 度做逆坐标变换，再按自动定位 selection
        的宽高比例映射到页面参考坐标。
        """

        analyzed_width, analyzed_height = analyzed_size
        if analyzed_width <= 0 or analyzed_height <= 0:
            raise ValueError("图片差异分析尺寸无效")

        x = float(box["x"])
        y = float(box["y"])
        width = float(box["width"])
        height = float(box["height"])
        rotation = int(match["rotation"])

        if rotation == 0:
            crop_x, crop_y = x, y
            crop_width, crop_height = width, height
            crop_image_width, crop_image_height = (
                analyzed_width,
                analyzed_height,
            )
        elif rotation == 180:
            crop_x = analyzed_width - (x + width)
            crop_y = analyzed_height - (y + height)
            crop_width, crop_height = width, height
            crop_image_width, crop_image_height = (
                analyzed_width,
                analyzed_height,
            )
        elif rotation == 90:
            # 转正时对重排裁剪图顺时针旋转了 90°；映射回页面需逆变换。
            crop_x = y
            crop_y = analyzed_width - (x + width)
            crop_width, crop_height = height, width
            crop_image_width, crop_image_height = (
                analyzed_height,
                analyzed_width,
            )
        elif rotation == 270:
            crop_x = analyzed_height - (y + height)
            crop_y = x
            crop_width, crop_height = height, width
            crop_image_width, crop_image_height = (
                analyzed_height,
                analyzed_width,
            )
        else:
            raise ValueError(f"不支持的差异框旋转角度: {rotation}")

        selection_x1, selection_y1, selection_x2, selection_y2 = match[
            "selection"
        ]
        selection_width = float(selection_x2 - selection_x1)
        selection_height = float(selection_y2 - selection_y1)

        page_x1 = float(selection_x1) + (
            crop_x / crop_image_width * selection_width
        )
        page_y1 = float(selection_y1) + (
            crop_y / crop_image_height * selection_height
        )
        page_x2 = float(selection_x1) + (
            (crop_x + crop_width) / crop_image_width * selection_width
        )
        page_y2 = float(selection_y1) + (
            (crop_y + crop_height) / crop_image_height * selection_height
        )
        return page_x1, page_y1, page_x2, page_y2

    @classmethod
    def _map_text_predictions_to_page(
        cls,
        mapping: dict[str, object],
        predictions: list[dict[str, object]],
        analyzed_size: tuple[int, int],
    ) -> list[tuple[float, float, float, float]]:
        """把 CNN document_text 像素框映射回PDF页面参考坐标。"""

        mapped: list[tuple[float, float, float, float]] = []
        for prediction in predictions:
            x, y, width, height = prediction["bbox"]
            mapped.append(
                cls._map_difference_box_to_page(
                    mapping,
                    {
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                    },
                    analyzed_size,
                )
            )
        return mapped

    @staticmethod
    def _ocr_records_to_paragraph_predictions(
        records: list[OCRRecord],
    ) -> list[dict[str, object]]:
        """把 PP-OCRv6 文字行按版面间距合并成段落级区域。

        CNN 的候选框只决定 OCR 搜索范围，最终段落边界由实际识别出的文字
        行决定。这样候选框中即使混有产品图片，绿色框也只包围文字段落。
        """

        if not records:
            return []

        boxes = [tuple(float(value) for value in record.bbox) for record in records]

        def should_join(first_index: int, second_index: int) -> bool:
            x1, y1, x2, y2 = boxes[first_index]
            ax1, ay1, ax2, ay2 = boxes[second_index]
            width1 = max(x2 - x1, 1.0)
            height1 = max(y2 - y1, 1.0)
            width2 = max(ax2 - ax1, 1.0)
            height2 = max(ay2 - ay1, 1.0)
            horizontal_overlap = max(0.0, min(x2, ax2) - max(x1, ax1))
            vertical_overlap = max(0.0, min(y2, ay2) - max(y1, ay1))
            horizontal_gap = max(0.0, max(x1, ax1) - min(x2, ax2))
            vertical_gap = max(0.0, max(y1, ay1) - min(y2, ay2))
            height_ratio = max(height1, height2) / min(height1, height2)

            # 同一横排段落中的相邻行：高度接近、横向重叠且行距较小。
            stacked_lines = (
                height_ratio <= 2.5
                and horizontal_overlap / min(width1, width2) >= 0.15
                and vertical_gap <= max(height1, height2) * 0.90
            )
            # OCR 有时会把同一行拆成多个片段，将紧邻片段重新组成一行。
            same_line_fragments = (
                vertical_overlap / min(height1, height2) >= 0.35
                and horizontal_gap <= max(height1, height2) * 2.2
            )
            return stacked_lines or same_line_fragments

        visited: set[int] = set()
        paragraphs: list[dict[str, object]] = []
        for start_index in range(len(boxes)):
            if start_index in visited:
                continue
            visited.add(start_index)
            stack = [start_index]
            component: list[int] = []
            while stack:
                current = stack.pop()
                component.append(current)
                for candidate_index in range(len(boxes)):
                    if candidate_index in visited:
                        continue
                    if should_join(current, candidate_index):
                        visited.add(candidate_index)
                        stack.append(candidate_index)

            left = min(boxes[index][0] for index in component)
            top = min(boxes[index][1] for index in component)
            right = max(boxes[index][2] for index in component)
            bottom = max(boxes[index][3] for index in component)
            paragraphs.append(
                {
                    "bbox": [
                        int(round(left)),
                        int(round(top)),
                        int(round(right - left)),
                        int(round(bottom - top)),
                    ],
                    "class_id": 0,
                    "class_name": "document_text",
                    "confidence": min(
                        float(records[index].confidence) for index in component
                    ),
                    "ocr_line_count": len(component),
                }
            )
        paragraphs.sort(
            key=lambda prediction: (
                prediction["bbox"][1],
                prediction["bbox"][0],
            )
        )
        return paragraphs

    def _multilingual_processor(self) -> MultilingualOCRProcessor:
        processor = getattr(self.ocr_owner, "multilingual_processor", None)
        if processor is None:
            processor = MultilingualOCRProcessor()
            self.ocr_owner.multilingual_processor = processor
        return processor

    def _cnn_classifier(self) -> CNNRegionClassifier:
        """延迟启动独立 CNN 运行时，并在多次对比中复用已加载模型。"""

        if self.cnn_region_classifier is None:
            self.cnn_region_classifier = CNNRegionClassifier(device="auto")
        return self.cnn_region_classifier

    @staticmethod
    def _recognize_region(
        ocr,
        path: Path,
        processor: MultilingualOCRProcessor | None = None,
    ) -> list[OCRRecord]:
        return ComparisonAlgorithms._recognize_regions(ocr, [path], processor)[0]

    @staticmethod
    def _recognize_regions(
        ocr,
        paths: list[Path],
        processor: MultilingualOCRProcessor | None = None,
        *,
        batch_size: int = 8,
    ) -> list[list[OCRRecord]]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("OCR batch_size must be a positive integer")
        processor = processor or MultilingualOCRProcessor()
        grouped_records: list[list[OCRRecord]] = []
        for start in range(0, len(paths), batch_size):
            batch = paths[start:start + batch_size]
            predictions = ocr.predict(
                [str(path) for path in batch],
                use_doc_orientation_classify=True,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            before = len(grouped_records)
            for result in predictions:
                # Language context is computed within one image, so retaining
                # all lines here preserves it without holding every page image.
                grouped_records.extend(processor.refine_results([result]))
                del result
            if len(grouped_records) - before != len(batch):
                raise RuntimeError(
                    f"OCR returned {len(grouped_records) - before} result groups "
                    f"for a batch of {len(batch)} input regions"
                )
        if len(grouped_records) != len(paths):
            raise RuntimeError(f"OCR 返回 {len(grouped_records)} 组结果，但输入了 {len(paths)} 个区域")
        return grouped_records

    @staticmethod
    def _tokens(records: list[OCRRecord]) -> list[str]:
        text = "\n".join(record.text for record in records)
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    @classmethod
    def _difference_lines(
        cls,
        original_records: list[OCRRecord],
        compare_records: list[OCRRecord],
    ) -> tuple[float, list[str]]:
        original_tokens = cls._tokens(original_records)
        compare_tokens = cls._tokens(compare_records)
        matcher = SequenceMatcher(None, original_tokens, compare_tokens, autojunk=False)
        differences: list[str] = []
        for operation, i1, i2, j1, j2 in matcher.get_opcodes():
            if operation == "equal":
                continue
            original_text = " ".join(original_tokens[i1:i2])
            compare_text = " ".join(compare_tokens[j1:j2])
            if operation == "replace":
                original_slice = original_tokens[i1:i2]
                compare_slice = compare_tokens[j1:j2]
                paired_count = min(len(original_slice), len(compare_slice))
                if paired_count:
                    paired_original = " ".join(original_slice[:paired_count])
                    paired_compare = " ".join(compare_slice[:paired_count])
                    differences.append(f"替换：原图「{paired_original}」 → 重排图「{paired_compare}」")
                if len(original_slice) > paired_count:
                    missing_text = " ".join(original_slice[paired_count:])
                    differences.append(f"重排图缺少：「{missing_text}」")
                if len(compare_slice) > paired_count:
                    added_text = " ".join(compare_slice[paired_count:])
                    differences.append(f"重排图新增：「{added_text}」")
            elif operation == "delete":
                differences.append(f"重排图缺少：「{original_text}」")
            elif operation == "insert":
                differences.append(f"重排图新增：「{compare_text}」")
        return matcher.ratio(), differences

    @staticmethod
    def _is_critical_text_difference(difference: str) -> bool:
        """判断文字差异是否涉及工业场景中的关键信息。

        数字、型号、批次、日期、容量、条码和警告信息发生替换或缺失时，
        即使整体文字相似度仍然很高，也应降低最终评分。
        """

        # 多版面对比会添加“区域 N：”前缀。先去掉此前缀，避免区域编号
        # 被误判为产品型号数字，也确保后面的“替换/缺少”能被识别。
        content = re.sub(r"^区域\s+\d+：", "", difference)
        upper = content.upper()
        critical_keywords = (
            "ITEM",
            "MODEL",
            "LOT",
            "BATCH",
            "DATE",
            "EXP",
            "WARNING",
            "STORAGE",
            "CAPACITY",
            "型号",
            "编号",
            "批号",
            "日期",
            "容量",
            "条码",
            "警告",
            "注意",
        )
        has_number = re.search(r"\d", content) is not None
        has_keyword = any(keyword in upper for keyword in critical_keywords)
        important_operation = (
            content.startswith("替换")
            or "缺少" in content
            or "新增" in content
        )
        return important_operation and (has_number or has_keyword)

    @staticmethod
    def _analyze_movement(
        image_result: dict[str, object],
    ) -> dict[str, object]:
        """只根据 ORB 对齐参数判断整个框选区域是否发生移动。

        图片已经先经过 ORB 对齐，再执行内容差分。因此：
        - 对齐矩阵中的平移、旋转、缩放用于判断“移动”；
        - 对齐后的 difference_boxes 用于判断“新增、缺失或内容变化”。

        这两个结论互不加权，避免同一问题被重复计分。
        """

        alignment = image_result.get("alignment_offset")
        if not isinstance(alignment, dict) or not alignment.get("success"):
            reason = (
                str(alignment.get("reason", "ORB 对齐失败"))
                if isinstance(alignment, dict)
                else "缺少 ORB 对齐结果"
            )
            return {
                "status": "uncertain",
                "movement_detected": False,
                "translation_pixels": 0.0,
                "translation_ratio": 0.0,
                "offset_x": 0.0,
                "offset_y": 0.0,
                "rotation_degrees": 0.0,
                "scale_change": 0.0,
                "reason": reason,
            }

        offset_x = float(alignment.get("x", 0.0))
        offset_y = float(alignment.get("y", 0.0))
        rotation = abs(float(alignment.get("rotation_degrees", 0.0)))
        scale_change = abs(float(alignment.get("scale", 1.0)) - 1.0)
        translation = (offset_x * offset_x + offset_y * offset_y) ** 0.5

        comparison_size = image_result.get("comparison_size", (0, 0))
        width, height = comparison_size
        diagonal = (float(width) ** 2 + float(height) ** 2) ** 0.5
        translation_ratio = (
            translation / diagonal
            if diagonal > 0
            else 0.0
        )
        translation_moved = (
            translation_ratio >= 0.005
            if diagonal > 0
            else translation >= 3.0
        )
        movement_detected = (
            translation_moved
            or rotation >= 0.5
            or scale_change >= 0.01
        )
        return {
            "status": "moved" if movement_detected else "not_moved",
            "movement_detected": movement_detected,
            "translation_pixels": round(translation, 3),
            "translation_ratio": round(translation_ratio, 8),
            "offset_x": round(offset_x, 3),
            "offset_y": round(offset_y, 3),
            "rotation_degrees": round(rotation, 4),
            "scale_change": round(scale_change, 6),
            "reason": "",
        }

    @classmethod
    def _build_rule_result(
        cls,
        original_records: list[OCRRecord],
        comparison_results: list[
            tuple[dict[str, object], list[OCRRecord]]
        ],
        image_results: list[dict[str, object]],
    ) -> dict[str, object]:
        """按明确规则生成最终结论，不计算 OCR/图像加权分数。

        判定优先级：
        1. OCR 文字变化，或对齐后出现图片内容差异：ERROR；
        2. 仅发生整体平移、旋转、缩放，或移动无法确认：WARNING；
        3. 文字、内容和位置均一致：PASS。
        """

        if not comparison_results or not image_results:
            raise ValueError("缺少 OCR 或图片检测结果，无法执行规则判定")
        if len(comparison_results) != len(image_results):
            raise ValueError("OCR 与图片检测的版面数量不一致")

        ocr_similarities: list[float] = []
        all_text_differences: list[str] = []
        for index, (_match, records) in enumerate(
            comparison_results,
            start=1,
        ):
            similarity, differences = cls._difference_lines(
                original_records,
                records,
            )
            ocr_similarities.append(similarity)
            all_text_differences.extend(
                f"区域 {index}：{difference}"
                for difference in differences
            )

        movement_results = [
            cls._analyze_movement(result)
            for result in image_results
        ]
        image_content_errors = [
            {
                "region_index": region_index,
                "box_index": box_index,
                "box": box,
            }
            for region_index, result in enumerate(image_results, start=1)
            for box_index, box in enumerate(
                result["difference_boxes"],
                start=1,
            )
        ]
        critical_text_errors = [
            difference
            for difference in all_text_differences
            if cls._is_critical_text_difference(difference)
        ]

        ocr_has_error = bool(all_text_differences)
        image_has_error = bool(image_content_errors)
        movement_detected = any(
            bool(result["movement_detected"])
            for result in movement_results
        )
        movement_uncertain = any(
            result["status"] == "uncertain"
            for result in movement_results
        )

        if ocr_has_error or image_has_error:
            final_status = "ERROR"
        elif movement_detected or movement_uncertain:
            final_status = "WARNING"
        else:
            final_status = "PASS"

        return {
            # 相似度仅供查看，不参与最终状态判定。
            "ocr_score": round(min(ocr_similarities) * 100.0, 2),
            "image_score": round(
                min(float(result["image_score"]) for result in image_results),
                2,
            ),
            "ocr_status": "ERROR" if ocr_has_error else "PASS",
            "image_status": (
                "ERROR"
                if image_has_error
                else (
                    "WARNING"
                    if movement_detected or movement_uncertain
                    else "PASS"
                )
            ),
            "final_status": final_status,
            "critical_text_errors": critical_text_errors,
            "text_differences": all_text_differences,
            "image_content_errors": image_content_errors,
            "movement_results": movement_results,
            "movement_detected": movement_detected,
            "movement_uncertain": movement_uncertain,
        }

    @staticmethod
    def _format_integrated_result(
        fusion: dict[str, object],
        image_results: list[dict[str, object]],
        ocr_output: str,
        visual_output_dir: Path,
    ) -> str:
        """先输出可着色的错误区，再显示规则结论和完整检测详情。"""

        critical_errors = list(fusion["critical_text_errors"])
        text_differences = list(fusion["text_differences"])
        image_content_errors = list(fusion["image_content_errors"])
        movement_results = list(fusion["movement_results"])
        lines: list[str] = []

        # UI 会把这一段标成红色。文字错误始终位于结果开头，避免用户
        # 在大量 OCR 明细中向下查找。
        if text_differences:
            lines.append("===== OCR 对比错误（红色） =====")
            critical_set = set(critical_errors)
            for index, difference in enumerate(text_differences, start=1):
                prefix = "【关键】" if difference in critical_set else ""
                lines.append(f"{index}. {prefix}{difference}")

        # 整个初始框选版面的 OpenCV 差分错误紧随 OCR 错误，UI 标成
        # 黄色。此处不使用 CNN image_content 分类结果。
        if image_content_errors:
            if lines:
                lines.append("")
            lines.append("===== 整版图片对比错误（黄色） =====")
            for error in image_content_errors:
                box = error["box"]
                lines.append(
                    f"区域 {int(error['region_index'])}-"
                    f"{int(error['box_index'])}："
                    f"坐标 ({int(box['x'])}, {int(box['y'])}, "
                    f"{int(box['width'])}, {int(box['height'])})；"
                    f"占比 {float(box['area_ratio']):.4%}；"
                    f"等级 {box['severity_level']}；"
                    f"{box['possible_change']}"
                )

        if lines:
            lines.append("")
        lines.extend(
            (
                "===== 规则判定结果 =====",
                f"最终状态：{fusion['final_status']}",
                "规则：OCR文字变化或整版图片变化 = ERROR；仅移动 = WARNING；完全一致 = PASS",
                "OCR 与图像分数仅作参考，不进行加权。",
            )
        )

        moved_regions = [
            (index, movement)
            for index, movement in enumerate(movement_results, start=1)
            if movement["movement_detected"]
        ]
        uncertain_regions = [
            (index, movement)
            for index, movement in enumerate(movement_results, start=1)
            if movement["status"] == "uncertain"
        ]

        if moved_regions:
            lines.append("【位置移动：WARNING】")
            for region_index, movement in moved_regions:
                lines.append(
                    f"区域 {region_index}：平移 "
                    f"{float(movement['translation_pixels']):.2f}px "
                    f"(X={float(movement['offset_x']):.2f}, "
                    f"Y={float(movement['offset_y']):.2f})；"
                    f"旋转 {float(movement['rotation_degrees']):.3f}°；"
                    f"缩放变化 {float(movement['scale_change']):.3%}"
                )
        if uncertain_regions:
            lines.append("【移动状态无法确认：WARNING】")
            lines.extend(
                f"区域 {index}：{movement['reason']}"
                for index, movement in uncertain_regions
            )
        if (
            not text_differences
            and not image_content_errors
            and not moved_regions
            and not uncertain_regions
        ):
            lines.append("未发现文字变化、图片内容变化或位置移动。")

        lines.extend(
            (
                "",
                "===== 检测汇总（规则判定，不加权） =====",
                f"OCR 检测：{fusion['ocr_status']}",
                f"图片检测：{fusion['image_status']}",
                f"最终状态：{fusion['final_status']}",
                f"OCR 相似度（仅参考）：{float(fusion['ocr_score']):.2f}%",
                f"图片相似度（仅参考）：{float(fusion['image_score']):.2f}%",
                "",
                "===== 图片检测详情 =====",
            )
        )

        for index, result in enumerate(image_results, start=1):
            boxes = result["difference_boxes"]
            movement = movement_results[index - 1]
            lines.extend(
                (
                    f"区域 {index}：",
                    f"  移动状态：{movement['status']}",
                    f"  平移：{float(movement['translation_pixels']):.2f}px；"
                    f"旋转：{float(movement['rotation_degrees']):.3f}°；"
                    f"缩放变化：{float(movement['scale_change']):.3%}",
                    f"  全局 SSIM（仅参考）：{float(result['global_ssim']):.4f}",
                    f"  局部 SSIM（仅参考）：{float(result['local_ssim']):.4f}",
                    f"  严重等级：{result['severity_level']}",
                    f"  新增、缺失或内容变化区域：{len(boxes)} 个",
                )
            )
            if movement["reason"]:
                lines.append(f"  移动检测说明：{movement['reason']}")
            for box_index, box in enumerate(boxes, start=1):
                lines.append(
                    f"    {box_index}. 坐标 "
                    f"({int(box['x'])}, {int(box['y'])}, "
                    f"{int(box['width'])}, {int(box['height'])})；"
                    f"占比 {float(box['area_ratio']):.4%}；"
                    f"{box['possible_change']}；"
                    f"等级 {box['severity_level']}"
                )
            lines.extend(
                (
                    f"  红点像素图：{result['difference_pixels_path']}",
                    f"  差异热力图：{result['heatmap_path']}",
                )
            )

        lines.extend(
            (
                "",
                f"图片检测文件目录：{visual_output_dir.resolve()}",
                "",
                "===== OCR 检测详情与文字差异 =====",
                ocr_output,
            )
        )
        return "\n".join(lines)

    @classmethod
    def _format_comparison(
        cls,
        original_records: list[OCRRecord],
        compare_records: list[OCRRecord],
    ) -> str:
        similarity, differences = cls._difference_lines(original_records, compare_records)
        lines = [f"相似度：{similarity:.2%}", "", "===== 原图框选 OCR ====="]
        lines.extend(record.output_line() for record in original_records)
        lines.extend(("", "===== 重排图框选 OCR ====="))
        lines.extend(record.output_line() for record in compare_records)
        lines.extend(("", "===== 文字差异 ====="))
        if differences:
            lines.extend(f"{index}. {difference}" for index, difference in enumerate(differences, 1))
        else:
            lines.append("未发现文字或标点差异。")
        return "\n".join(lines)

    @classmethod
    def _format_multiple_comparisons(
        cls,
        original_records: list[OCRRecord],
        comparison_results: list[tuple[dict[str, object], list[OCRRecord]]],
    ) -> str:
        lines = [
            f"自动找到 {len(comparison_results)} 个对应版面。",
            "",
            "===== 原图框选 OCR =====",
        ]
        lines.extend(record.output_line() for record in original_records)
        for index, (match, records) in enumerate(comparison_results, start=1):
            similarity, differences = cls._difference_lines(original_records, records)
            lines.extend(
                (
                    "",
                    f"===== 重排图区域 {index} =====",
                    f"位置：第 {int(match['page_index']) + 1} 页；"
                    f"旋转 {int(match['rotation'])}°；"
                    f"视觉匹配度 {float(match['score']):.2%}；"
                    f"文字相似度 {similarity:.2%}",
                    "",
                    "--- OCR ---",
                )
            )
            lines.extend(record.output_line() for record in records)
            lines.extend(("", "--- 与原图的文字差异 ---"))
            if differences:
                lines.extend(f"{item_index}. {difference}" for item_index, difference in enumerate(differences, 1))
            else:
                lines.append("未发现文字或标点差异。")
        return "\n".join(lines)
