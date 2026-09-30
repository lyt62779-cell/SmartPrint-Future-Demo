from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
MODEL_CACHE = MODULE_DIR / "model_cache"
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(MODEL_CACHE))
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "bos")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

LOW_CONFIDENCE = 0.90
WESTERN_LANGUAGE_REVIEW_CONFIDENCE = 0.97
AMBIGUOUS_LATIN_REVIEW_CONFIDENCE = 0.94

LANGUAGE_NAMES = {
    "zh": "中文",
    "zh+latin": "中西混合",
    "en": "英文",
    "de": "德文",
    "fr": "法文",
    "es": "西班牙文",
    "latin": "拉丁语系",
    "numeric": "数字/符号",
    "conflict": "语言冲突",
    "unknown": "未知",
}

LANGUAGE_WORDS = {
    "en": {
        "the",
        "and",
        "with",
        "for",
        "from",
        "ingredients",
        "warning",
        "made",
        "product",
        "contains",
        "use",
        "keep",
    },
    "de": {
        "der",
        "die",
        "das",
        "und",
        "mit",
        "für",
        "aus",
        "zutaten",
        "achtung",
        "hergestellt",
        "enthält",
        "enthalt",
        "nicht",
        "bei",
        "hinweise",
        "kinder",
        "grosse",
        "große",
        "kuhl",
        "kühl",
    },
    "fr": {
        "le",
        "la",
        "les",
        "des",
        "avec",
        "pour",
        "ingrédients",
        "ingredients",
        "attention",
        "fabriqué",
        "fabrique",
        "contient",
        "dans",
        "sur",
        "france",
        "fabrique",
        "precautions",
        "précautions",
        "temperature",
        "température",
    },
    "es": {
        "el",
        "la",
        "los",
        "las",
        "con",
        "para",
        "ingredientes",
        "advertencia",
        "fabricado",
        "contiene",
        "producto",
        "mantener",
        "del",
        "espana",
        "españa",
        "informacion",
        "información",
        "fabricacion",
        "fabricación",
        "atencion",
        "atención",
    },
}

GERMAN_CHARS = set("äöüÄÖÜß")
FRENCH_CHARS = set("àâçèêëîïôùûÿœæÀÂÇÈÊËÎÏÔÙÛŸŒÆ")
SPANISH_CHARS = set("ñÑ¿¡")
SHARED_LATIN_ACCENTS = set("áéíóúÁÉÍÓÚ")

CANONICAL_PACKAGING_WORDS = {
    "de": {
        "für",
        "größe",
        "enthält",
        "kühl",
        "außerhalb",
        "gemäß",
    },
    "fr": {
        "fabriqué",
        "ingrédients",
        "précautions",
        "température",
        "qualité",
    },
    "es": {
        "españa",
        "información",
        "fabricación",
        "atención",
        "precaución",
    },
}


@dataclass
class OCRRecord:
    text: str
    bbox: tuple[int, int, int, int]
    confidence: float
    language: str
    reviewed: bool = False
    primary_text: str | None = None
    primary_confidence: float | None = None

    @property
    def language_name(self) -> str:
        return LANGUAGE_NAMES.get(self.language, self.language)

    def output_line(self) -> str:
        review_mark = " 二次复核" if self.reviewed else ""
        return (
            f"{self.confidence:6.2%}  [{self.language}/{self.language_name}{review_mark}]  "
            f"bbox={self.bbox}  {self.text}"
        )


@dataclass(frozen=True)
class LanguageAnalysis:
    language: str
    conflict: bool
    scores: dict[str, int]


def _is_han(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _script_counts(text: str) -> dict[str, int]:
    counts = {"han": 0, "latin": 0, "cyrillic": 0, "greek": 0, "other_letters": 0}
    for character in text:
        if _is_han(character):
            counts["han"] += 1
            continue
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        if "LATIN" in name:
            counts["latin"] += 1
        elif "CYRILLIC" in name:
            counts["cyrillic"] += 1
        elif "GREEK" in name:
            counts["greek"] += 1
        else:
            counts["other_letters"] += 1
    return counts


def analyze_language(text: str, confidence: float = 1.0) -> LanguageAnalysis:
    clean_text = text.strip()
    if not clean_text:
        return LanguageAnalysis("unknown", True, {})
    scripts = _script_counts(clean_text)
    if scripts["cyrillic"] or scripts["greek"] or scripts["other_letters"]:
        return LanguageAnalysis("conflict", True, {})
    if scripts["han"]:
        language = "zh+latin" if scripts["latin"] else "zh"
        return LanguageAnalysis(language, False, {"zh": scripts["han"]})
    if not scripts["latin"]:
        return LanguageAnalysis("numeric", False, {})

    words = re.findall(r"[^\W\d_]+", clean_text.casefold(), flags=re.UNICODE)
    scores = {
        language: sum(1 for word in words if word in vocabulary)
        for language, vocabulary in LANGUAGE_WORDS.items()
    }
    if any(character in GERMAN_CHARS for character in clean_text):
        scores["de"] += 3
    if any(character in FRENCH_CHARS for character in clean_text):
        scores["fr"] += 3
    if any(character in SPANISH_CHARS for character in clean_text):
        scores["es"] += 3
    if any(character in SHARED_LATIN_ACCENTS for character in clean_text):
        for language in ("fr", "es"):
            scores[language] += 1

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_language, best_score = ranked[0]
    second_score = ranked[1][1]
    if best_score == 0:
        language = "en" if clean_text.isascii() and len(words) >= 2 else "latin"
        return LanguageAnalysis(language, False, scores)
    conflict = second_score > 0 and best_score == second_score
    if conflict:
        return LanguageAnalysis("latin", True, scores)
    return LanguageAnalysis(best_language, False, scores)


def _strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(character for character in normalized if not unicodedata.combining(character))


def _canonical_replacements(text: str, language: str) -> dict[str, str]:
    canonical_words = CANONICAL_PACKAGING_WORDS.get(language, set())
    if not canonical_words:
        return {}
    by_plain_form = {
        _strip_diacritics(canonical): canonical
        for canonical in canonical_words
    }
    replacements: dict[str, str] = {}
    for token in re.findall(r"[^\W\d_]+", text, flags=re.UNICODE):
        canonical = by_plain_form.get(_strip_diacritics(token))
        if canonical is None or canonical.casefold() == token.casefold():
            continue
        if token.isupper():
            canonical = canonical.upper()
        elif token[:1].isupper():
            canonical = canonical[:1].upper() + canonical[1:]
        replacements[token] = canonical
    return replacements


def needs_secondary_review(record: OCRRecord, context_language: str | None = None) -> bool:
    analysis = analyze_language(record.text, record.confidence)
    effective_language = context_language or analysis.language
    if record.confidence < LOW_CONFIDENCE or analysis.conflict or "\ufffd" in record.text:
        return True
    if _canonical_replacements(record.text, effective_language):
        return True
    if effective_language in {"de", "fr", "es"}:
        return record.confidence < WESTERN_LANGUAGE_REVIEW_CONFIDENCE
    if effective_language == "latin":
        return record.confidence < AMBIGUOUS_LATIN_REVIEW_CONFIDENCE
    if len(record.text.strip()) <= 4 and record.confidence < WESTERN_LANGUAGE_REVIEW_CONFIDENCE:
        return True
    return False


class MultilingualOCRProcessor:
    """PP-OCRv6-first multilingual post-processor with per-line lazy review."""

    def __init__(
        self,
        model_factory: Callable[..., object] | None = None,
    ) -> None:
        self._model_factory = model_factory
        self._review_models: dict[str, object] = {}

    def refine_results(self, primary_results: Iterable[object]) -> list[list[OCRRecord]]:
        results = list(primary_results)
        grouped_records: list[list[OCRRecord]] = []
        grouped_context_languages: list[list[str | None]] = []
        review_jobs: list[tuple[int, int, list[np.ndarray]]] = []

        for result_index, result in enumerate(results):
            records = self._records_from_result(result)
            grouped_records.append(records)
            context_languages = self._context_languages(records)
            grouped_context_languages.append(context_languages)
            for record, context_language in zip(records, context_languages):
                analysis = analyze_language(record.text, record.confidence)
                if (
                    context_language in {"de", "fr", "es"}
                    and analysis.language in {"latin", "en"}
                    and max(analysis.scores.values(), default=0) == 0
                ):
                    record.language = context_language
            output_image = self._output_image(result)
            if output_image is None:
                continue
            for record_index, record in enumerate(records):
                context_language = context_languages[record_index]
                if not needs_secondary_review(record, context_language):
                    continue
                crop = self._crop_text_line(output_image, record.bbox)
                if crop is None:
                    continue
                review_jobs.append(
                    (
                        result_index,
                        record_index,
                        [crop, self._contrast_variant(crop)],
                    )
                )

        if review_jobs:
            flattened_inputs: list[np.ndarray] = []
            input_owners: list[tuple[int, int]] = []
            for result_index, record_index, variants in review_jobs:
                for variant in variants:
                    flattened_inputs.append(variant)
                    input_owners.append((result_index, record_index))
            try:
                model = self._get_review_model()
                review_results = list(
                    model.predict(
                        input=flattened_inputs,
                        batch_size=min(8, len(flattened_inputs)),
                    )
                )
            except Exception:
                # The PP-OCRv6 primary result remains usable if an optional
                # review model is unavailable or cannot be downloaded.
                review_results = []
            candidates_by_record: dict[tuple[int, int], list[tuple[str, float]]] = {}
            for owner, review_result in zip(input_owners, review_results):
                candidates_by_record.setdefault(owner, []).append(
                    (
                        str(review_result.get("rec_text", "")).strip(),
                        float(review_result.get("rec_score", 0.0)),
                    )
                )
            for owner, candidates in candidates_by_record.items():
                result_index, record_index = owner
                record = grouped_records[result_index][record_index]
                context_language = grouped_context_languages[result_index][record_index]
                for candidate_text, candidate_score in sorted(
                    candidates,
                    key=lambda item: item[1],
                    reverse=True,
                ):
                    self._apply_review_candidate(
                        record,
                        candidate_text,
                        candidate_score,
                        context_language,
                    )
        return grouped_records

    @staticmethod
    def _records_from_result(result: object) -> list[OCRRecord]:
        texts = list(result.get("rec_texts", []))
        scores = list(result.get("rec_scores", []))
        boxes = list(result.get("rec_boxes", []))
        polygons = list(result.get("rec_polys", result.get("dt_polys", [])))
        records: list[OCRRecord] = []
        for index, raw_text in enumerate(texts):
            text = str(raw_text).strip()
            if not text:
                continue
            confidence = float(scores[index]) if index < len(scores) else 0.0
            bbox = MultilingualOCRProcessor._bbox_at(index, boxes, polygons)
            language = analyze_language(text, confidence).language
            records.append(
                OCRRecord(
                    text=text,
                    bbox=bbox,
                    confidence=confidence,
                    language=language,
                )
            )
        return records

    @staticmethod
    def _bbox_at(
        index: int,
        boxes: list[object],
        polygons: list[object],
    ) -> tuple[int, int, int, int]:
        if index < len(boxes):
            values = np.asarray(boxes[index]).reshape(-1)
            if len(values) >= 4:
                return tuple(int(round(float(value))) for value in values[:4])
        if index < len(polygons):
            polygon = np.asarray(polygons[index]).reshape(-1, 2)
            if polygon.size:
                return (
                    int(np.floor(polygon[:, 0].min())),
                    int(np.floor(polygon[:, 1].min())),
                    int(np.ceil(polygon[:, 0].max())),
                    int(np.ceil(polygon[:, 1].max())),
                )
        return (0, 0, 0, 0)

    @staticmethod
    def _output_image(result: object) -> np.ndarray | None:
        preprocessor_result = result.get("doc_preprocessor_res")
        if preprocessor_result is None:
            return None
        image = preprocessor_result.get("output_img")
        if image is None:
            return None
        return np.asarray(image)

    @staticmethod
    def _context_languages(records: list[OCRRecord]) -> list[str | None]:
        hints: list[str | None] = []
        analyses = [analyze_language(record.text, record.confidence) for record in records]
        for index, record in enumerate(records):
            aggregate = {"en": 0, "de": 0, "fr": 0, "es": 0}
            for language, score in analyses[index].scores.items():
                if language in aggregate:
                    aggregate[language] += score * 2
            for other_index, other_record in enumerate(records):
                if other_index == index or not MultilingualOCRProcessor._same_text_block(record, other_record):
                    continue
                for language, score in analyses[other_index].scores.items():
                    if language in aggregate:
                        aggregate[language] += score
            ranked = sorted(aggregate.items(), key=lambda item: item[1], reverse=True)
            best_language, best_score = ranked[0]
            second_score = ranked[1][1]
            hints.append(best_language if best_score >= 2 and best_score > second_score else None)
        return hints

    @staticmethod
    def _same_text_block(first: OCRRecord, second: OCRRecord) -> bool:
        ax1, ay1, ax2, ay2 = first.bbox
        bx1, by1, bx2, by2 = second.bbox
        first_width = max(ax2 - ax1, 1)
        second_width = max(bx2 - bx1, 1)
        first_height = max(ay2 - ay1, 1)
        second_height = max(by2 - by1, 1)
        horizontal_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
        overlap_ratio = horizontal_overlap / max(min(first_width, second_width), 1)
        vertical_gap = max(0, max(ay1, by1) - min(ay2, by2))
        return overlap_ratio >= 0.20 and vertical_gap <= max(90, 3 * max(first_height, second_height))

    @staticmethod
    def _crop_text_line(
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        if image.ndim < 2:
            return None
        image_height, image_width = image.shape[:2]
        x1, y1, x2, y2 = bbox
        box_width = max(x2 - x1, 1)
        box_height = max(y2 - y1, 1)
        pad_x = max(3, round(box_width * 0.04))
        pad_y = max(3, round(box_height * 0.16))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(image_width, x2 + pad_x)
        y2 = min(image_height, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image[y1:y2, x1:x2].copy()
        longest = max(crop.shape[:2])
        if longest < 1200:
            import cv2

            scale = min(2.0, 1200.0 / max(longest, 1))
            crop = cv2.resize(
                crop,
                (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
                interpolation=cv2.INTER_CUBIC,
            )
        return crop

    @staticmethod
    def _contrast_variant(crop: np.ndarray) -> np.ndarray:
        import cv2

        if crop.ndim == 2:
            gray = crop
        else:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        sharpened = cv2.addWeighted(enhanced, 1.45, blurred, -0.45, 0)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    def _get_review_model(self):
        model_key = "ppocrv6"
        if model_key in self._review_models:
            return self._review_models[model_key]
        if self._model_factory is None:
            from paddleocr import TextRecognition

            factory = TextRecognition
        else:
            factory = self._model_factory
        model = factory(
            model_name="PP-OCRv6_medium_rec",
            enable_mkldnn=False,
        )
        self._review_models[model_key] = model
        return model

    @staticmethod
    def _apply_review_candidate(
        record: OCRRecord,
        candidate_text: str,
        candidate_score: float,
        context_language: str | None = None,
    ) -> None:
        if not candidate_text or candidate_text == record.text:
            return
        primary_analysis = analyze_language(record.text, record.confidence)
        candidate_analysis = analyze_language(candidate_text, candidate_score)
        if candidate_analysis.conflict:
            return
        if (
            primary_analysis.language in {"en", "de", "fr", "es", "latin"}
            and candidate_analysis.language in {"zh", "zh+latin"}
        ):
            return
        text_similarity = SequenceMatcher(
            None,
            record.text.casefold(),
            candidate_text.casefold(),
            autojunk=False,
        ).ratio()
        if text_similarity < 0.45:
            return
        resolves_conflict = primary_analysis.conflict and not candidate_analysis.conflict
        improves_score = candidate_score >= record.confidence + 0.015
        preserves_specific_language = (
            primary_analysis.language in {"de", "fr", "es"}
            and candidate_analysis.language == primary_analysis.language
            and candidate_score >= record.confidence - 0.01
        )
        canonical_target = MultilingualOCRProcessor._canonical_target(record.text, context_language)
        restores_expected_diacritic = (
            canonical_target is not None
            and candidate_text.casefold() == canonical_target.casefold()
            and candidate_score >= record.confidence - 0.08
        )
        if not (resolves_conflict or improves_score or preserves_specific_language or restores_expected_diacritic):
            return
        if primary_analysis.language in {"zh", "zh+latin"} and candidate_analysis.language not in {"zh", "zh+latin"}:
            return
        record.primary_text = record.text
        record.primary_confidence = record.confidence
        record.text = candidate_text
        record.confidence = candidate_score
        record.language = candidate_analysis.language
        record.reviewed = True

    @staticmethod
    def _canonical_target(text: str, context_language: str | None) -> str | None:
        language = context_language or analyze_language(text).language
        replacements = _canonical_replacements(text, language)
        if not replacements:
            return None
        pattern = re.compile(
            r"\b(" + "|".join(re.escape(token) for token in sorted(replacements, key=len, reverse=True)) + r")\b",
            flags=re.UNICODE,
        )
        corrected = pattern.sub(lambda match: replacements[match.group(0)], text)
        return corrected if corrected != text else None
