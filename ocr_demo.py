from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, X, Button, Canvas, Frame, Label, OptionMenu, StringVar, Tk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageTk

from gpu_runtime import configure_paddle_gpu_dll_paths


# 必须在首次导入 Paddle/PaddleOCR 之前配置当前虚拟环境内的 CUDA DLL。
# 该调用不改变 OCR 模型、参数或识别流程。
configure_paddle_gpu_dll_paths()

from multilingual_ocr import MultilingualOCRProcessor


BASE_DIR = Path(__file__).resolve().parent
MODEL_CACHE = BASE_DIR / "model_cache"
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(MODEL_CACHE))
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "bos")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

SUPPORTED_INPUTS = [
    ("图片或 PDF", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff *.pdf"),
    ("PDF 文件", "*.pdf"),
    ("图片文件", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
    ("所有文件", "*.*"),
]

DIRECTION_OPTIONS = {
    "自动检测（推荐）": None,
    "不旋转（正常）": 0,
    "向左旋转 90°": 90,
    "旋转 180°（倒置）": 180,
    "向右旋转 90°": 270,
}
PREVIEW_MAX_SIDE = 1800.0


class OCRDemo:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("PaddleOCR 文字识别 Demo")
        self.root.geometry("1100x720")
        self.root.minsize(900, 600)

        self.input_path: Path | None = None
        self.original_size: tuple[int, int] | None = None
        self.pdf_page_count: int | None = None
        self.pdf_page_index = 0
        self.pdf_selection: tuple[float, float, float, float] | None = None
        self.preview_image_bounds: tuple[float, float, float, float] | None = None
        self.preview_pdf_size: tuple[float, float] | None = None
        self.selection_start: tuple[float, float] | None = None
        self.selection_rect_id: int | None = None
        self.preview_source_image: Image.Image | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.preview_zoom = 1.0
        self.preview_pan = (0.0, 0.0)
        self.interaction_mode = "select"
        self.pan_last: tuple[float, float] | None = None
        self.resize_job = None
        self.ocr = None
        self.multilingual_processor: MultilingualOCRProcessor | None = None
        self.running = False
        self.processing_lock = threading.Lock()
        self.compare_windows = []
        self.close_requested = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.status = StringVar(value="请选择一张包含文字的图片")
        self.page_status = StringVar(value="")
        self.zoom_status = StringVar(value="100%")
        self.direction_option = StringVar(value="自动检测（推荐）")

        self._build_ui()

    def _build_ui(self) -> None:
        toolbar = Frame(self.root, padx=12, pady=10)
        toolbar.pack(fill=X)

        Button(toolbar, text="选择文件", width=10, command=self.choose_input).pack(side=LEFT, padx=(0, 4))
        self.recognize_button = Button(toolbar, text="开始识别", width=10, command=self.start_recognition)
        self.recognize_button.pack(side=LEFT, padx=4)
        Button(toolbar, text="复制文字", width=10, command=self.copy_text).pack(side=LEFT, padx=4)
        Button(toolbar, text="保存 TXT", width=10, command=self.save_text).pack(side=LEFT, padx=4)
        Button(toolbar, text="双图对比", width=10, command=self.open_compare_window).pack(side=LEFT, padx=4)
        self.previous_page_button = Button(toolbar, text="上一页", width=6, state="disabled", command=self.previous_pdf_page)
        self.previous_page_button.pack(side=LEFT, padx=(12, 2))
        self.next_page_button = Button(toolbar, text="下一页", width=6, state="disabled", command=self.next_pdf_page)
        self.next_page_button.pack(side=LEFT, padx=2)
        Label(toolbar, textvariable=self.page_status, width=7).pack(side=LEFT, padx=2)
        self.clear_selection_button = Button(
            toolbar,
            text="清除框选",
            width=8,
            state="disabled",
            command=self.clear_pdf_selection,
        )
        self.clear_selection_button.pack(side=LEFT, padx=2)

        content = Frame(self.root, padx=12)
        content.pack(fill=BOTH, expand=True, pady=(0, 8))

        left = Frame(content, bd=1, relief="solid")
        left.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 6))
        Label(left, text="文件预览", anchor="w", padx=10, pady=6).pack(fill=X)
        preview_toolbar = Frame(left, padx=8, pady=3)
        preview_toolbar.pack(fill=X, pady=(0, 3))
        self.select_mode_button = Button(
            preview_toolbar,
            text="框选区域",
            width=9,
            state="disabled",
            command=lambda: self._set_interaction_mode("select"),
        )
        self.select_mode_button.pack(side=LEFT, padx=(0, 4))
        self.pan_mode_button = Button(
            preview_toolbar,
            text="拖动画面",
            width=9,
            state="disabled",
            command=lambda: self._set_interaction_mode("pan"),
        )
        self.pan_mode_button.pack(side=LEFT, padx=4)
        self.zoom_out_button = Button(
            preview_toolbar, text="缩小 −", width=8, state="disabled", command=self.zoom_out
        )
        self.zoom_out_button.pack(side=LEFT, padx=(12, 4))
        self.zoom_in_button = Button(
            preview_toolbar, text="放大 +", width=8, state="disabled", command=self.zoom_in
        )
        self.zoom_in_button.pack(side=LEFT, padx=4)
        self.fit_button = Button(
            preview_toolbar, text="适应窗口", width=9, state="disabled", command=self.fit_preview
        )
        self.fit_button.pack(side=LEFT, padx=4)
        Label(preview_toolbar, textvariable=self.zoom_status, width=7).pack(side=LEFT, padx=4)
        direction_toolbar = Frame(left, padx=8, pady=2)
        direction_toolbar.pack(fill=X, pady=(0, 3))
        Label(direction_toolbar, text="框选文字方向：").pack(side=LEFT)
        self.direction_menu = OptionMenu(
            direction_toolbar,
            self.direction_option,
            *DIRECTION_OPTIONS.keys(),
            command=self._direction_changed,
        )
        self.direction_menu.configure(width=18, state="disabled")
        self.direction_menu.pack(side=LEFT, padx=4)
        Label(direction_toolbar, text="自动检测只增加轻量分类，不会跑四次 OCR", fg="#555555").pack(side=LEFT, padx=6)
        self.preview_canvas = Canvas(left, bg="#202020", highlightthickness=0, cursor="crosshair")
        self.preview_canvas.pack(fill=BOTH, expand=True, padx=8, pady=(0, 8))
        self.preview_canvas.create_text(260, 270, text="尚未选择图片或 PDF", fill="#dddddd")
        self.preview_canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.preview_canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.preview_canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.preview_canvas.bind("<Configure>", self._on_canvas_resize)

        right = Frame(content, bd=1, relief="solid")
        right.pack(side=RIGHT, fill=BOTH, expand=True, padx=(6, 0))
        Label(right, text="识别结果（置信度  文字）", anchor="w", padx=10, pady=8).pack(fill=X)
        self.result_text = ScrolledText(right, wrap="word", font=("Microsoft YaHei UI", 11), padx=10, pady=10)
        self.result_text.pack(fill=BOTH, expand=True, padx=8, pady=(0, 8))

        Label(self.root, textvariable=self.status, anchor="w", relief="sunken", padx=10, pady=5).pack(fill=X)

    def choose_input(self) -> None:
        selected = filedialog.askopenfilename(title="选择待识别图片或 PDF", filetypes=SUPPORTED_INPUTS)
        if not selected:
            return
        self.input_path = Path(selected)
        self._show_preview(self.input_path)
        if self.input_path.suffix.lower() == ".pdf":
            pages = self.pdf_page_count or 0
            self.status.set(f"已选择：{self.input_path.name}｜PDF 共 {pages} 页；可在预览中拖动框选")
            return
        size_text = (
            f"{self.original_size[0]}×{self.original_size[1]}"
            if self.original_size
            else "未知尺寸"
        )
        self.status.set(f"已选择：{self.input_path.name}｜原图 {size_text}（识别使用原图）")

    def _show_preview(self, path: Path) -> None:
        try:
            if path.suffix.lower() == ".pdf":
                import fitz

                with fitz.open(path) as document:
                    self.pdf_page_count = document.page_count
                    if document.page_count == 0:
                        raise ValueError("PDF 没有可识别的页面")
                self.original_size = None
                self.pdf_page_index = 0
                self.pdf_selection = None
                self._set_interaction_mode("select", announce=False)
                self._render_pdf_page()
                return
            else:
                self.pdf_page_count = None
                self.pdf_page_index = 0
                self.pdf_selection = None
                self.preview_pdf_size = None
                self._set_interaction_mode("pan", announce=False)
                with Image.open(path) as source:
                    self.original_size = source.size
                    image = source.convert("RGB")
                image.thumbnail((int(PREVIEW_MAX_SIDE), int(PREVIEW_MAX_SIDE)), Image.Resampling.LANCZOS)
            self._display_preview_image(image)
            self._update_pdf_controls()
        except Exception as exc:
            self.original_size = None
            self.preview_source_image = None
            self.preview_image_bounds = None
            self.preview_photo = None
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(260, 270, text=f"文件预览失败\n{exc}", fill="#dddddd")
            self._update_pdf_controls()

    def _display_preview_image(self, image: Image.Image) -> None:
        self.preview_source_image = image.copy()
        self.preview_zoom = 1.0
        self.preview_pan = (0.0, 0.0)
        self._redraw_preview()

    def _redraw_preview(self) -> None:
        if self.preview_source_image is None:
            return
        self.root.update_idletasks()
        measured_width = self.preview_canvas.winfo_width()
        measured_height = self.preview_canvas.winfo_height()
        canvas_width = measured_width if measured_width > 1 else 520
        canvas_height = measured_height if measured_height > 1 else 540
        source_width, source_height = self.preview_source_image.size
        fit_scale = min(
            max(canvas_width - 16, 1) / source_width,
            max(canvas_height - 16, 1) / source_height,
        )
        display_scale = fit_scale * self.preview_zoom
        display_width = max(1, int(round(source_width * display_scale)))
        display_height = max(1, int(round(source_height * display_scale)))
        display = self.preview_source_image.resize(
            (display_width, display_height),
            Image.Resampling.LANCZOS,
        )
        self.preview_photo = ImageTk.PhotoImage(display)
        offset_x = (canvas_width - display.width) / 2 + self.preview_pan[0]
        offset_y = (canvas_height - display.height) / 2 + self.preview_pan[1]
        self.preview_image_bounds = (
            offset_x,
            offset_y,
            offset_x + display.width,
            offset_y + display.height,
        )
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(
            offset_x,
            offset_y,
            image=self.preview_photo,
            anchor="nw",
            tags=("preview_content",),
        )
        self.selection_rect_id = None
        self._draw_saved_selection()
        self.zoom_status.set(f"{self.preview_zoom:.0%}")

    def _draw_saved_selection(self) -> None:
        if self.pdf_selection is None or self.preview_image_bounds is None or self.preview_pdf_size is None:
            return
        left, top, right, bottom = self.preview_image_bounds
        page_width, page_height = self.preview_pdf_size
        x1, y1, x2, y2 = self.pdf_selection
        canvas_x1 = left + x1 / page_width * (right - left)
        canvas_y1 = top + y1 / page_height * (bottom - top)
        canvas_x2 = left + x2 / page_width * (right - left)
        canvas_y2 = top + y2 / page_height * (bottom - top)
        self.selection_rect_id = self.preview_canvas.create_rectangle(
            canvas_x1,
            canvas_y1,
            canvas_x2,
            canvas_y2,
            outline="#ff3b30",
            width=3,
            tags=("preview_content", "selection"),
        )

    def zoom_in(self) -> None:
        self._zoom_preview(1.25)

    def zoom_out(self) -> None:
        self._zoom_preview(0.8)

    def fit_preview(self) -> None:
        if self.preview_source_image is None:
            return
        self.preview_zoom = 1.0
        self.preview_pan = (0.0, 0.0)
        self._redraw_preview()
        self.status.set("预览已恢复为适应窗口大小")

    def _zoom_preview(self, factor: float, focus: tuple[float, float] | None = None) -> None:
        if self.preview_source_image is None or self.preview_image_bounds is None:
            return
        old_zoom = self.preview_zoom
        new_zoom = min(max(old_zoom * factor, 0.25), 6.0)
        if abs(new_zoom - old_zoom) < 1e-9:
            return

        canvas_width = max(self.preview_canvas.winfo_width(), 1)
        canvas_height = max(self.preview_canvas.winfo_height(), 1)
        focus_x, focus_y = focus or (canvas_width / 2, canvas_height / 2)
        left, top, right, bottom = self.preview_image_bounds
        relative_x = (focus_x - left) / (right - left)
        relative_y = (focus_y - top) / (bottom - top)

        self.preview_zoom = new_zoom
        self._redraw_preview()
        if self.preview_image_bounds is None:
            return
        new_left, new_top, new_right, new_bottom = self.preview_image_bounds
        point_x = new_left + relative_x * (new_right - new_left)
        point_y = new_top + relative_y * (new_bottom - new_top)
        self._move_preview(focus_x - point_x, focus_y - point_y)
        self.zoom_status.set(f"{self.preview_zoom:.0%}")
        self.status.set(f"预览缩放：{self.preview_zoom:.0%}；可切换到“拖动画面”查看其他位置")

    def _move_preview(self, delta_x: float, delta_y: float) -> None:
        if self.preview_image_bounds is None:
            return
        self.preview_pan = (
            self.preview_pan[0] + delta_x,
            self.preview_pan[1] + delta_y,
        )
        left, top, right, bottom = self.preview_image_bounds
        self.preview_image_bounds = (
            left + delta_x,
            top + delta_y,
            right + delta_x,
            bottom + delta_y,
        )
        self.preview_canvas.move("preview_content", delta_x, delta_y)

    def _on_mouse_wheel(self, event) -> None:
        factor = 1.25 if event.delta > 0 else 0.8
        self._zoom_preview(factor, (event.x, event.y))

    def _on_canvas_resize(self, _event) -> None:
        if self.preview_source_image is None:
            return
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(120, self._finish_canvas_resize)

    def _finish_canvas_resize(self) -> None:
        self.resize_job = None
        self._redraw_preview()

    def _render_pdf_page(self) -> None:
        if self.input_path is None or self.input_path.suffix.lower() != ".pdf":
            return
        import fitz

        with fitz.open(self.input_path) as document:
            page = document.load_page(self.pdf_page_index)
            self.preview_pdf_size = (page.rect.width, page.rect.height)
            scale = min(1.5, PREVIEW_MAX_SIDE / max(page.rect.width, page.rect.height))
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        self._display_preview_image(image)
        self._update_pdf_controls()

    def _update_pdf_controls(self) -> None:
        is_pdf = self.input_path is not None and self.input_path.suffix.lower() == ".pdf"
        page_count = self.pdf_page_count or 0
        self.page_status.set(f"{self.pdf_page_index + 1}/{page_count}" if is_pdf and page_count else "")
        self.previous_page_button.configure(
            state="normal" if is_pdf and self.pdf_page_index > 0 else "disabled"
        )
        self.next_page_button.configure(
            state="normal" if is_pdf and self.pdf_page_index + 1 < page_count else "disabled"
        )
        self.clear_selection_button.configure(
            state="normal" if is_pdf and self.pdf_selection is not None else "disabled"
        )
        has_preview = self.preview_source_image is not None
        preview_state = "normal" if has_preview else "disabled"
        self.pan_mode_button.configure(state=preview_state)
        self.select_mode_button.configure(state="normal" if is_pdf else "disabled")
        self.zoom_in_button.configure(state=preview_state)
        self.zoom_out_button.configure(state=preview_state)
        self.fit_button.configure(state=preview_state)
        self.direction_menu.configure(state="normal" if is_pdf else "disabled")
        self.select_mode_button.configure(relief="sunken" if self.interaction_mode == "select" else "raised")
        self.pan_mode_button.configure(relief="sunken" if self.interaction_mode == "pan" else "raised")

    def previous_pdf_page(self) -> None:
        self._change_pdf_page(-1)

    def next_pdf_page(self) -> None:
        self._change_pdf_page(1)

    def _change_pdf_page(self, offset: int) -> None:
        page_count = self.pdf_page_count or 0
        new_index = self.pdf_page_index + offset
        if not 0 <= new_index < page_count:
            return
        self.pdf_page_index = new_index
        self.pdf_selection = None
        self._render_pdf_page()
        self.status.set(f"当前第 {self.pdf_page_index + 1} 页；拖动鼠标可框选识别区域")

    def _clamp_to_preview(self, x: float, y: float) -> tuple[float, float] | None:
        if self.preview_image_bounds is None:
            return None
        left, top, right, bottom = self.preview_image_bounds
        return min(max(x, left), right), min(max(y, top), bottom)

    def _set_interaction_mode(self, mode: str, announce: bool = True) -> None:
        self.interaction_mode = mode
        self.pan_last = None
        self.selection_start = None
        self.preview_canvas.configure(cursor="fleur" if mode == "pan" else "crosshair")
        if hasattr(self, "select_mode_button"):
            self._update_pdf_controls()
        if not announce:
            return
        if mode == "pan":
            self.status.set("拖动画面模式：按住鼠标左键移动预览；滚轮可缩放")
        else:
            self.status.set("框选区域模式：按住鼠标左键拖出红框；滚轮可缩放")

    def _direction_changed(self, _value: str) -> None:
        angle = DIRECTION_OPTIONS[self.direction_option.get()]
        if angle is None:
            self.status.set("框选文字方向已设为自动检测；轻量分类后只运行一次 OCR")
        else:
            self.status.set(f"框选文字方向已设为手动旋转 {angle}°；只运行一次 OCR")

    def _on_canvas_press(self, event) -> None:
        if self.interaction_mode == "pan":
            if self.preview_source_image is not None:
                self.pan_last = (event.x, event.y)
            return
        self._start_pdf_selection(event)

    def _on_canvas_drag(self, event) -> None:
        if self.interaction_mode == "pan":
            if self.pan_last is None:
                return
            last_x, last_y = self.pan_last
            self._move_preview(event.x - last_x, event.y - last_y)
            self.pan_last = (event.x, event.y)
            return
        self._drag_pdf_selection(event)

    def _on_canvas_release(self, event) -> None:
        if self.interaction_mode == "pan":
            self.pan_last = None
            return
        self._finish_pdf_selection(event)

    def _start_pdf_selection(self, event) -> None:
        if self.input_path is None or self.input_path.suffix.lower() != ".pdf":
            return
        point = self._clamp_to_preview(event.x, event.y)
        if point is None:
            return
        self.clear_pdf_selection(redraw=False)
        self.selection_start = point
        self.selection_rect_id = self.preview_canvas.create_rectangle(
            point[0],
            point[1],
            point[0],
            point[1],
            outline="#ff3b30",
            width=3,
            tags=("preview_content", "selection"),
        )

    def _drag_pdf_selection(self, event) -> None:
        if self.selection_start is None or self.selection_rect_id is None:
            return
        point = self._clamp_to_preview(event.x, event.y)
        if point is None:
            return
        self.preview_canvas.coords(
            self.selection_rect_id,
            self.selection_start[0],
            self.selection_start[1],
            point[0],
            point[1],
        )

    def _finish_pdf_selection(self, event) -> None:
        if self.selection_start is None or self.preview_image_bounds is None or self.preview_pdf_size is None:
            return
        point = self._clamp_to_preview(event.x, event.y)
        if point is None:
            return
        start_x, start_y = self.selection_start
        end_x, end_y = point
        self.selection_start = None
        if abs(end_x - start_x) < 8 or abs(end_y - start_y) < 8:
            self.clear_pdf_selection()
            self.status.set("框选区域太小，请重新拖动选择")
            return

        left, top, right, bottom = self.preview_image_bounds
        page_width, page_height = self.preview_pdf_size
        x1, x2 = sorted((start_x, end_x))
        y1, y2 = sorted((start_y, end_y))
        self.pdf_selection = (
            (x1 - left) / (right - left) * page_width,
            (y1 - top) / (bottom - top) * page_height,
            (x2 - left) / (right - left) * page_width,
            (y2 - top) / (bottom - top) * page_height,
        )
        self._update_pdf_controls()
        self.status.set(f"已框选第 {self.pdf_page_index + 1} 页区域；开始识别时仅处理红框内容")

    def clear_pdf_selection(self, redraw: bool = True) -> None:
        self.pdf_selection = None
        self.selection_start = None
        if self.selection_rect_id is not None:
            self.preview_canvas.delete(self.selection_rect_id)
            self.selection_rect_id = None
        self._update_pdf_controls()
        if redraw and self.input_path is not None and self.input_path.suffix.lower() == ".pdf":
            self.status.set(f"已清除框选；开始识别将处理整份 PDF（共 {self.pdf_page_count or 0} 页）")

    def start_recognition(self) -> None:
        if self.running:
            return
        if self.input_path is None:
            messagebox.showinfo("提示", "请先选择一张图片或 PDF。")
            return

        self.running = True
        self.recognize_button.configure(state="disabled")
        self.result_text.delete("1.0", END)
        input_path = self.input_path
        page_index = self.pdf_page_index
        selection = self.pdf_selection
        direction_angle = DIRECTION_OPTIONS[self.direction_option.get()]
        if input_path.suffix.lower() == ".pdf" and selection is not None:
            direction_text = "自动检测方向" if direction_angle is None else f"旋转 {direction_angle}°"
            self.status.set(f"正在识别第 {page_index + 1} 页框选区域（{direction_text}，单次 OCR）……")
        else:
            self.status.set("正在准备 PaddleOCR；首次运行可能需要下载模型到 D 盘……")
        threading.Thread(
            target=self._recognize_worker,
            args=(input_path, page_index, selection, direction_angle),
            daemon=True,
        ).start()

    @staticmethod
    def _render_pdf_region(
        pdf_path: Path,
        page_index: int,
        selection: tuple[float, float, float, float],
    ) -> Path:
        import fitz

        output_dir = BASE_DIR / "tmp" / "pdf_regions"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"page_{page_index + 1}_{time.time_ns()}.png"
        with fitz.open(pdf_path) as document:
            page = document.load_page(page_index)
            clip = fitz.Rect(*selection) & page.rect
            if clip.is_empty or clip.width < 1 or clip.height < 1:
                raise ValueError("框选区域无效，请重新选择")
            if clip.width * clip.height * 9 > 20_000_000:
                raise MemoryError("框选区域超过 2000 万像素，请缩小框选范围")
            pixmap = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0), clip=clip, alpha=False)
            pixmap.save(output_path)
        return output_path

    @staticmethod
    def _create_direction_variant(source_path: Path, angle: int) -> Path:
        if angle == 0:
            return source_path
        operations = {
            90: Image.Transpose.ROTATE_90,
            180: Image.Transpose.ROTATE_180,
            270: Image.Transpose.ROTATE_270,
        }
        target = source_path.with_name(f"{source_path.stem}_rot{angle}.png")
        with Image.open(source_path) as source:
            source.convert("RGB").transpose(operations[angle]).save(target, quality=100)
        return target

    def _recognize_worker(
        self,
        input_path: Path,
        page_index: int,
        selection: tuple[float, float, float, float] | None,
        direction_angle: int | None,
    ) -> None:
        with self.processing_lock:
            self._run_recognition(input_path, page_index, selection, direction_angle)

    def _run_recognition(
        self,
        input_path: Path,
        page_index: int,
        selection: tuple[float, float, float, float] | None,
        direction_angle: int | None,
    ) -> None:
        started = time.perf_counter()
        temporary_regions: list[Path] = []
        try:
            if self.ocr is None:
                from paddleocr import PaddleOCR

                MODEL_CACHE.mkdir(parents=True, exist_ok=True)
                self.ocr = PaddleOCR(
                    text_detection_model_name="PP-OCRv6_medium_det",
                    text_recognition_model_name="PP-OCRv6_medium_rec",
                    enable_mkldnn=False,
                    use_doc_orientation_classify=True,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )

            is_pdf = input_path.suffix.lower() == ".pdf"
            selected_region = is_pdf and selection is not None
            if selected_region:
                temporary_region = self._render_pdf_region(input_path, page_index, selection)
                temporary_regions.append(temporary_region)
                manual_angle = direction_angle or 0
                predict_path = self._create_direction_variant(temporary_region, manual_angle)
                if predict_path != temporary_region:
                    temporary_regions.append(predict_path)
            else:
                predict_path = input_path
            results = self.ocr.predict(
                str(predict_path),
                use_doc_orientation_classify=selected_region and direction_angle is None,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            if self.multilingual_processor is None:
                self.multilingual_processor = MultilingualOCRProcessor()

            lines: list[str] = []
            text_count = 0
            page_count = 0
            reviewed_count = 0
            if selected_region:
                direction_text = "自动检测方向" if direction_angle is None else f"旋转 {direction_angle}°"
                lines.append(f"===== 第 {page_index + 1} 页（框选区域，{direction_text}） =====")
            for result_index, page in enumerate(results):
                records = self.multilingual_processor.refine_results([page])[0]
                page_count += 1
                if is_pdf and not selected_region:
                    result_page_index = page.get("page_index")
                    page_number = int(result_page_index) + 1 if result_page_index is not None else result_index + 1
                    if lines:
                        lines.append("")
                    lines.append(f"===== 第 {page_number} 页 =====")
                for record in records:
                    lines.append(record.output_line())
                    text_count += 1
                    reviewed_count += int(record.reviewed)
                del page, records

            elapsed = time.perf_counter() - started
            if text_count:
                output = "\n".join(lines)
                if selected_region:
                    direction_text = "自动检测方向" if direction_angle is None else f"旋转 {direction_angle}°"
                    page_text = f"第 {page_index + 1} 页框选区域（{direction_text}），"
                else:
                    page_text = f"{page_count} 页，" if is_pdf else ""
                review_text = f"，二次复核 {reviewed_count} 条" if reviewed_count else ""
                status = f"识别完成：{page_text}{text_count} 条文字{review_text}，用时 {elapsed:.2f} 秒"
            else:
                output = "未识别到文字。请尝试更清晰、文字更大的图片或 PDF。"
                status = f"识别完成：未检测到文字，用时 {elapsed:.2f} 秒"
            self.root.after(0, self._finish_recognition, output, status)
        except Exception as exc:
            hint = (
                "识别失败。首次运行需要联网下载高精度模型，模型缓存位置：\n"
                f"{MODEL_CACHE}\n\n错误信息：\n{type(exc).__name__}: {exc}"
            )
            self.root.after(0, self._finish_recognition, hint, "识别失败，请查看右侧错误信息")
        finally:
            for temporary_region in temporary_regions:
                try:
                    temporary_region.unlink(missing_ok=True)
                except OSError:
                    pass

    def _finish_recognition(self, output: str, status: str) -> None:
        self.result_text.delete("1.0", END)
        self.result_text.insert("1.0", output)
        self.status.set(status)
        self.recognize_button.configure(state="normal")
        self.running = False

    def copy_text(self) -> None:
        text = self.result_text.get("1.0", END).strip()
        if not text:
            messagebox.showinfo("提示", "当前没有可复制的识别结果。")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status.set("识别结果已复制到剪贴板")

    def save_text(self) -> None:
        text = self.result_text.get("1.0", END).strip()
        if not text:
            messagebox.showinfo("提示", "当前没有可保存的识别结果。")
            return
        default_name = f"{self.input_path.stem}_ocr.txt" if self.input_path else "ocr_result.txt"
        target = filedialog.asksaveasfilename(
            title="保存识别结果",
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[("文本文件", "*.txt")],
        )
        if not target:
            return
        Path(target).write_text(text, encoding="utf-8")
        self.status.set(f"结果已保存：{target}")

    def open_compare_window(self) -> None:
        from ocr_compare_demo import OCRCompareWindow

        window = OCRCompareWindow(self.root, self)
        self.compare_windows.append(window)

    def close(self) -> None:
        self.close_requested = True
        self.root.withdraw()
        for window in list(self.compare_windows):
            window.close()
        if self.running or any(window.running for window in self.compare_windows):
            self.root.after(100, self.close)
            return
        self.root.destroy()


def main() -> None:
    root = Tk()
    OCRDemo(root)
    root.mainloop()


if __name__ == "__main__":
    main()
