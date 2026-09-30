from __future__ import annotations

import threading
import time
from pathlib import Path
from tkinter import (
    BOTH,
    END,
    HORIZONTAL,
    LEFT,
    RIGHT,
    VERTICAL,
    X,
    Button,
    Canvas,
    Frame,
    Label,
    LabelFrame,
    PanedWindow,
    StringVar,
    Toplevel,
    filedialog,
    messagebox,
)
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageTk

from ocr_demo import BASE_DIR, PREVIEW_MAX_SIDE, SUPPORTED_INPUTS
from experiment_core import ComparisonEngine


from comparison_algorithms import ComparisonAlgorithms


class RegionPane:
    def __init__(
        self,
        parent,
        title: str,
        selection_color: str,
        status_callback,
        selection_callback=None,
    ) -> None:
        self.title = title
        self.selection_color = selection_color
        self.status_callback = status_callback
        self.selection_callback = selection_callback
        self.path: Path | None = None
        self.page_count = 0
        self.page_index = 0
        self.reference_size: tuple[float, float] | None = None
        self.selection: tuple[float, float, float, float] | None = None
        self.detected_selections: list[tuple[float, float, float, float]] = []
        self.difference_selections_by_page: dict[
            int,
            list[tuple[float, float, float, float]],
        ] = {}
        self.text_selections_by_page: dict[
            int,
            list[tuple[float, float, float, float]],
        ] = {}
        self.source_image: Image.Image | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.image_bounds: tuple[float, float, float, float] | None = None
        self.zoom = 1.0
        self.pan = (0.0, 0.0)
        self.mode = "select"
        self.selection_start: tuple[float, float] | None = None
        self.selection_rect_id: int | None = None
        self.pan_last: tuple[float, float] | None = None
        self.resize_job = None
        self.busy = False
        self._button_states = []
        self.page_status = StringVar(value="")
        self.zoom_status = StringVar(value="100%")
        self.file_status = StringVar(value="尚未选择文件")

        self.frame = LabelFrame(parent, text=title, padx=6, pady=6)
        self.frame.pack(side=LEFT, fill=BOTH, expand=True, padx=4)
        self._build_ui()

    def _build_ui(self) -> None:
        file_row = Frame(self.frame)
        file_row.pack(fill=X, pady=(0, 4))
        Button(file_row, text="选择文件", width=9, command=self.choose_file).pack(side=LEFT, padx=(0, 4))
        self.previous_button = Button(file_row, text="上一页", width=6, state="disabled", command=lambda: self.change_page(-1))
        self.previous_button.pack(side=LEFT, padx=2)
        self.next_button = Button(file_row, text="下一页", width=6, state="disabled", command=lambda: self.change_page(1))
        self.next_button.pack(side=LEFT, padx=2)
        Label(file_row, textvariable=self.page_status, width=7).pack(side=LEFT, padx=2)
        Label(file_row, textvariable=self.file_status, anchor="w").pack(side=LEFT, fill=X, expand=True, padx=4)

        tools = Frame(self.frame)
        tools.pack(fill=X, pady=(0, 4))
        self.select_button = Button(tools, text="框选", width=7, command=lambda: self.set_mode("select"))
        self.select_button.pack(side=LEFT, padx=(0, 2))
        self.pan_button = Button(tools, text="拖动", width=7, command=lambda: self.set_mode("pan"))
        self.pan_button.pack(side=LEFT, padx=2)
        Button(tools, text="缩小 −", width=7, command=lambda: self.zoom_by(0.8)).pack(side=LEFT, padx=(8, 2))
        Button(tools, text="放大 +", width=7, command=lambda: self.zoom_by(1.25)).pack(side=LEFT, padx=2)
        Button(tools, text="适应", width=6, command=self.fit).pack(side=LEFT, padx=2)
        Button(tools, text="清除框选", width=8, command=self.clear_selection).pack(side=LEFT, padx=2)
        Label(tools, textvariable=self.zoom_status, width=7).pack(side=LEFT, padx=2)

        self.canvas = Canvas(self.frame, bg="#202020", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=BOTH, expand=True)
        self.canvas.create_text(240, 220, text="请选择 PDF 或图片", fill="#dddddd")
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Configure>", self._on_resize)
        self.set_mode("select", announce=False)

    def choose_file(self) -> None:
        if self.busy:
            return
        selected = filedialog.askopenfilename(title=f"选择{self.title}", filetypes=SUPPORTED_INPUTS)
        if selected:
            self.load_path(Path(selected))

    def load_path(self, path: Path) -> None:
        if self.busy:
            return
        self.path = path
        self.page_index = 0
        self.selection = None
        self.detected_selections = []
        self.difference_selections_by_page = {}
        self.text_selections_by_page = {}
        try:
            if path.suffix.lower() == ".pdf":
                import fitz

                with fitz.open(path) as document:
                    self.page_count = document.page_count
                    if self.page_count == 0:
                        raise ValueError("PDF 没有页面")
                self._load_pdf_page()
            else:
                self.page_count = 1
                with Image.open(path) as source:
                    image = source.convert("RGB")
                self.reference_size = (float(image.width), float(image.height))
                image.thumbnail((int(PREVIEW_MAX_SIDE), int(PREVIEW_MAX_SIDE)), Image.Resampling.LANCZOS)
                self._set_source_image(image)
            self.file_status.set(path.name)
            self._update_page_controls()
            self.status_callback(f"{self.title}已加载：{path.name}；请框选需要比较的区域")
        except Exception as exc:
            self.source_image = None
            self.reference_size = None
            self.canvas.delete("all")
            self.canvas.create_text(240, 220, text=f"预览失败\n{exc}", fill="#dddddd")
            self.status_callback(f"{self.title}加载失败：{exc}")

    def _load_pdf_page(self) -> None:
        if self.path is None:
            return
        import fitz

        with fitz.open(self.path) as document:
            page = document.load_page(self.page_index)
            self.reference_size = (page.rect.width, page.rect.height)
            preview_scale = min(1.5, PREVIEW_MAX_SIDE / max(page.rect.width, page.rect.height))
            pixmap = page.get_pixmap(matrix=fitz.Matrix(preview_scale, preview_scale), alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        self._set_source_image(image)

    def set_busy(self, busy: bool) -> None:
        if busy == self.busy:
            return
        self.busy = busy
        self.selection_start = None
        self.pan_last = None
        if busy:
            pending = [self.frame]
            while pending:
                widget = pending.pop()
                pending.extend(widget.winfo_children())
                if isinstance(widget, Button):
                    self._button_states.append((widget, str(widget.cget("state"))))
                    widget.configure(state="disabled")
        else:
            for widget, state in self._button_states:
                widget.configure(state=state)
            self._button_states.clear()
            self._update_page_controls()

    def _set_source_image(self, image: Image.Image) -> None:
        self.source_image = image.copy()
        self.zoom = 1.0
        self.pan = (0.0, 0.0)
        self._redraw()

    def _redraw(self) -> None:
        if self.source_image is None:
            return
        self.frame.update_idletasks()
        canvas_width = self.canvas.winfo_width() if self.canvas.winfo_width() > 1 else 620
        canvas_height = self.canvas.winfo_height() if self.canvas.winfo_height() > 1 else 480
        source_width, source_height = self.source_image.size
        fit_scale = min(
            max(canvas_width - 12, 1) / source_width,
            max(canvas_height - 12, 1) / source_height,
        )
        scale = fit_scale * self.zoom
        display_size = (
            max(1, int(round(source_width * scale))),
            max(1, int(round(source_height * scale))),
        )
        display = self.source_image.resize(display_size, Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(display)
        left = (canvas_width - display.width) / 2 + self.pan[0]
        top = (canvas_height - display.height) / 2 + self.pan[1]
        self.image_bounds = (left, top, left + display.width, top + display.height)
        self.canvas.delete("all")
        self.canvas.create_image(left, top, image=self.preview_photo, anchor="nw", tags=("content",))
        self.selection_rect_id = None
        self._draw_selection()
        self.zoom_status.set(f"{self.zoom:.0%}")

    def _draw_selection(self) -> None:
        if self.image_bounds is None or self.reference_size is None:
            return
        left, top, right, bottom = self.image_bounds
        ref_width, ref_height = self.reference_size
        selections = self.detected_selections or ([self.selection] if self.selection is not None else [])
        for index, selection in enumerate(selections, start=1):
            x1, y1, x2, y2 = selection
            canvas_x1 = left + x1 / ref_width * (right - left)
            canvas_y1 = top + y1 / ref_height * (bottom - top)
            canvas_x2 = left + x2 / ref_width * (right - left)
            canvas_y2 = top + y2 / ref_height * (bottom - top)
            rectangle_id = self.canvas.create_rectangle(
                canvas_x1,
                canvas_y1,
                canvas_x2,
                canvas_y2,
                outline=self.selection_color,
                width=3,
                dash=(8, 4),
                tags=("content", "selection"),
            )
            if index == 1:
                self.selection_rect_id = rectangle_id
            if self.detected_selections:
                self.canvas.create_text(
                    canvas_x1 + 5,
                    canvas_y1 + 5,
                    text=str(index),
                    fill="#111111",
                    font=("Microsoft YaHei UI", 11, "bold"),
                    anchor="nw",
                    tags=("content", "selection"),
                )

        # 图片检测得到的局部差异使用红色实线框，叠加在原有黄色自动定位框
        # 上。这里只增加显示图层，不改变用户原来的框选和拖拽逻辑。
        difference_selections = self.difference_selections_by_page.get(
            self.page_index,
            [],
        )
        for index, selection in enumerate(difference_selections, start=1):
            x1, y1, x2, y2 = selection
            canvas_x1 = left + x1 / ref_width * (right - left)
            canvas_y1 = top + y1 / ref_height * (bottom - top)
            canvas_x2 = left + x2 / ref_width * (right - left)
            canvas_y2 = top + y2 / ref_height * (bottom - top)
            self.canvas.create_rectangle(
                canvas_x1,
                canvas_y1,
                canvas_x2,
                canvas_y2,
                outline="#ff2d2d",
                width=3,
                tags=("content", "image_difference"),
            )
            self.canvas.create_text(
                canvas_x1 + 4,
                canvas_y1 + 4,
                text=f"D{index}",
                fill="#ff2d2d",
                font=("Microsoft YaHei UI", 10, "bold"),
                anchor="nw",
                tags=("content", "image_difference"),
            )

        # CNN 判定为 document_text 的候选框使用绿色实线显示。它们只用于
        # 检查分类和 OCR 输入范围，不替代用户框选或自动定位框。
        text_selections = self.text_selections_by_page.get(
            self.page_index,
            [],
        )
        for index, selection in enumerate(text_selections, start=1):
            x1, y1, x2, y2 = selection
            canvas_x1 = left + x1 / ref_width * (right - left)
            canvas_y1 = top + y1 / ref_height * (bottom - top)
            canvas_x2 = left + x2 / ref_width * (right - left)
            canvas_y2 = top + y2 / ref_height * (bottom - top)
            self.canvas.create_rectangle(
                canvas_x1,
                canvas_y1,
                canvas_x2,
                canvas_y2,
                outline="#18a558",
                width=2,
                tags=("content", "cnn_text_region"),
            )
            self.canvas.create_text(
                canvas_x1 + 3,
                canvas_y1 + 3,
                text=f"T{index}",
                fill="#08783c",
                font=("Microsoft YaHei UI", 9, "bold"),
                anchor="nw",
                tags=("content", "cnn_text_region"),
            )

    def set_difference_selections(
        self,
        selections_by_page: dict[
            int,
            list[tuple[float, float, float, float]],
        ],
    ) -> None:
        """设置图片差异框并在当前页原画布中重绘。"""

        self.difference_selections_by_page = selections_by_page
        self._redraw()

    def set_text_selections(
        self,
        selections_by_page: dict[
            int,
            list[tuple[float, float, float, float]],
        ],
    ) -> None:
        """设置 CNN 文字区域框并在当前PDF页重绘。"""

        self.text_selections_by_page = selections_by_page
        self._redraw()

    def _update_page_controls(self) -> None:
        self.page_status.set(f"{self.page_index + 1}/{self.page_count}" if self.page_count else "")
        self.previous_button.configure(state="normal" if self.page_index > 0 else "disabled")
        self.next_button.configure(state="normal" if self.page_index + 1 < self.page_count else "disabled")

    def change_page(self, offset: int) -> None:
        if self.busy:
            return
        new_page = self.page_index + offset
        if not 0 <= new_page < self.page_count or self.path is None or self.path.suffix.lower() != ".pdf":
            return
        self.page_index = new_page
        self.selection = None
        self.detected_selections = []
        self._load_pdf_page()
        self._update_page_controls()
        self.status_callback(f"{self.title}切换到第 {self.page_index + 1} 页")

    def set_mode(self, mode: str, announce: bool = True) -> None:
        self.mode = mode
        self.pan_last = None
        self.selection_start = None
        self.canvas.configure(cursor="fleur" if mode == "pan" else "crosshair")
        self.select_button.configure(relief="sunken" if mode == "select" else "raised")
        self.pan_button.configure(relief="sunken" if mode == "pan" else "raised")
        if announce:
            action = "按住左键拖出红框" if mode == "select" else "按住左键移动画面"
            self.status_callback(f"{self.title}：{action}；滚轮可缩放")

    def fit(self) -> None:
        self.zoom = 1.0
        self.pan = (0.0, 0.0)
        self._redraw()

    def zoom_by(self, factor: float, focus: tuple[float, float] | None = None) -> None:
        if self.source_image is None or self.image_bounds is None:
            return
        old_zoom = self.zoom
        new_zoom = min(max(old_zoom * factor, 0.25), 8.0)
        if abs(new_zoom - old_zoom) < 1e-9:
            return
        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        focus_x, focus_y = focus or (canvas_width / 2, canvas_height / 2)
        left, top, right, bottom = self.image_bounds
        relative_x = (focus_x - left) / (right - left)
        relative_y = (focus_y - top) / (bottom - top)
        self.zoom = new_zoom
        self._redraw()
        if self.image_bounds is None:
            return
        left, top, right, bottom = self.image_bounds
        point_x = left + relative_x * (right - left)
        point_y = top + relative_y * (bottom - top)
        self._move(focus_x - point_x, focus_y - point_y)

    def _move(self, delta_x: float, delta_y: float) -> None:
        if self.image_bounds is None:
            return
        self.pan = (self.pan[0] + delta_x, self.pan[1] + delta_y)
        left, top, right, bottom = self.image_bounds
        self.image_bounds = (left + delta_x, top + delta_y, right + delta_x, bottom + delta_y)
        self.canvas.move("content", delta_x, delta_y)

    def clear_selection(self) -> None:
        if self.busy:
            return
        self.selection = None
        self.detected_selections = []
        self.difference_selections_by_page = {}
        self.text_selections_by_page = {}
        self.selection_start = None
        self.canvas.delete("selection")
        self.selection_rect_id = None
        self.status_callback(f"{self.title}框选已清除")

    def _clamp(self, x: float, y: float) -> tuple[float, float] | None:
        if self.image_bounds is None:
            return None
        left, top, right, bottom = self.image_bounds
        return min(max(x, left), right), min(max(y, top), bottom)

    def _on_press(self, event) -> None:
        if self.busy:
            return
        if self.mode == "pan":
            if self.source_image is not None:
                self.pan_last = (event.x, event.y)
            return
        point = self._clamp(event.x, event.y)
        if point is None:
            return
        self.clear_selection()
        self.selection_start = point
        self.selection_rect_id = self.canvas.create_rectangle(
            point[0],
            point[1],
            point[0],
            point[1],
            outline=self.selection_color,
            width=3,
            dash=(8, 4),
            tags=("content", "selection"),
        )

    def _on_drag(self, event) -> None:
        if self.mode == "pan":
            if self.pan_last is None:
                return
            last_x, last_y = self.pan_last
            self._move(event.x - last_x, event.y - last_y)
            self.pan_last = (event.x, event.y)
            return
        if self.selection_start is None or self.selection_rect_id is None:
            return
        point = self._clamp(event.x, event.y)
        if point is not None:
            self.canvas.coords(self.selection_rect_id, *self.selection_start, *point)

    def _on_release(self, event) -> None:
        if self.mode == "pan":
            self.pan_last = None
            return
        if self.selection_start is None or self.image_bounds is None or self.reference_size is None:
            return
        point = self._clamp(event.x, event.y)
        if point is None:
            return
        start_x, start_y = self.selection_start
        end_x, end_y = point
        self.selection_start = None
        if abs(end_x - start_x) < 8 or abs(end_y - start_y) < 8:
            self.clear_selection()
            self.status_callback(f"{self.title}框选太小，请重新选择")
            return
        left, top, right, bottom = self.image_bounds
        ref_width, ref_height = self.reference_size
        x1, x2 = sorted((start_x, end_x))
        y1, y2 = sorted((start_y, end_y))
        self.selection = (
            (x1 - left) / (right - left) * ref_width,
            (y1 - top) / (bottom - top) * ref_height,
            (x2 - left) / (right - left) * ref_width,
            (y2 - top) / (bottom - top) * ref_height,
        )
        if self.selection_callback is not None:
            self.selection_callback()
        self.status_callback(f"{self.title}第 {self.page_index + 1} 页框选完成")

    def _on_wheel(self, event) -> None:
        if self.busy:
            return
        self.zoom_by(1.25 if event.delta > 0 else 0.8, (event.x, event.y))

    def _on_resize(self, _event) -> None:
        if self.source_image is None:
            return
        if self.resize_job is not None:
            self.frame.after_cancel(self.resize_job)
        self.resize_job = self.frame.after(120, self._finish_resize)

    def _finish_resize(self) -> None:
        self.resize_job = None
        self._redraw()

    def render_selection(self, role: str) -> Path:
        if self.path is None or self.selection is None:
            raise ValueError(f"请先在{self.title}中框选区域")
        output_dir = BASE_DIR / "tmp" / "compare_regions"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{role}_{time.time_ns()}.png"
        if self.path.suffix.lower() == ".pdf":
            import fitz

            with fitz.open(self.path) as document:
                page = document.load_page(self.page_index)
                clip = fitz.Rect(*self.selection) & page.rect
                if clip.is_empty:
                    raise ValueError(f"{self.title}框选区域无效")
                pixmap = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0), clip=clip, alpha=False)
                pixmap.save(output_path)
        else:
            with Image.open(self.path) as source:
                image = source.convert("RGB")
                x1, y1, x2, y2 = self.selection
                crop = image.crop((round(x1), round(y1), round(x2), round(y2)))
                longest = max(crop.width, crop.height)
                scale = min(3.0, max(1.0, 2200.0 / max(longest, 1)))
                if scale > 1.01:
                    crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)
                crop.save(output_path, quality=100)
        return output_path

    def render_search_page(self, page_index: int) -> tuple[Image.Image, tuple[float, float]]:
        if self.path is None:
            raise ValueError(f"请先加载{self.title}")
        if self.path.suffix.lower() == ".pdf":
            import fitz

            with fitz.open(self.path) as document:
                page = document.load_page(page_index)
                reference_size = (page.rect.width, page.rect.height)
                # 3200px is enough for panel-level visual matching while keeping
                # the automatic search responsive on large imposition sheets.
                scale = min(1.5, 3200.0 / max(page.rect.width, page.rect.height))
                pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            return image, reference_size
        with Image.open(self.path) as source:
            image = source.convert("RGB")
        return image, (float(image.width), float(image.height))


class OCRCompareWindow(ComparisonAlgorithms):
    def __init__(self, parent, ocr_owner) -> None:
        self.ocr_owner = ocr_owner
        self.window = Toplevel(parent)
        self.window.title("PaddleOCR 双图区域文字对比")
        self.window.geometry("1500x900")
        self.window.minsize(1050, 700)
        self.status = StringVar(value="请分别加载原图和重排图，并在两边框选对应版面")
        self.image_score_status = StringVar(value="Image Check: --")
        self.ocr_score_status = StringVar(value="OCR Check: --")
        self.final_score_status = StringVar(value="Decision: 规则判定（未加权）")
        self.final_result_status = StringVar(value="Result: --")
        self.running = False
        self.selection_source: str | None = None
        if not hasattr(ocr_owner, "processing_lock"):
            ocr_owner.processing_lock = threading.Lock()
        self.engine = ComparisonEngine(owner=ocr_owner)
        self.close_requested = False
        self.last_result = None
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        actions = Frame(self.window, padx=10, pady=8)
        actions.pack(fill=X)
        self.compare_button = Button(actions, text="自动定位并对比", width=16, command=self.start_compare)
        self.compare_button.pack(side=LEFT, padx=(0, 6))
        Button(actions, text="清空结果", width=10, command=self.clear_result).pack(side=LEFT, padx=6)
        Label(actions, text="流程：任意一侧框选 → 双向自动定位；重排图会查找全部相似版面 → OCR 并对比", anchor="w").pack(
            side=LEFT, fill=X, expand=True, padx=10
        )

        # 保留原窗口的状态栏位置，但最终结论改为规则判定，不再将 OCR
        # 分数和图片分数加权成 Final Score。
        score_area = Frame(self.window, padx=10)
        score_area.pack(fill=X, pady=(0, 6))
        Label(
            score_area,
            textvariable=self.image_score_status,
            width=21,
            anchor="w",
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(side=LEFT, padx=(0, 12))
        Label(
            score_area,
            textvariable=self.ocr_score_status,
            width=19,
            anchor="w",
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(side=LEFT, padx=12)
        Label(
            score_area,
            textvariable=self.final_score_status,
            width=21,
            anchor="w",
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(side=LEFT, padx=12)
        self.final_result_label = Label(
            score_area,
            textvariable=self.final_result_status,
            width=19,
            anchor="w",
            fg="#444444",
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        self.final_result_label.pack(side=LEFT, padx=12)

        main_panes = PanedWindow(self.window, orient=VERTICAL, sashwidth=6)
        main_panes.pack(fill=BOTH, expand=True, padx=8, pady=(0, 6))
        preview_area = Frame(main_panes)
        main_panes.add(preview_area, stretch="always", minsize=390)
        result_area = LabelFrame(main_panes, text="OCR 与文字差异", padx=6, pady=6)
        main_panes.add(result_area, stretch="never", minsize=220)

        self.original_pane = RegionPane(
            preview_area,
            "原图（青色框）",
            "#00d8ff",
            self.set_status,
            lambda: self._set_selection_source("original"),
        )
        self.compare_pane = RegionPane(
            preview_area,
            "重排图（自动黄色框）",
            "#ffd60a",
            self.set_status,
            lambda: self._set_selection_source("rearranged"),
        )
        self.original_pane.frame.pack_forget()
        self.compare_pane.frame.pack_forget()
        preview_area.grid_rowconfigure(0, weight=1)
        preview_area.grid_columnconfigure(0, weight=1, uniform="preview")
        preview_area.grid_columnconfigure(1, weight=1, uniform="preview")
        self.original_pane.frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.compare_pane.frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self.result_text = ScrolledText(result_area, wrap="word", font=("Microsoft YaHei UI", 10), padx=8, pady=8)
        self.result_text.pack(fill=BOTH, expand=True)
        self.result_text.tag_configure(
            "ocr_error",
            foreground="#c62828",
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        self.result_text.tag_configure(
            "image_error",
            foreground="#d49a00",
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        Label(self.window, textvariable=self.status, anchor="w", relief="sunken", padx=8, pady=4).pack(fill=X)

    def set_status(self, text: str) -> None:
        self.status.set(text)

    def _set_selection_source(self, source: str) -> None:
        """记录用户最后手动框选的一侧，作为自动定位的模板来源。"""

        self.selection_source = source

    def clear_result(self) -> None:
        self.result_text.delete("1.0", END)
        self.image_score_status.set("Image Check: --")
        self.ocr_score_status.set("OCR Check: --")
        self.final_score_status.set("Decision: 规则判定（未加权）")
        self.final_result_status.set("Result: --")
        self.final_result_label.configure(fg="#444444")
        self.compare_pane.set_difference_selections({})

    def start_compare(self) -> None:
        if self.running:
            return
        if self.original_pane.path is None:
            messagebox.showinfo("提示", "请先在左侧加载原图。", parent=self.window)
            return
        if self.compare_pane.path is None:
            messagebox.showinfo("提示", "请先在右侧加载重排图。", parent=self.window)
            return
        source = self._resolve_selection_source()
        if source is None:
            messagebox.showinfo(
                "提示",
                "请在左侧原图或右侧重排图中框选需要比较的版面。",
                parent=self.window,
            )
            return
        self.running = True
        self.last_result = None
        self.original_pane.set_busy(True)
        self.compare_pane.set_busy(True)
        self.compare_button.configure(state="disabled")
        self.result_text.delete("1.0", END)
        self.image_score_status.set("Image Check: 检测中")
        self.ocr_score_status.set("OCR Check: 检测中")
        self.final_score_status.set("Decision: 规则判定（未加权）")
        self.final_result_status.set("Result: --")
        self.final_result_label.configure(fg="#444444")
        self.compare_pane.set_difference_selections({})
        self.original_pane.set_text_selections({})
        self.compare_pane.set_text_selections({})
        if source == "original":
            self.status.set("正在重排图中搜索原图框选版面的全部对应位置……")
        else:
            self.status.set("正在原图中反向定位，并搜索重排图中的其他相似版面……")
        threading.Thread(
            target=self._compare_worker,
            args=(source, self._snapshot_pane(self.original_pane), self._snapshot_pane(self.compare_pane)),
            daemon=True,
        ).start()

    def _resolve_selection_source(self) -> str | None:
        """确定本次由哪一侧的手动框选发起。

        自动定位完成后，两侧都可能保留框，因此优先采用用户最后一次手动
        框选的侧别；若该框已被清除，再退回到仍然存在的有效框选。
        """

        if (
            self.selection_source == "original"
            and self.original_pane.selection is not None
        ):
            return "original"
        if (
            self.selection_source == "rearranged"
            and self.compare_pane.selection is not None
        ):
            return "rearranged"
        if self.original_pane.selection is not None:
            return "original"
        if self.compare_pane.selection is not None:
            return "rearranged"
        return None

    @staticmethod
    def _snapshot_pane(pane: RegionPane) -> dict[str, object]:
        return {
            "path": str(pane.path),
            "page": pane.page_index,
            "selection": tuple(pane.selection) if pane.selection is not None else None,
        }

    def _compare_worker(
        self,
        source: str,
        source_input: dict[str, object],
        target_input: dict[str, object],
    ) -> None:
        output_dir = (
            BASE_DIR / "results"
            / f"integrated_compare_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
        )
        try:
            with self.ocr_owner.processing_lock:
                result = self.engine.compare_sample(
                    source_input,
                    target_input,
                    mode="fusion",
                    output_dir=output_dir,
                    selection_source=source,
                )
            self.window.after(0, self._display_comparison_result, result, source)
        except Exception as exc:
            self.window.after(
                0, self._finish_compare,
                f"对比失败：\n{type(exc).__name__}: {exc}", "对比失败", None,
            )

    def _display_comparison_result(self, result: dict[str, object], source: str) -> None:
        self.last_result = result
        if not result["status"]["completed"]:
            self._finish_compare(
                f"对比失败：\n{result['status']['error_message']}", "对比失败", None,
            )
            return
        display = result["display"]
        self._show_auto_matches(
            display["matches"], source, display["original_mapping"],
        )
        self.original_pane.set_text_selections(display["original_text"])
        self.compare_pane.set_text_selections(display["target_text"])
        self.compare_pane.set_difference_selections(display["difference_boxes"])
        decision = display["decision"]
        elapsed = result["timing"]["total_s"]
        status = (
            f"工业双图检测完成：双向定位到 {len(display['matches'])} 个重排版面；"
            f"最终 {decision['final_status']}；用时 {elapsed:.2f} 秒"
        )
        self._finish_compare(display["text"], status, decision)

    def close(self) -> None:
        self.close_requested = True
        if self.running:
            self.window.withdraw()
            return
        self.engine.close()
        self.window.destroy()
        windows = getattr(self.ocr_owner, "compare_windows", [])
        if self in windows:
            windows.remove(self)

    def _show_auto_matches(
        self,
        matches: list[dict[str, object]],
        source: str = "original",
        original_match: dict[str, object] | None = None,
    ) -> None:
        if source == "rearranged":
            if original_match is None:
                raise ValueError("反向定位缺少原图匹配结果")
            self._display_matches(self.original_pane, [original_match])

        page_matches = self._display_matches(self.compare_pane, matches)
        direction_text = (
            "已从原图定位重排图"
            if source == "original"
            else "已从重排图反向定位原图"
        )
        self.status.set(
            f"{direction_text}，并在重排图找到 {len(matches)} 个相似版面；"
            f"当前页显示 {len(page_matches)} 个黄色框；正在逐个 OCR……"
        )

    @staticmethod
    def _display_matches(
        pane: RegionPane,
        matches: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """在指定预览侧显示某一页的全部自动定位框。"""

        first_match = matches[0]
        page_index = int(first_match["page_index"])
        pane.page_index = page_index
        if pane.path is not None and pane.path.suffix.lower() == ".pdf":
            pane._load_pdf_page()
        page_matches = [match for match in matches if int(match["page_index"]) == page_index]
        pane.detected_selections = [match["selection"] for match in page_matches]
        pane.selection = pane.detected_selections[0]
        pane._redraw()
        pane._update_page_controls()
        return page_matches

    def _finish_compare(
        self,
        output: str,
        status: str,
        fusion: dict[str, object] | None = None,
    ) -> None:
        """在原窗口显示 OCR、图片和融合结果，并恢复按钮状态。"""

        self.result_text.delete("1.0", END)
        self.result_text.insert("1.0", output)
        self._highlight_result_errors()
        self.status.set(status)
        if fusion is None:
            self.image_score_status.set("Image Check: --")
            self.ocr_score_status.set("OCR Check: --")
            self.final_score_status.set("Decision: 规则判定（未加权）")
            self.final_result_status.set("Result: ERROR")
            self.final_result_label.configure(fg="#c62828")
        else:
            self.image_score_status.set(
                f"Image Check: {fusion['image_status']}"
            )
            self.ocr_score_status.set(
                f"OCR Check: {fusion['ocr_status']}"
            )
            self.final_score_status.set(
                "Decision: 规则判定（未加权）"
            )
            final_status = str(fusion["final_status"])
            self.final_result_status.set(f"Result: {final_status}")
            status_colors = {
                "PASS": "#16803a",
                "WARNING": "#d97706",
                "ERROR": "#c62828",
            }
            self.final_result_label.configure(
                fg=status_colors.get(final_status, "#444444")
            )
        self.compare_button.configure(state="normal")
        self.original_pane.set_busy(False)
        self.compare_pane.set_busy(False)
        self.running = False
        if self.close_requested:
            self.close()

    def _highlight_result_errors(self) -> None:
        """把结果开头的 OCR 错误标红、整版图片错误标黄。"""

        sections = (
            (
                "===== OCR 对比错误（红色） =====",
                "ocr_error",
                (
                    "===== 整版图片对比错误（黄色） =====",
                    "===== 规则判定结果 =====",
                ),
            ),
            (
                "===== 整版图片对比错误（黄色） =====",
                "image_error",
                ("===== 规则判定结果 =====",),
            ),
        )
        for header, tag, following_headers in sections:
            start = self.result_text.search(header, "1.0", stopindex=END)
            if not start:
                continue
            end = self.result_text.index(END)
            search_from = f"{start}+1c"
            for following_header in following_headers:
                candidate = self.result_text.search(
                    following_header,
                    search_from,
                    stopindex=END,
                )
                if candidate and self.result_text.compare(candidate, "<", end):
                    end = candidate
            self.result_text.tag_add(tag, start, end)
