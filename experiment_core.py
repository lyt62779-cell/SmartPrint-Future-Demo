"""GUI-independent execution of the legacy panel detection pipeline."""
from __future__ import annotations
import os
import math
import tempfile
import time
import threading
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
from PIL import Image
from gpu_runtime import configure_paddle_gpu_dll_paths
configure_paddle_gpu_dll_paths()
BASE_DIR = Path(__file__).resolve().parent
MODEL_CACHE = BASE_DIR / 'model_cache'
os.environ.setdefault('PADDLE_PDX_CACHE_HOME', str(MODEL_CACHE))
os.environ.setdefault('PADDLE_PDX_MODEL_SOURCE', 'bos')
os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
from comparison_algorithms import ComparisonAlgorithms
from cnn_region_pipeline import classify_detected_regions, consolidate_document_text_regions, crop_document_text_patches
from multilingual_ocr import OCRRecord
from image_diff import compare_images, load_image, resize_to_reference
from image_align import align_images
from experiment_config import ExperimentConfig

class ResourceMonitor:
    """Sample process-tree RSS. GPU attribution is unavailable on WDDM."""
    def __init__(self):
        self.peak = None
        self.samples = 0
        self.stop_event = threading.Event()
    def start(self):
        self._sample_once()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
    def _sample_once(self):
        try:
            import psutil
            parent = psutil.Process()
            processes = [parent]
            try:
                processes.extend(parent.children(recursive=True))
            except psutil.Error:
                pass
            total = 0
            observed = False
            for process in processes:
                try:
                    total += process.memory_info().rss
                    observed = True
                except psutil.Error:
                    pass
            if observed:
                self.peak = max(self.peak or 0, total / 1024**2)
                self.samples += 1
        except ImportError:
            return
        except psutil.Error:
            return
    def sample(self):
        while not self.stop_event.wait(.05):
            self._sample_once()
    def stop(self):
        self.stop_event.set()
        thread = getattr(self, 'thread', None)
        if thread is not None:
            thread.join(timeout=2)
        self._sample_once()


class InputError(ValueError):
    def __init__(self, message, error_type="unsupported_input"):
        super().__init__(message)
        self.error_type = error_type


def normalize_input(spec):
    if isinstance(spec, (str, Path)):
        spec = {"path": str(spec)}
    if not isinstance(spec, dict) or not isinstance(spec.get("path"), (str, Path)):
        raise InputError("Input must be a path or an object containing a path")
    if not str(spec["path"]).strip():
        raise InputError("Input path must not be empty")
    page = spec.get("page", 0)
    if isinstance(page, bool) or not isinstance(page, int) or page < 0:
        raise InputError("page must be a nonnegative zero-based integer")
    selection = spec.get("selection")
    if selection is not None:
        if (
            not isinstance(selection, (list, tuple)) or len(selection) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in selection)
            or selection[0] < 0 or selection[1] < 0
            or selection[2] <= selection[0] or selection[3] <= selection[1]
        ):
            raise InputError("selection must be a finite, positive xyxy rectangle")
        selection = list(selection)
    return {"path": str(Path(spec["path"]).resolve()), "page": page, "selection": selection}

class InputPane:
    """Private input snapshot and one-page search cache; never touches Tk state."""
    def __init__(self, spec, workspace, role, config):
        spec = normalize_input(spec)
        self.path = Path(spec['path']).resolve()
        self.page_index = int(spec.get('page', 0))
        self.selection = spec.get('selection')
        self.workspace, self.role, self.config = Path(workspace), role, config
        self.cache = {}
        if self.path.suffix.lower() not in {'.pdf','.png','.jpg','.jpeg','.bmp','.webp','.tif','.tiff'}:
            raise InputError('unsupported_input: '+str(self.path))
        is_pdf = self.path.suffix.lower() == '.pdf'
        try:
            if is_pdf:
                import fitz
                with fitz.open(self.path) as doc:
                    self.page_count = doc.page_count
                    if self.page_index >= self.page_count:
                        raise InputError("PDF page is outside the document")
                    page = doc.load_page(self.page_index)
                    self.reference_size = (page.rect.width, page.rect.height)
            else:
                if self.page_index != 0:
                    raise InputError("Image inputs support page 0 only")
                with Image.open(self.path) as im:
                    self.reference_size = im.size
                    self._check_budget(im.width, im.height)
                    if getattr(im, "n_frames", 1) != 1:
                        raise InputError("Multi-frame image inputs require explicit frame extraction")
                    im.verify()
                self.page_count = 1
        except (InputError, MemoryError):
            raise
        except Exception as exc:
            raise InputError(str(exc), "pdf_load_error" if is_pdf else "image_load_error") from exc
        if self.selection is None:
            self.selection = (0,0,*self.reference_size)
        self._validate_selection(self.reference_size)

    def _validate_selection(self, size):
        x1, y1, x2, y2 = self.selection
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > size[0] + 1e-6 or y2 > size[1] + 1e-6:
            raise InputError("selection must lie inside the selected page/image")

    def _check_budget(self, width, height):
        if width * height > self.config.max_roi_pixels:
            raise MemoryError("Image/ROI exceeds configured pixel safety budget; choose a smaller ROI/input")

    def close(self):
        for image, _size in self.cache.values():
            image.close()
        self.cache.clear()
    def render_selection(self, role):
        output = self.workspace / (role + '_' + str(time.time_ns()) + '.png')
        if self.path.suffix.lower() == '.pdf':
            import fitz
            with fitz.open(self.path) as doc:
                page = doc.load_page(self.page_index)
                self._validate_selection((page.rect.width, page.rect.height))
                clip = fitz.Rect(*self.selection) & page.rect
                if clip.is_empty: raise ValueError('empty PDF ROI')
                self._check_budget(math.ceil(clip.width * self.config.dpi_scale) + 1,
                                   math.ceil(clip.height * self.config.dpi_scale) + 1)
                pix = page.get_pixmap(matrix=fitz.Matrix(self.config.dpi_scale,self.config.dpi_scale),clip=clip,alpha=False)
                pix.save(output)
        else:
            with Image.open(self.path) as source:
                self._check_budget(source.width, source.height)
                self._validate_selection(source.size)
                rectangle = tuple(round(x) for x in self.selection)
                size = (rectangle[2] - rectangle[0], rectangle[3] - rectangle[1])
                if min(size) < 1:
                    raise InputError("ROI rounds to an empty pixel rectangle")
                scale = min(3.,max(1.,2200./max(size)))
                output_size = tuple(round(x * scale) for x in size)
                self._check_budget(*output_size)
                with source.crop(rectangle).convert('RGB') as crop:
                    if scale > 1.01:
                        with crop.resize(output_size, Image.Resampling.LANCZOS) as resized:
                            resized.save(output)
                    else:
                        crop.save(output)
        return output
    def render_search_page(self, page_index):
        if page_index in self.cache: return self.cache[page_index]
        if self.path.suffix.lower() == '.pdf':
            import fitz
            with fitz.open(self.path) as doc:
                page = doc.load_page(page_index)
                size = (page.rect.width,page.rect.height)
                scale = min(1.5,3200/max(size))
                pix = page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False)
                image = Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
        else:
            with Image.open(self.path) as opened: image = opened.convert('RGB')
            size = image.size
        self.close()
        self.cache[page_index] = (image,size)
        return image,size

class ComparisonEngine(ComparisonAlgorithms):
    def __init__(self, owner=None, classifier=None):
        self.ocr_owner = owner or SimpleNamespace(ocr=None,multilingual_processor=None)
        self.cnn_region_classifier = classifier
        self.ocr_batch_size = 8
        self.ocr_diagnostics = {}
    def close(self):
        if self.cnn_region_classifier is not None:
            self.cnn_region_classifier.close()
            self.cnn_region_classifier = None
    @contextmanager
    def stage(self, name, result):
        self.current_stage = name
        start = time.perf_counter()
        try: yield
        finally:
            key = name + '_s'
            result['timing'][key] = (result['timing'].get(key) or 0) + time.perf_counter()-start
    def recognize_panels(self, analysis_regions, original_mapping, matches, pipeline_directory):
        temporary_paths = []
        self.ocr_diagnostics = {"model_initialization_s": 0.0}
        route_started = time.perf_counter()
        classifier = self._cnn_classifier()
        if hasattr(classifier, "ensure_ready"):
            init_started = time.perf_counter()
            classifier.ensure_ready()
            self.ocr_diagnostics["model_initialization_s"] += time.perf_counter() - init_started
        routed_regions = [
            classify_detected_regions(path, classifier)
            for path in analysis_regions
        ]
        self.ocr_diagnostics["cnn_route_total_s"] = time.perf_counter() - route_started
        analysis_sizes: list[tuple[int, int]] = []
        for path in analysis_regions:
            with Image.open(path) as analysis_image:
                analysis_sizes.append(analysis_image.size)
        # CNN 已找到完整文字段落时，不再保留段落内部的行框、单词框
        # 或字符框。每个去重后的完整段落只进行一次 OCR。
        consolidated_text_regions = [
            consolidate_document_text_regions(
                routed["document_text"],
                image_size,
            )
            for routed, image_size in zip(routed_regions, analysis_sizes)
        ]

        # 先 CNN、再 OCR：只裁剪去重后的完整文字段落（例如 T1），
        # 不再把 T1 内部的每一行小框逐个送入 OCR。原 OCR 初始化参数、
        # _recognize_regions 和多语言复核代码均保持不变。
        text_patch_groups: list[list[dict[str, object]]] = []
        for panel_index, (path, text_regions) in enumerate(
            zip(analysis_regions, consolidated_text_regions),
        ):
            patches = crop_document_text_patches(
                path,
                text_regions,
                pipeline_directory,
                f"panel_{panel_index:02d}",
            )
            text_patch_groups.append(patches)
            temporary_paths.extend(
                Path(patch["path"])
                for patch in patches
            )

        flat_patches = [
            patch
            for group in text_patch_groups
            for patch in group
        ]
        self.ocr_diagnostics["panel_patch_counts"] = [len(group) for group in text_patch_groups]
        self.ocr_diagnostics["routed_counts"] = [
            {name: len(items) for name, items in routed.items()}
            for routed in routed_regions
        ]
        self.ocr_diagnostics["ocr_executed"] = bool(flat_patches)
        if flat_patches:
            if self.ocr_owner.ocr is None:
                init_started = time.perf_counter()
                from paddleocr import PaddleOCR

                MODEL_CACHE.mkdir(parents=True, exist_ok=True)
                self.ocr_owner.ocr = PaddleOCR(
                    text_detection_model_name="PP-OCRv6_medium_det",
                    text_recognition_model_name="PP-OCRv6_medium_rec",
                    enable_mkldnn=False,
                    use_doc_orientation_classify=True,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
                self.ocr_diagnostics["model_initialization_s"] += time.perf_counter() - init_started
            predict_started = time.perf_counter()
            patch_records = self._recognize_regions(
                self.ocr_owner.ocr,
                [Path(patch["path"]) for patch in flat_patches],
                self._multilingual_processor(),
                batch_size=self.ocr_batch_size,
            )
            self.ocr_diagnostics["ocr_prediction_and_review_s"] = time.perf_counter() - predict_started
        else:
            patch_records = []

        # OCR bbox 原本相对于各 patch；这里只做坐标平移，使后续原有
        # 文本比较与结果显示继续使用完整框选区域坐标。
        record_index = 0
        all_records: list[list[OCRRecord]] = []
        for patch_group in text_patch_groups:
            panel_records: list[OCRRecord] = []
            for patch in patch_group:
                x, y, _width, _height = patch["bbox"]
                for record in patch_records[record_index]:
                    x1, y1, x2, y2 = record.bbox
                    record.bbox = (
                        x1 + int(x),
                        y1 + int(y),
                        x2 + int(x),
                        y2 + int(y),
                    )
                    panel_records.append(record)
                record_index += 1
            panel_records.sort(
                key=lambda record: (record.bbox[1], record.bbox[0])
            )
            all_records.append(panel_records)

        # CNN 只负责粗筛可能含文字的区域。最终绿色框以 PP-OCRv6
        # 检出的真实文字行为基础，按行距合并成段落，因此不会把候选
        # 区域中的产品图片一起画进文字框。
        paragraph_regions = [
            self._ocr_records_to_paragraph_predictions(records)
            for records in all_records
        ]
        self.ocr_diagnostics['consolidated_region_counts'] = [len(items) for items in consolidated_text_regions]
        self.ocr_diagnostics['paragraph_counts'] = [len(items) for items in paragraph_regions]
        original_text_by_page: dict[
            int,
            list[tuple[float, float, float, float]],
        ] = {
            int(original_mapping["page_index"]): (
                self._map_text_predictions_to_page(
                    original_mapping,
                    paragraph_regions[0],
                    analysis_sizes[0],
                )
            )
        }
        comparison_text_by_page: dict[
            int,
            list[tuple[float, float, float, float]],
        ] = {}
        for match, paragraphs, comparison_analysis_size in zip(
            matches,
            paragraph_regions[1:],
            analysis_sizes[1:],
        ):
            page_index = int(match["page_index"])
            comparison_text_by_page.setdefault(page_index, []).extend(
                self._map_text_predictions_to_page(
                    match,
                    paragraphs,
                    comparison_analysis_size,
                )
            )

        return all_records, original_text_by_page, comparison_text_by_page
    def compare_sample(self, source_input, target_input, *, sample_id='', mode='fusion', config=None, output_dir=None, selection_source='original'):
        result = {
            'schema_version':'1.0', 'sample_id':sample_id, 'mode':mode,
            'source_file':None, 'target_file':None,
            'input_spec':None,
            'config':None,
            'status':{'completed':False,'alignment_success':None,'ocr_success':None,'crashed':False,'error_type':None,'error_message':None,'error_stage':None},
            'timing':{k+'_s':None for k in ('load','preprocess','alignment','visual','ocr','fusion','report','total')},
            'resources':{'peak_ram_mb':None,'peak_vram_mb':None,'ram_scope':'sampled process tree RSS','vram_note':'No reliable per-process VRAM attribution; no whole-GPU estimate used'},
            'detections':[], 'ocr_records':[], 'regions':[],
            'warnings':[],
            'summary':{'detection_count':0,'final_status':None,'fusion_method':'legacy_rule_OR_with_movement_warning' if mode=='fusion' else None},
        }
        self.current_stage = 'validation'
        started = time.perf_counter()
        monitor = ResourceMonitor(); monitor.start()
        try:
            cfg = config or ExperimentConfig(mode=mode)
            cfg.validate()
            if cfg.mode != mode:
                raise InputError('mode and config.mode must agree')
            if selection_source not in {'original', 'rearranged'}:
                raise InputError('selection_source must be original or rearranged')
            source_spec, target_spec = normalize_input(source_input), normalize_input(target_input)
            result.update(source_file=source_spec['path'], target_file=target_spec['path'],
                          config=cfg.to_dict(), input_spec={'source':source_spec, 'target':target_spec,
                                                          'selection_source':selection_source})
            selected = source_spec if selection_source == 'original' else target_spec
            selected_raw = source_input if selection_source == 'original' else target_input
            if Path(selected['path']).suffix.lower() == '.pdf' and (
                selected['selection'] is None or not isinstance(selected_raw, dict) or 'page' not in selected_raw
            ):
                raise InputError('The selected PDF side requires explicit zero-based page and selection ROI')
            self.ocr_batch_size = cfg.ocr_batch_size
            self.ocr_diagnostics = {}
            self.current_stage = 'report'
            output = Path(output_dir) if output_dir else BASE_DIR/'results'/('experiment_'+str(time.time_ns()))
            output.mkdir(parents=True,exist_ok=True)
            temporary_root = BASE_DIR / 'tmp'
            temporary_root.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='smartprint_', dir=temporary_root) as tmp:
                workspace = Path(tmp)
                with self.stage('load',result):
                    self.original_pane = InputPane(source_spec,workspace,'source',cfg)
                    self.compare_pane = InputPane(target_spec,workspace,'target',cfg)
                with self.stage('load',result):
                    if selection_source == 'original':
                        original_region = self.original_pane.render_selection('original')
                    else:
                        template = self.compare_pane.render_selection('template')
                with self.stage('alignment',result):
                    if selection_source == 'original':
                        matches = self._find_all_matches(original_region,self.original_pane,self.compare_pane)
                        original_mapping = {'page_index':self.original_pane.page_index,'rotation':0,'selection':self.original_pane.selection}
                    else:
                        originals = self._find_all_matches(template,self.compare_pane,self.original_pane)
                        if not originals:
                            raise InputError('No original panel passed the existing matching threshold', 'alignment_failure')
                        original_mapping = max(originals,key=lambda m:float(m['score']))
                        matches = self._find_all_matches(template,self.compare_pane,self.compare_pane,seed_page_index=self.compare_pane.page_index,seed_selection=self.compare_pane.selection)
                        self.original_pane.page_index = int(original_mapping['page_index'])
                        self.original_pane.selection = original_mapping['selection']
                    if not matches:
                        raise InputError('No target panel passed the existing matching threshold; select a distinct inner ROI', 'alignment_failure')
                if selection_source == 'rearranged':
                    with self.stage('load',result):
                        original_region = self.original_pane.render_selection('original')
                    with self.stage('preprocess',result):
                        original_region = self._normalize_region_orientation(original_region,int(original_mapping['rotation']),0)
                targets = []
                for index,match in enumerate(matches):
                    with self.stage('load',result):
                        self.compare_pane.page_index = int(match['page_index'])
                        self.compare_pane.selection = match['selection']
                        raw = self.compare_pane.render_selection('target_'+str(index))
                    with self.stage('preprocess',result):
                        targets.append(self._normalize_region_orientation(raw,int(match['rotation']),index))
                with self.stage('load',result):
                    with Image.open(original_region) as im: analyzed_size = im.size
                # Same unwarped, orientation-normalized OCR inputs as the legacy GUI.
                # Geometry and visual evidence never change CNN/OCR input by mode.
                all_records = [[] for _ in range(len(matches)+1)]
                original_text, target_text = {},{}
                if mode in {'ocr','fusion'}:
                    with self.stage('ocr',result):
                        all_records, original_text, target_text = self.recognize_panels([original_region,*targets],original_mapping,matches,workspace)
                    result['status']['ocr_success'] = True if self.ocr_diagnostics.get('ocr_executed', True) else None
                    result['ocr_records'] = [[asdict(r) for r in records] for records in all_records]
                    result['ocr_diagnostics'] = self.ocr_diagnostics
                    if any(not records for records in all_records):
                        result['warnings'].append('One or more panels have no OCR records; text equality is not proof of defect absence.')
                image_results = []
                page_boxes = {}
                for index,(match,target) in enumerate(zip(matches,targets),start=1):
                    with self.stage('load',result):
                        a,b,resize_scale = resize_to_reference(load_image(original_region),load_image(target))
                    with self.stage('alignment',result):
                        aligned,alignment = align_images(a,b,**cfg.alignment_options())
                        alignment['resize_scale'] = resize_scale
                    if mode in {'visual','fusion'}:
                        self.current_stage = 'visual'
                        image_result = compare_images(a,b,output_path=output/f'region_{index:02d}_diff_result.jpg',prepared_pair=(a,aligned,alignment),threshold_value=cfg.difference_threshold,blur_kernel=cfg.blur_kernel,morphology_kernel=cfg.morphology_kernel,morphology_iterations=cfg.morphology_iterations,min_area=cfg.min_area,merge_gap=cfg.merge_gap)
                        for key,value in image_result['timing'].items():
                            if key!='total_s': result['timing'][key]=(result['timing'].get(key) or 0)+value
                    else:
                        image_result={'alignment_offset':alignment,'difference_boxes':[],'image_score':None}
                    image_result['comparison_size']=analyzed_size
                    image_results.append(image_result)
                    result['regions'].append({'match':match,'alignment':alignment,'visual':image_result if mode!='ocr' else None})
                    for box in image_result['difference_boxes']:
                        page_box = self._map_difference_box_to_page(match,box,analyzed_size)
                        page_boxes.setdefault(int(match['page_index']),[]).append(page_box)
                        result['detections'].append({'detection_id':f"visual-{index}-{len(result['detections'])+1}",'type':'visual_difference','bbox':[box['x'],box['y'],box['width'],box['height']],'bbox_space':'aligned_source_roi_pixels_xywh','target_page_bbox_approx':list(page_box),'page':int(match['page_index']),'region_index':index,'confidence':None,'visual_score':image_result['image_score'],'ocr_score':None,'ocr_text_source':None,'ocr_text_target':None,'details':box})
                    del a,b,aligned
                result['status']['alignment_success']=all(r['alignment_offset']['success'] for r in image_results)
                comparisons=list(zip(matches,all_records[1:]))
                with self.stage('fusion',result):
                    if mode=='fusion':
                        decision=self._build_rule_result(all_records[0],comparisons,image_results)
                    elif mode=='visual':
                        movements=[self._analyze_movement(r) for r in image_results]
                        final='ERROR' if result['detections'] else ('WARNING' if any(m['status']!='not_moved' for m in movements) else 'PASS')
                        decision={'final_status':final,'movement_results':movements}
                    else:
                        decision={'final_status':'PASS'}
                    if mode in {'ocr','fusion'}:
                        for index,(match,records) in enumerate(comparisons,start=1):
                            score,differences=self._difference_lines(all_records[0],records)
                            for difference in differences:
                                result['detections'].append({'detection_id':f"ocr-{index}-{len(result['detections'])+1}",'type':'ocr_text_difference','bbox':None,'bbox_space':None,'page':int(match['page_index']),'region_index':index,'confidence':None,'visual_score':None,'ocr_score':score,'ocr_text_source':'\n'.join(r.text for r in all_records[0]),'ocr_text_target':'\n'.join(r.text for r in records),'description':difference})
                            if mode=='ocr' and differences: decision['final_status']='ERROR'
                    result['summary'].update(detection_count=len(result['detections']),final_status=decision['final_status'])
                if not result['status']['alignment_success']:
                    result['status'].update(error_type='alignment_failure',error_stage='alignment',error_message='One or more ROI registrations rejected; see regions. Legacy downstream decision retained.')
                with self.stage('report',result):
                    ocr_text = self._format_multiple_comparisons(all_records[0],comparisons) if mode in {'ocr','fusion'} else ''
                    if self.ocr_diagnostics:
                        classification_lines = [
                            '===== CNN 区域分类 =====',
                            '类别 0 document_text → 原 OCR；类别 1 image_content → 暂不处理',
                            '图片检测使用 CNN 分类前的完整框选版面做 OpenCV 差分。',
                        ]
                        for panel_index, routed in enumerate(self.ocr_diagnostics['routed_counts']):
                            panel_name = '原图' if panel_index == 0 else f'重排区域 {panel_index}'
                            classification_lines.append(
                                f"{panel_name}：CNN文字候选 {routed['document_text']} 个，"
                                f"CNN粗筛区 {self.ocr_diagnostics['consolidated_region_counts'][panel_index]} 个，"
                                f"最终OCR段落 {self.ocr_diagnostics['paragraph_counts'][panel_index]} 个，"
                                f"图片内容 {routed['image_content']} 个"
                            )
                        ocr_text = '\n'.join(classification_lines) + '\n\n' + ocr_text
                    text=self._format_integrated_result(decision,image_results,ocr_text,output) if mode=='fusion' else str(decision) + '\n\n' + ocr_text
                    if result['warnings']:
                        text += '\n\n' + '\n'.join(result['warnings'])
                    (output/'report.txt').write_text(text,encoding='utf-8')
                    result['display']={'original_mapping':original_mapping,'matches':matches,'original_text':original_text,'target_text':target_text,'difference_boxes':page_boxes,'decision':decision,'text':text}
                result['status']['completed']=True
        except Exception as exc:
            message=str(exc)
            kind='algorithm_exception'
            if isinstance(exc,MemoryError) or 'out of memory' in message.lower(): kind='out_of_memory'
            elif isinstance(exc, InputError): kind=exc.error_type
            elif self.current_stage=='validation': kind='unsupported_input'
            elif self.current_stage=='ocr': kind='ocr_failure'
            elif self.current_stage=='alignment': kind='alignment_failure'
            result['status'].update(error_type=kind,error_stage=self.current_stage,error_message=f'{type(exc).__name__}: {exc}')
            if self.current_stage=='ocr': result['status']['ocr_success']=False
        finally:
            for name in ('original_pane', 'compare_pane'):
                pane = getattr(self, name, None)
                if pane is not None:
                    pane.close()
            monitor.stop()
            result['resources']['peak_ram_mb']=monitor.peak
            result['resources']['ram_samples']=monitor.samples
            result['resources']['ram_sampling_interval_s']=0.05
            result['timing']['total_s']=time.perf_counter()-started
        return result

def compare_sample(source_input,target_input,*,sample_id='',mode='fusion',config=None,output_dir=None,selection_source='original'):
    engine=ComparisonEngine()
    try:
        return engine.compare_sample(source_input,target_input,sample_id=sample_id,mode=mode,config=config,output_dir=output_dir,selection_source=selection_source)
    finally: engine.close()
