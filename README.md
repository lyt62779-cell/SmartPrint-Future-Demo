# SmartPrint Future

## 智印未来——人工智能赋能印前质量检测系统

本项目是 Windows 本地桌面 AI 软件，用于辅助复核客户确认单品原稿与工厂加工拼版稿之间的内容一致性。生产稿允许复制、旋转、等比例缩放、平移和重复拼版，但产品内容应保持一致。系统先建立空间对应关系，再组合 Visual 图像差异、OCR 字符级复核与 Spatial Evidence，输出定位证据和差异说明，由质检人员保留最终业务判断权。

## 核心功能

- 双稿导入，支持 PDF 与 PNG/JPG/BMP/WebP/TIFF 图片
- PDF 逐页预览、翻页、缩放、拖动和区域框选
- 在加工拼版稿中自动定位原稿产品单元的旋转、缩放和重复副本
- 几何对齐、图像差异与局部结构变化检测
- PP-OCRv6 中文、英文和数字识别，低置信度/语言冲突区域二次复核
- Visual + OCR + Spatial Evidence Fusion，列出替换、缺少、新增和位置证据
- 差异结果、运行日志和 JSON/图片输出；支持修改后再次复检

## 技术栈

Python、Tkinter/ttk、Pillow、OpenCV、NumPy、PyMuPDF、PaddleOCR 3.7、PaddlePaddle 3.3、PyTorch/torchvision（EfficientNet-B0 区域分类器）。Tkinter 通常随 Windows Python 提供，不通过 pip 安装。

## 系统要求

Windows 10/11；推荐 Python 3.10。CPU 可以运行，但首次 OCR 和拼版比对会较慢；GPU 可选。PaddleOCR 模型首次使用可能联网下载。拼版比对的 EfficientNet 分类器需要 PyTorch/torchvision，默认从 `CNN_CLASSIFIER_PYTHON` 指定的解释器启动独立子进程。

## 环境安装

建议使用 Conda：

```powershell
conda create -n smartprint python=3.10 -y
conda activate smartprint
python -m pip install -r requirements.txt
```

如果不希望主环境安装 PyTorch，可单独创建分类器环境，并在运行前设置：

```powershell
conda create -n smartprint-cnn python=3.10 -y
conda activate smartprint-cnn
python -m pip install torch torchvision
$env:CNN_CLASSIFIER_PYTHON = (Get-Command python).Source
```

GPU 版 PaddlePaddle 请按 Paddle 官方 Windows/CUDA 安装说明选择对应构建；CPU 版无需 CUDA。程序不会上传业务文件到云端。

## 启动程序

首选双击 `run_demo.bat`。命令行备用入口：

```powershell
python .\ocr_demo.py
```

脚本先切换到自身目录，未设置 `CNN_CLASSIFIER_PYTHON` 时使用当前 Python；若当前解释器没有 PyTorch，打开双图比对时会提示需要设置分类器解释器。

## 使用流程

1. 导入客户确认单品原稿（PDF 或图片）。
2. 导入工厂加工拼版稿。
3. 预览页面并在原稿侧框选目标产品单元。
4. 点击自动定位并比对，程序搜索拼版稿中的旋转/缩放/重复副本。
5. 查看几何对齐、Visual、OCR 和 Fusion 结果及差异列表。
6. 人工复核疑似差异，保存输出和日志；修改文件后重新运行复检。

## 目录结构

```text
SmartPrint_Future_Submission_Demo/
├─ run_demo.bat                 Windows 启动脚本
├─ ocr_demo.py                  主 Tkinter GUI 与 OCR 入口
├─ ocr_compare_demo.py          双图区域比对窗口
├─ experiment_core.py           比对执行与结果融合
├─ comparison_algorithms.py     定位、对齐、Visual/OCR 比对规则
├─ image_align.py               几何对齐
├─ image_diff.py                图像差异
├─ multilingual_ocr.py          OCR 语言特征与二次复核
├─ cnn_region_classifier.py     EfficientNet-B0 分类器子进程
├─ cnn_region_pipeline.py       候选区域路由与文字区域合并
├─ region_separator.py          候选区域检测
├─ experiment_config.py         配置与运行环境记录
├─ gpu_runtime.py               Windows GPU DLL 路径配置
├─ model_cache/                 本地模型目录（官方 OCR 模型首次运行下载）
├─ demo_data/
│  ├─ original/.gitkeep         原稿样本占位目录
│  └─ test/.gitkeep             加工稿样本占位目录
├─ output/                      默认输出目录
├─ requirements.txt             精简依赖
└─ README.md                    本说明
```

## 模型与资源

OCR 使用 `PP-OCRv6_medium_det`、`PP-OCRv6_medium_rec`，并按需使用方向分类模型。官方 OCR 权重未放入仓库，首次调用时由 PaddleOCR 下载到 `model_cache/official_models`，需要联网；也可以预先把同名模型放入该目录。EfficientNet-B0 分类器检查点 `model_cache/cnn_region_classifier/best_efficientnet_b0_classifier.pth` 已随提交版保留；其推理解释器由 `CNN_CLASSIFIER_PYTHON` 指定。模型体积较大时请按仓库/比赛平台限制改用外部下载，并在本地保持相同路径。

## Demo 数据

原项目未提供可以确认公开授权的客户原稿/加工拼版稿配对样本。本提交版保留 `demo_data/original` 与 `demo_data/test` 空目录，避免擅自公开企业文件；运行时可选择本地业务文件。公开演示样本需要人工确认脱敏和授权后再放入这两个目录。

## 输出

默认结果写入 `output/` 或界面指定目录，包括比对 JSON、差异掩膜/可视化图片、文字差异和运行日志。输出目录默认不纳入 Git。

## 数据安全

软件在 Windows 本地运行，正式业务文件无需上传公共云服务。公开仓库不应包含真实客户包装稿、订单、合同、人员信息或内部实验材料；发布前请继续检查 `demo_data`、`output` 和模型目录。

## 版本来源

本提交版由正式项目副本整理而来，未修改原项目目录、算法规则或实验数据。
