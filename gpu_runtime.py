"""为 Windows 下 pip 安装的 Paddle GPU 运行库配置 DLL 搜索路径。

PaddlePaddle GPU 3.3.x 会把 CUDA 13 和 cuDNN DLL 安装到当前虚拟环境的
``site-packages/nvidia``，但 Windows 不会自动搜索这些子目录。本模块只把
已经安装在当前环境中的 DLL 目录加入当前 Python 进程，不下载依赖、不导入
Paddle，也不修改 OCR 识别逻辑。
"""

from __future__ import annotations

import os
import site
from pathlib import Path


# ``os.add_dll_directory`` 返回的句柄必须在进程存活期间保留，否则目录可能
# 被移出 DLL 搜索范围。
_DLL_DIRECTORY_HANDLES: list[object] = []
_CONFIGURED_DIRECTORIES: tuple[Path, ...] | None = None


def configure_paddle_gpu_dll_paths() -> tuple[Path, ...]:
    """配置当前虚拟环境内 CUDA/cuDNN DLL 路径，并支持重复安全调用。"""

    global _CONFIGURED_DIRECTORIES
    if _CONFIGURED_DIRECTORIES is not None:
        return _CONFIGURED_DIRECTORIES
    if os.name != "nt":
        _CONFIGURED_DIRECTORIES = ()
        return _CONFIGURED_DIRECTORIES

    candidates: list[Path] = []
    for site_directory in site.getsitepackages():
        nvidia_root = Path(site_directory) / "nvidia"
        candidates.extend(
            (
                nvidia_root / "cu13" / "bin" / "x86_64",
                nvidia_root / "cudnn" / "bin",
            )
        )

    existing: list[Path] = []
    for directory in candidates:
        if not directory.is_dir() or directory in existing:
            continue
        existing.append(directory)
        if hasattr(os, "add_dll_directory"):
            _DLL_DIRECTORY_HANDLES.append(
                os.add_dll_directory(str(directory))
            )

    # 某些第三方库仍通过 PATH 查找其下游 DLL，因此同时只对当前进程追加
    # PATH；不使用 setx，不污染用户或系统的永久环境变量。
    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    prepend = [str(path) for path in existing if str(path) not in path_entries]
    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + path_entries)

    _CONFIGURED_DIRECTORIES = tuple(existing)
    return _CONFIGURED_DIRECTORIES
